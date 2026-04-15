# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker"""

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")

        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))

        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_log_prob_with_module(
        self,
        module: nn.Module,
        micro_batch,
        temperature,
        calculate_entropy=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generic forward helper that mirrors _forward_micro_batch, but runs with the given module.

        Returns:
            entropy: (bs, response_len) or None
            log_probs: (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo
                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(getattr(module, "module", module).config, "vision_config")
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0) if calculate_entropy else None
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)

                    inplace_backward = not calculate_entropy
                    log_probs = verl_F.logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1: -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1: -1]

            else:
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1: -1]
                    entropy = output.entropy[:, -response_length - 1: -1] if calculate_entropy else None
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1: -1, :]
                    log_probs = verl_F.logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_log_prob_with_module(
            module=self.actor_module,
            micro_batch=micro_batch,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
        )

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    def _build_distill_inputs_for_micro_batch(self, micro_batch, tokenizer, selected_indices=None):
        import torch

        prompt_texts = micro_batch.non_tensor_batch["prompt_text"]
        demo_texts = micro_batch.non_tensor_batch["demo_text"]
        wrong_response_texts = micro_batch.non_tensor_batch["wrong_response_text"]
        distill_types = micro_batch.non_tensor_batch["distill_type"]

        if selected_indices is None:
            selected_indices = list(range(len(distill_types)))

        target_input_ids_list = []
        target_attention_mask_list = []
        target_position_ids_list = []
        target_responses_list = []
        target_response_mask_list = []
        selected_distill_types = []
        original_indices = []

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        for idx in selected_indices:
            prompt_text = prompt_texts[idx]
            demo_text = demo_texts[idx]
            wrong_resp_text = wrong_response_texts[idx]
            distill_type = distill_types[idx]

            prefix_text = (
                f"{prompt_text}\n\n"
                f"Correct solution:\n{demo_text}\n\n"
                f"Correctly solve the original question:\n"
            )
            full_text = prefix_text + wrong_resp_text

            encoded_prefix = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
            encoded_full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

            prefix_ids = encoded_prefix["input_ids"][0]
            full_ids = encoded_full["input_ids"][0]
            response_ids = full_ids[prefix_ids.shape[0]:]

            if response_ids.numel() == 0:
                encoded_resp = tokenizer(wrong_resp_text, return_tensors="pt", add_special_tokens=False)
                response_ids = encoded_resp["input_ids"][0]
                full_ids = torch.cat([prefix_ids, response_ids], dim=0)

            attention_mask = torch.ones_like(full_ids, dtype=torch.long)
            position_ids = torch.arange(full_ids.shape[0], dtype=torch.long)
            response_mask = torch.ones_like(response_ids, dtype=torch.long)

            target_input_ids_list.append(full_ids)
            target_attention_mask_list.append(attention_mask)
            target_position_ids_list.append(position_ids)
            target_responses_list.append(response_ids)
            target_response_mask_list.append(response_mask)
            selected_distill_types.append(distill_type)
            original_indices.append(idx)

        if len(target_input_ids_list) == 0:
            return {
                "input_ids": None,
                "attention_mask": None,
                "position_ids": None,
                "responses": None,
                "response_mask": None,
                "distill_types": [],
                "original_indices": [],
            }

        max_input_len = max(x.shape[0] for x in target_input_ids_list)
        max_resp_len = max(x.shape[0] for x in target_responses_list)

        def pad_1d(x, max_len, pad_value):
            if x.shape[0] == max_len:
                return x
            pad = torch.full((max_len - x.shape[0],), pad_value, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        target_input_ids = torch.stack([pad_1d(x, max_input_len, pad_id) for x in target_input_ids_list], dim=0)
        target_attention_mask = torch.stack([pad_1d(x, max_input_len, 0) for x in target_attention_mask_list], dim=0)
        target_position_ids = torch.stack([pad_1d(x, max_input_len, 0) for x in target_position_ids_list], dim=0)
        target_responses = torch.stack([pad_1d(x, max_resp_len, pad_id) for x in target_responses_list], dim=0)
        target_response_mask = torch.stack([pad_1d(x, max_resp_len, 0) for x in target_response_mask_list], dim=0)

        return {
            "input_ids": target_input_ids,
            "attention_mask": target_attention_mask,
            "position_ids": target_position_ids,
            "responses": target_responses,
            "response_mask": target_response_mask,
            "distill_types": selected_distill_types,
            "original_indices": original_indices,
        }

    @GPUMemoryLogger(role="dp actor distill", logger=logger)
    def update_policy_distill(self, data: DataProto, teacher_actor, student_base_actor, tokenizer):
        import torch
        import torch.distributed as dist

        self.actor_module.train()

        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
        ]
        non_tensor_select_keys = [
            "distill_type",
            "demo_text",
            "wrong_response_text",
            "prompt_text",
            "uid",
        ]

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}

        sdft_weight = data.meta_info.get("sdft_weight", 1.0)
        icl_opd_weight = data.meta_info.get("icl_opd_weight", 1.0)
        distill_temperature = data.meta_info.get("distill_temperature", 1.0)

        loss_agg_mode = self.config.get("loss_agg_mode", "token-mean")
        entropy_coeff = self.config.get("entropy_coeff", 0.0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        temperature = data.meta_info.get("temperature", 1.0)

        # ===== new configs =====
        advantage_normalize = self.config.get("distill_advantage_normalize", True)
        advantage_clip = self.config.get("distill_advantage_clip", 1.0)
        anchor_coef = self.config.get("distill_anchor_coef", 0.5)
        anchor_loss_type = self.config.get("distill_anchor_loss_type", "mse")
        logprob_gap_clip = self.config.get("distill_logprob_gap_clip", None)
        # =======================

        device = get_device_id()
        rank = dist.get_rank() if dist.is_initialized() else 0

        def gather_tensor_sub_batch(batch_dict, indices):
            return {
                "input_ids": batch_dict["input_ids"][indices],
                "attention_mask": batch_dict["attention_mask"][indices],
                "position_ids": batch_dict["position_ids"][indices],
                "responses": batch_dict["responses"][indices],
                "response_mask": batch_dict["response_mask"][indices],
            }

        def gather_tensor_sub_batch_or_dummy(batch_dict, indices):
            if len(indices) > 0:
                return gather_tensor_sub_batch(batch_dict, indices), False

            dummy_idx = [0]
            sub_batch = {
                "input_ids": batch_dict["input_ids"][dummy_idx],
                "attention_mask": batch_dict["attention_mask"][dummy_idx],
                "position_ids": batch_dict["position_ids"][dummy_idx],
                "responses": batch_dict["responses"][dummy_idx],
                "response_mask": torch.zeros_like(batch_dict["response_mask"][dummy_idx]),
            }
            return sub_batch, True

        def forward_target_module(target_module, sub_batch, target_temperature):
            target_module.eval()
            with torch.no_grad():
                entropy, log_probs = self._forward_log_prob_with_module(
                    module=target_module,
                    micro_batch=sub_batch,
                    temperature=target_temperature,
                    calculate_entropy=False,
                )
            return entropy, log_probs

        def zero_loss_from_tensors(*tensors):
            loss = None
            for t in tensors:
                if t is not None:
                    z = t.sum() * 0.0
                    loss = z if loss is None else loss + z
            if loss is None:
                loss = torch.tensor(0.0, device=device)
            return loss

        def _to_scalar(x, name="metric"):
            import numpy as np
            if torch.is_tensor(x):
                if x.numel() != 1:
                    raise ValueError(f"{name} must be scalar tensor, got shape={tuple(x.shape)}")
                return float(x.detach().item())
            if isinstance(x, np.ndarray):
                if x.size != 1:
                    raise ValueError(f"{name} must be scalar ndarray, got shape={x.shape}")
                return float(x.item())
            if isinstance(x, (list, tuple)):
                raise ValueError(f"{name} must be scalar, got {type(x)}")
            return float(x)

        def normalize_and_clip_advantages(advantages: torch.Tensor, mask: torch.Tensor):
            valid = mask > 0
            if valid.any():
                adv_valid = advantages[valid]
                if logprob_gap_clip is not None:
                    adv_valid = torch.clamp(adv_valid, min=-logprob_gap_clip, max=logprob_gap_clip)
                    advantages = torch.clamp(advantages, min=-logprob_gap_clip, max=logprob_gap_clip)

                if advantage_normalize:
                    adv_mean = adv_valid.mean()
                    adv_std = adv_valid.std(unbiased=False).clamp_min(1e-6)
                    advantages = (advantages - adv_mean) / adv_std

                if advantage_clip is not None:
                    advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)

            return advantages

        def compute_anchor_loss(actor_log_prob, base_log_prob, response_mask_local):
            if anchor_coef == 0:
                return actor_log_prob.sum() * 0.0

            if anchor_loss_type == "mse":
                anchor_mat = (actor_log_prob - base_log_prob.detach()) ** 2
            elif anchor_loss_type == "kl_token":
                # sampled-token surrogate, not true full-vocab KL
                # keeps actor close to base on the sampled tokens
                anchor_mat = torch.exp(base_log_prob.detach()) * (base_log_prob.detach() - actor_log_prob)
            else:
                raise ValueError(f"Unknown distill_anchor_loss_type: {anchor_loss_type}")

            anchor_loss = agg_loss(
                loss_mat=anchor_mat,
                loss_mask=response_mask_local,
                loss_agg_mode=loss_agg_mode,
            )
            return anchor_loss

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                    gradient_accumulation = len(micro_batches)

                    if dist.is_initialized():
                        local_mb_num = torch.tensor([gradient_accumulation], device=device, dtype=torch.int64)
                        min_mb_num = local_mb_num.clone()
                        max_mb_num = local_mb_num.clone()
                        dist.all_reduce(min_mb_num, op=dist.ReduceOp.MIN)
                        dist.all_reduce(max_mb_num, op=dist.ReduceOp.MAX)
                        if min_mb_num.item() != max_mb_num.item():
                            raise RuntimeError(
                                f"[Rank {rank}] Inconsistent number of micro-batches across ranks "
                                f"when use_dynamic_bsz=True: local={local_mb_num.item()}, "
                                f"global_min={min_mb_num.item()}, global_max={max_mb_num.item()}."
                            )
                else:
                    gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                mini_batch_metrics_accum = {}
                mini_batch_metric_count = 0
                last_pg_loss = None

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(device)
                    micro_batch_metrics = {}

                    local_distill_types = list(micro_batch.non_tensor_batch["distill_type"])
                    sdft_indices = [i for i, t in enumerate(local_distill_types) if t == "sdft"]
                    icl_indices = [i for i, t in enumerate(local_distill_types) if t == "icl_opd"]

                    type_specs = [
                        ("sdft", sdft_indices, student_base_actor.actor_module, sdft_weight),
                        ("icl_opd", icl_indices, teacher_actor.actor_module, icl_opd_weight),
                    ]

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = micro_batch.batch["input_ids"].shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / gradient_accumulation

                    micro_total_loss = None
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    for distill_type_name, local_indices, target_module, loss_weight in type_specs:
                        local_has_type = torch.tensor(
                            [1 if len(local_indices) > 0 else 0], device=device, dtype=torch.int64
                        )
                        if dist.is_initialized():
                            dist.all_reduce(local_has_type, op=dist.ReduceOp.MAX)
                        global_has_type = bool(local_has_type.item())

                        if not global_has_type:
                            continue

                        actor_sub_batch, actor_is_dummy = gather_tensor_sub_batch_or_dummy(
                            micro_batch.batch, local_indices
                        )
                        response_mask = actor_sub_batch["response_mask"].float()

                        calculate_entropy = entropy_coeff != 0
                        actor_entropy, actor_log_prob = self._forward_micro_batch(
                            actor_sub_batch,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )

                        _, base_log_prob = forward_target_module(
                            student_base_actor.actor_module,
                            actor_sub_batch,
                            target_temperature=distill_temperature,
                        )

                        target_inputs = self._build_distill_inputs_for_micro_batch(
                            micro_batch,
                            tokenizer,
                            selected_indices=local_indices if len(local_indices) > 0 else [0],
                        )

                        target_input_ids = target_inputs["input_ids"].to(device)
                        target_attention_mask = target_inputs["attention_mask"].to(device)
                        target_position_ids = target_inputs["position_ids"].to(device)
                        target_responses = target_inputs["responses"].to(device)
                        target_response_mask = target_inputs["response_mask"].to(device).float()

                        if actor_is_dummy:
                            target_response_mask = torch.zeros_like(target_response_mask)

                        target_sub_batch = {
                            "input_ids": target_input_ids,
                            "attention_mask": target_attention_mask,
                            "position_ids": target_position_ids,
                            "responses": target_responses,
                        }

                        _, target_log_prob = forward_target_module(
                            target_module,
                            target_sub_batch,
                            target_temperature=distill_temperature,
                        )

                        resp_len = min(
                            actor_log_prob.shape[1],
                            base_log_prob.shape[1],
                            target_log_prob.shape[1],
                            response_mask.shape[1],
                            target_response_mask.shape[1],
                        )

                        actor_log_prob = actor_log_prob[:, :resp_len]
                        base_log_prob = base_log_prob[:, :resp_len]
                        target_log_prob = target_log_prob[:, :resp_len]
                        response_mask_local = response_mask[:, :resp_len] * target_response_mask[:, :resp_len]

                        if actor_entropy is not None:
                            actor_entropy = actor_entropy[:, :resp_len]

                        valid_tokens = response_mask_local.sum()

                        if valid_tokens.item() <= 0:
                            type_loss = zero_loss_from_tensors(
                                actor_log_prob,
                                base_log_prob,
                                target_log_prob,
                                actor_entropy if actor_entropy is not None else None,
                            )
                            pg_metrics = {
                                "actor/pg_clipfrac": 0.0,
                                "actor/ppo_kl": 0.0,
                                "actor/pg_clipfrac_lower": 0.0,
                            }
                            pg_loss = type_loss
                            anchor_loss = type_loss
                        else:
                            old_log_prob = base_log_prob.detach()
                            advantages = target_log_prob.detach() - base_log_prob.detach()
                            advantages = normalize_and_clip_advantages(advantages, response_mask_local)

                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=actor_log_prob,
                                advantages=advantages,
                                response_mask=response_mask_local,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=None,
                            )

                            anchor_loss = compute_anchor_loss(
                                actor_log_prob=actor_log_prob,
                                base_log_prob=base_log_prob,
                                response_mask_local=response_mask_local,
                            )

                            type_loss = pg_loss + anchor_coef * anchor_loss

                            if entropy_coeff != 0:
                                entropy_loss = agg_loss(
                                    loss_mat=actor_entropy,
                                    loss_mask=response_mask_local,
                                    loss_agg_mode=loss_agg_mode,
                                )
                                type_loss = type_loss - entropy_loss * entropy_coeff

                        scaled_type_loss = type_loss * loss_scale_factor * loss_weight
                        micro_total_loss = (
                            scaled_type_loss if micro_total_loss is None else micro_total_loss + scaled_type_loss
                        )

                        for k, v in pg_metrics.items():
                            micro_batch_metrics[k] = (
                                micro_batch_metrics.get(k, 0.0) + v * loss_weight * loss_scale_factor
                            )

                        if valid_tokens.item() > 0:
                            micro_batch_metrics[f"actor/{distill_type_name}_pg_loss"] = (
                                pg_loss.detach().item() * loss_weight * loss_scale_factor
                            )
                            micro_batch_metrics[f"actor/{distill_type_name}_anchor_loss"] = (
                                anchor_loss.detach().item() * loss_weight * loss_scale_factor
                            )
                            micro_batch_metrics[f"actor/{distill_type_name}_adv_mean"] = (
                                ((advantages * response_mask_local).sum() / response_mask_local.sum()).detach().item()
                            )
                            micro_batch_metrics[f"actor/{distill_type_name}_adv_abs_mean"] = (
                                ((advantages.abs() * response_mask_local).sum() / response_mask_local.sum()).detach().item()
                            )
                            last_pg_loss = pg_loss

                    if micro_total_loss is None:
                        micro_total_loss = torch.tensor(0.0, device=device)

                    if self.scaler is not None:
                        self.scaler.scale(micro_total_loss).backward()
                    else:
                        micro_total_loss.backward()

                    if last_pg_loss is not None:
                        micro_batch_metrics["actor/pg_loss"] = last_pg_loss.detach().item() * loss_scale_factor

                    sanitized_micro_batch_metrics = {
                        k: _to_scalar(v, k) for k, v in micro_batch_metrics.items()
                    }
                    append_to_dict(metrics, sanitized_micro_batch_metrics)

                    for k, v in sanitized_micro_batch_metrics.items():
                        mini_batch_metrics_accum[k] = mini_batch_metrics_accum.get(k, 0.0) + v
                    mini_batch_metric_count += 1

                # ===== moved optimizer step outside micro-batch loop =====
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": _to_scalar(grad_norm, "actor/grad_norm")})

                if mini_batch_metric_count > 0:
                    avg_grad_norm = _to_scalar(grad_norm, "actor/grad_norm")
                    _ = avg_grad_norm  # keep for symmetry/debug if needed
                # ========================================================

        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
         # Include base model log probs for corrected reward computation
        # These are computed when actor_rollout_ref.model.base_model_path and
        # actor_rollout_ref.ref.model.base_model_path are both specified
        if "base_log_prob" in data.batch.keys():
            select_keys.append("base_log_prob")
        if "base_ref_log_prob" in data.batch.keys():
            select_keys.append("base_ref_log_prob")
        # Include ref_log_prob for only_reverse_kl_advantages mode
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include opd_teacher for multi-teacher distillation
        if "opd_teacher" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("opd_teacher")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # only use reverse KL for advantages if only_reverse_kl_advantages is True
                    if self.config.policy_loss.only_reverse_kl_advantages:
                        # Corrected reverse KL with base model normalization if base log probs are available
                        # Formula: (log_prob_actor - log_prob_ref) - (log_prob_actor_base - log_prob_ref_base)
                        # This removes the base model bias from both actor and ref models
                        if "base_log_prob" in model_inputs and "base_ref_log_prob" in model_inputs:
                            lambda_vals = self.config.policy_loss.lambda_vals

                            if self.config.policy_loss.multi_teacher_distill:
                                #### multi-teacher distillation ####
                                if "opd_teacher" in model_inputs:
                                    opd_teacher = model_inputs["opd_teacher"]
                                    batch_size = old_log_prob.shape[0]

                                    reverse_kl = torch.zeros_like(old_log_prob)

                                    for i in range(batch_size):
                                        teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                                        # TODO: need to improve the logic here
                                        if teacher_type == "math":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        elif teacher_type == "code":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["base_ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        else:
                                            reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                else:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                #### multi-teacher distillation ####
                            else:
                                #### single-teacher distillation ####
                                reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                                reward_correction = model_inputs["ref_log_prob"] - model_inputs["base_log_prob"]

                                if lambda_vals == 1.0:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                else:
                                    reverse_kl = reverse_kl - reward_correction * lambda_vals
                                #### single-teacher distillation ####
                        else:
                            # Standard reverse KL: log(π_actor / π_ref) = log_prob_actor - log_prob_ref
                            reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                        advantages = (- (reverse_kl))
                   
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        if "actor/sdft_pg_loss" in metrics:
            print("[update_policy_distill] actor/sdft_pg_loss type:", type(metrics["actor/sdft_pg_loss"]))
            print("[update_policy_distill] actor/sdft_pg_loss len:", len(metrics["actor/sdft_pg_loss"]))
            print("[update_policy_distill] actor/sdft_pg_loss first3:", metrics["actor/sdft_pg_loss"][:3])
            print("[update_policy_distill] first elem type:", type(metrics["actor/sdft_pg_loss"][0]))

        self.actor_optimizer.zero_grad()
        return metrics
