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
from verl.trainer.ppo.core_algos import (
    agg_loss,
    build_mpopd_target,
    compute_mpopd_kl_loss,
    get_policy_loss_fn,
    kl_penalty,
)
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
from verl.utils.torch_functional import logprobs_from_logits, topk_logprobs_from_logits

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def add_masked_advantage_metrics(metrics, advantages, response_mask, prefix: str):
    """
    advantages: [bsz, response_len] or [bsz, response_len, k]
    response_mask: [bsz, response_len]
    """
    with torch.no_grad():
        mask = response_mask.bool()
        adv = advantages.detach().float()
        if adv.dim() == 3:
            adv = adv.sum(dim=-1)
        valid_adv = adv[mask]

        if valid_adv.numel() == 0:
            return

        metrics[f"{prefix}_mean"] = valid_adv.mean().item()
        metrics[f"{prefix}_std"] = valid_adv.std(unbiased=False).item()
        metrics[f"{prefix}_max"] = valid_adv.max().item()
        metrics[f"{prefix}_min"] = valid_adv.min().item()
        metrics[f"{prefix}_abs_mean"] = valid_adv.abs().mean().item()
        metrics[f"{prefix}_positive_frac"] = (valid_adv > 0).float().mean().item()
        metrics[f"{prefix}_negative_frac"] = (valid_adv < 0).float().mean().item()


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

    @staticmethod
    def _append_ref_select_keys(select_keys: list[str], batch_keys, use_ref_inputs: bool = True) -> list[str]:
        if use_ref_inputs and "ref_input_ids" in batch_keys:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
            if "ref_responses" in batch_keys:
                select_keys.append("ref_responses")
        return select_keys

    @staticmethod
    def _response_length_for_forward(micro_batch, use_ref_inputs: bool) -> int:
        if use_ref_inputs and "ref_responses" in micro_batch:
            return micro_batch["ref_responses"].size(-1)
        return micro_batch["responses"].size(-1)

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        return_topk=False,
        top_k=None,
        return_selected_log_probs=False,
        return_topk_log_probs=False,
        return_topk_with_log_probs=False,
        use_ref_inputs: bool = False,
    ):
        response_length = self._response_length_for_forward(micro_batch, use_ref_inputs)
        selected_ids = micro_batch.get("union_topk_ids", None)
        selected_mask = micro_batch.get("union_topk_mask", None)
        student_topk_ids = micro_batch.get("student_topk_ids", None) if return_topk_log_probs else None

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if use_ref_inputs and "ref_input_ids" in micro_batch:
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            log_probs = None
            topk_ids = None
            selected_log_probs = None
            topk_log_probs = None

            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1),
                    attention_mask,
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(
                            rearrange(position_ids, "c b s ... -> (b s) c ..."),
                            indices,
                        )
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo
                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids,
                        attention_mask,
                        position_ids,
                        cu_seqlens,
                        multi_modal_inputs,
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                selected_ids_rmpad = None
                selected_mask_rmpad = None
                student_topk_ids_rmpad = None

                if return_topk_log_probs:
                    topk_m = student_topk_ids.size(-1)
                    full_student_topk_ids = torch.zeros(
                        batch_size,
                        seqlen,
                        topk_m,
                        dtype=student_topk_ids.dtype,
                        device=student_topk_ids.device,
                    )
                    full_student_topk_ids[:, -response_length - 1 : -1, :] = student_topk_ids
                    student_topk_ids_rmpad = index_first_axis(
                        rearrange(full_student_topk_ids, "b s m -> (b s) m"),
                        indices,
                    )

                if return_selected_log_probs:
                    selected_m = selected_ids.size(-1)

                    full_selected_ids = torch.zeros(
                        batch_size,
                        seqlen,
                        selected_m,
                        dtype=selected_ids.dtype,
                        device=selected_ids.device,
                    )
                    full_selected_ids[:, -response_length - 1 : -1, :] = selected_ids

                    if selected_mask is None:
                        full_selected_mask = torch.zeros(
                            batch_size,
                            seqlen,
                            selected_m,
                            dtype=torch.bool,
                            device=selected_ids.device,
                        )
                        full_selected_mask[:, -response_length - 1 : -1, :] = True
                    else:
                        full_selected_mask = torch.zeros(
                            batch_size,
                            seqlen,
                            selected_m,
                            dtype=torch.bool,
                            device=selected_mask.device,
                        )
                        full_selected_mask[:, -response_length - 1 : -1, :] = selected_mask.bool()

                    selected_ids_rmpad = index_first_axis(
                        rearrange(full_selected_ids, "b s m -> (b s) m"),
                        indices,
                    )
                    selected_mask_rmpad = index_first_axis(
                        rearrange(full_selected_mask, "b s m -> (b s) m"),
                        indices,
                    )

                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config,
                        "vision_config",
                    )

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

                    if return_selected_log_probs:
                        selected_ids_rmpad_t, _, _ = ulysses_pad_and_slice_inputs(
                            selected_ids_rmpad.transpose(0, 1),
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        selected_mask_rmpad_t, _, _ = ulysses_pad_and_slice_inputs(
                            selected_mask_rmpad.transpose(0, 1).long(),
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        selected_ids_rmpad = selected_ids_rmpad_t.transpose(0, 1)
                        selected_mask_rmpad = selected_mask_rmpad_t.transpose(0, 1).bool()

                    if return_topk_log_probs:
                        student_topk_ids_rmpad_t, _, _ = ulysses_pad_and_slice_inputs(
                            student_topk_ids_rmpad.transpose(0, 1),
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        student_topk_ids_rmpad = student_topk_ids_rmpad_t.transpose(0, 1)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                use_fused = (
                    self.use_fused_kernels
                    and not return_topk
                    and not return_selected_log_probs
                    and not return_topk_log_probs
                    and not return_topk_with_log_probs
                )
                extra_args = {}

                if use_fused:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if use_fused:
                    log_probs_rmpad = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)

                    if return_topk or return_topk_with_log_probs:
                        topk_ids_rmpad = torch.topk(logits_rmpad, k=top_k, dim=-1).indices

                    if return_topk_with_log_probs:
                        topk_log_probs_rmpad = topk_logprobs_from_logits(
                            logits_rmpad,
                            topk_ids_rmpad.long(),
                        )

                    if return_selected_log_probs:
                        selected_logits_rmpad = torch.gather(
                            logits_rmpad,
                            dim=-1,
                            index=selected_ids_rmpad.long(),
                        )
                        selected_logits_rmpad = selected_logits_rmpad.masked_fill(
                            ~selected_mask_rmpad.bool(),
                            torch.finfo(selected_logits_rmpad.dtype).min,
                        )
                        selected_log_probs_rmpad = torch.log_softmax(
                            selected_logits_rmpad.float(),
                            dim=-1,
                        ).to(logits_rmpad.dtype)

                    if return_topk_log_probs:
                        topk_log_probs_rmpad = topk_logprobs_from_logits(
                            logits_rmpad,
                            student_topk_ids_rmpad.long(),
                        )

                    if (
                        not return_selected_log_probs
                        and not return_topk_log_probs
                        and not return_topk_with_log_probs
                    ):
                        inplace_backward = not calculate_entropy and not return_topk
                        log_probs_rmpad = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits,
                                logits_rmpad,
                            )

                if self.use_ulysses_sp:
                    if (
                        not return_selected_log_probs
                        and not return_topk_log_probs
                        and not return_topk_with_log_probs
                    ):
                        log_probs_rmpad = gather_outputs_and_unpad(
                            log_probs_rmpad,
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

                    if return_topk or return_topk_with_log_probs:
                        topk_ids_rmpad = gather_outputs_and_unpad(
                            topk_ids_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                    if return_topk_with_log_probs:
                        topk_log_probs_rmpad = gather_outputs_and_unpad(
                            topk_log_probs_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                    if return_selected_log_probs:
                        selected_log_probs_rmpad = gather_outputs_and_unpad(
                            selected_log_probs_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                    if return_topk_log_probs:
                        topk_log_probs_rmpad = gather_outputs_and_unpad(
                            topk_log_probs_rmpad,
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
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]

                if (
                    not return_selected_log_probs
                    and not return_topk_log_probs
                    and not return_topk_with_log_probs
                ):
                    full_log_probs = pad_input(
                        hidden_states=log_probs_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

                if return_topk or return_topk_with_log_probs:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]

                if return_topk_with_log_probs:
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

                if return_selected_log_probs:
                    full_selected_log_probs = pad_input(
                        hidden_states=selected_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    selected_log_probs = full_selected_log_probs[:, -response_length - 1 : -1, :]

                if return_topk_log_probs:
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:
                use_fused = (
                    self.use_fused_kernels
                    and not return_topk
                    and not return_selected_log_probs
                    and not return_topk_log_probs
                    and not return_topk_with_log_probs
                )
                extra_args = {}

                if use_fused:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if use_fused:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]

                    if return_topk or return_topk_with_log_probs:
                        topk_ids = torch.topk(logits, k=top_k, dim=-1).indices

                    if return_topk_with_log_probs:
                        flat_logits = logits.reshape(-1, logits.size(-1))
                        flat_topk_ids = topk_ids.reshape(-1, topk_ids.size(-1))
                        topk_log_probs = topk_logprobs_from_logits(
                            flat_logits,
                            flat_topk_ids.long(),
                        ).view_as(topk_ids)

                    if return_selected_log_probs:
                        selected_logits = torch.gather(
                            logits,
                            dim=-1,
                            index=selected_ids.long(),
                        )

                        if selected_mask is not None:
                            selected_logits = selected_logits.masked_fill(
                                ~selected_mask.bool(),
                                torch.finfo(selected_logits.dtype).min,
                            )

                        selected_log_probs = torch.log_softmax(
                            selected_logits.float(),
                            dim=-1,
                        ).to(logits.dtype)
                    elif return_topk_log_probs:
                        flat_logits = logits.reshape(-1, logits.size(-1))
                        flat_topk_ids = student_topk_ids.reshape(-1, student_topk_ids.size(-1))
                        topk_log_probs = topk_logprobs_from_logits(flat_logits, flat_topk_ids.long()).view_as(
                            student_topk_ids
                        )
                    elif not return_topk_with_log_probs:
                        log_probs = logprobs_from_logits(
                            logits,
                            micro_batch["responses"],
                        )

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(
                                verl_F.entropy_from_logits,
                                logits,
                            )

            if return_topk_with_log_probs:
                return entropy, log_probs, topk_ids, topk_log_probs

            if return_topk_log_probs:
                return entropy, log_probs, topk_log_probs

            if return_selected_log_probs:
                return entropy, log_probs, selected_log_probs

            if return_topk:
                return entropy, log_probs, topk_ids

            return entropy, log_probs

    def _optimizer_step(self, lr: float | None = None, lr_scale: float | None = None):
        assert self.config.grad_clip is not None
        assert not (lr is not None and lr_scale is not None), "lr and lr_scale cannot be set at the same time"

        old_lrs = [group["lr"] for group in self.actor_optimizer.param_groups]

        if lr is not None:
            for group in self.actor_optimizer.param_groups:
                group["lr"] = lr
        elif lr_scale is not None:
            for group in self.actor_optimizer.param_groups:
                group["lr"] = group["lr"] * lr_scale

        try:
            if self.scaler is not None:
                self.scaler.unscale_(self.actor_optimizer)

            if isinstance(self.actor_module, FSDP):
                grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
            elif isinstance(self.actor_module, FSDPModule):
                grad_norm = fsdp2_clip_grad_norm_(
                    self.actor_module.parameters(),
                    max_norm=self.config.grad_clip,
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_module.parameters(),
                    max_norm=self.config.grad_clip,
                )

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

        finally:    
            for group, old_lr in zip(self.actor_optimizer.param_groups, old_lrs):
                group["lr"] = old_lr


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys()
        use_ref_inputs = data.meta_info.get("use_ref_inputs", has_ref_input_ids)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys = self._append_ref_select_keys(select_keys, data.batch.keys(), use_ref_inputs=use_ref_inputs)
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
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    use_ref_inputs=use_ref_inputs,
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


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_topk_ids(self, data: DataProto, top_k: int) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys = self._append_ref_select_keys(select_keys, data.batch.keys())

        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_lst = []
        entropy_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                entropy, _, topk_ids = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=True,
                    return_topk=True,
                    top_k=top_k,
                )

            topk_lst.append(topk_ids)
            entropy_lst.append(entropy)

        topk_ids = torch.concat(topk_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            topk_ids = restore_dynamic_batch(topk_ids, batch_idx_list)
            entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return entropys, topk_ids

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_topk_ids_and_log_probs(self, data: DataProto, top_k: int):
        """Single forward: student top-k ids + log pi on those ids, shapes (B,T,K)."""
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys = self._append_ref_select_keys(select_keys, data.batch.keys())

        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_lst = []
        topk_lp_lst = []
        entropy_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                entropy, _, topk_ids, topk_log_probs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=True,
                    return_topk_with_log_probs=True,
                    top_k=top_k,
                )

            topk_lst.append(topk_ids)
            topk_lp_lst.append(topk_log_probs)
            entropy_lst.append(entropy)

        topk_ids = torch.concat(topk_lst, dim=0)
        topk_log_probs = torch.concat(topk_lp_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            topk_ids = restore_dynamic_batch(topk_ids, batch_idx_list)
            topk_log_probs = restore_dynamic_batch(topk_log_probs, batch_idx_list)
            entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return entropys, topk_ids, topk_log_probs


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_selected_log_probs(self, data: DataProto) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys()

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "union_topk_ids",
            "union_topk_mask",
        ]
        select_keys = self._append_ref_select_keys(select_keys, data.batch.keys())

        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(
            batch_keys=select_keys,
            non_tensor_batch_keys=non_tensor_select_keys,
        )

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, _, selected_log_probs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    return_selected_log_probs=True,
                )

            log_probs_lst.append(selected_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_topk_log_probs_on_ids(self, data: DataProto) -> torch.Tensor:
        """Compute true log pi(token) on fixed student top-k ids, shape (B, T, K)."""
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys()

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "student_topk_ids",
        ]
        select_keys = self._append_ref_select_keys(select_keys, data.batch.keys())

        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(
            batch_keys=select_keys,
            non_tensor_batch_keys=non_tensor_select_keys,
        )

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, _, topk_log_probs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    return_topk_log_probs=True,
                )

            log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_grpo(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]

        if self.config.use_kl_loss and "base_log_probs" in data.batch.keys():
            select_keys.append("base_log_probs")

        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        mini_batches = [
            mb for mb in mini_batches
            if mb.batch.batch_size[0] == self.config.ppo_mini_batch_size
        ]

        # ================ algorithm config ================
        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        policy_loss_fn = get_policy_loss_fn(loss_mode)

        grpo_lr_scale = self.config.get('grpo_lr_scale', 1.0)
        print('[DEBUG] grpo_lr_scale: ', grpo_lr_scale)
        # ================ algorithm config ================

        metrics = {}

        for epoch_i in range(self.config.ppo_epochs):
            for mini_i, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                self._dbg_train_eval_once = False

                for micro_i, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())

                    micro_batch_metrics = {}

                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    add_masked_advantage_metrics(
                        micro_batch_metrics,
                        advantages=advantages,
                        response_mask=response_mask,
                        prefix="actor/grpo_advantages",
                    )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    calculate_entropy = entropy_coeff != 0

                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

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

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                        micro_batch_metrics["actor/grpo_entropy_loss"] = entropy_loss.detach().item() * loss_scale_factor
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss and "base_log_probs" in model_inputs:
                        base_log_prob = model_inputs["base_log_probs"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=base_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/grpo_kl_loss"] = kl_loss.detach().item() * loss_scale_factor

                    loss = policy_loss * loss_scale_factor

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/grpo_pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step(lr_scale=grpo_lr_scale)
                append_to_dict(metrics, {"actor/grpo_grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_opd(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "teacher_log_probs",
            "teacher_base_log_probs",
            "old_log_probs",
            "base_log_probs",
        ]

        has_topk_opd = "student_topk_ids" in data.batch.keys()
        if has_topk_opd:
            select_keys.append("student_topk_ids")
        if "cross_token_opd_mask" in data.batch.keys():
            select_keys.append("cross_token_opd_mask")
        has_pace_token_lambda = "pace_token_lambda" in data.batch.keys()
        if has_pace_token_lambda:
            select_keys.append("pace_token_lambda")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        # mini_batches = [
        #     mb for mb in mini_batches
        #     if mb.batch.batch_size[0] == self.config.ppo_mini_batch_size
        # ]

        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode
        policy_loss_fn = get_policy_loss_fn(loss_mode)

        opd_lr_scale = self.config.get("opd_lr_scale", 0.2)
        print("[DEBUG] opd_lr_scale: ", opd_lr_scale)

        lambda_vals = self.config.policy_loss.get("lambda_vals", 1.0)

        metrics = {}
        opd_weight = 1.0

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    if has_topk_opd:
                        padded_max_seq_len = mini_batch.batch["attention_mask"].shape[-1]
                        topk_cap = getattr(self.config, "ppo_topk_max_token_len_per_gpu", 0)
                        if not topk_cap or topk_cap <= 0:
                            topk_cap = max(4096, self.config.ppo_max_token_len_per_gpu // 4)
                        topk_cap = max(topk_cap, padded_max_seq_len)
                        max_token_len = min(
                            max_token_len, topk_cap * self.ulysses_sequence_parallel_size
                        )
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    actual_mini_batch_size = len(mini_batch)
                    self.gradient_accumulation = actual_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                did_backward = False

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}

                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    response_mask = model_inputs["response_mask"]
                    teacher_log_probs = model_inputs["teacher_log_probs"].detach()
                    teacher_base_log_probs = model_inputs["teacher_base_log_probs"].detach()
                    old_log_prob = model_inputs["old_log_probs"].detach()
                    base_log_probs = model_inputs["base_log_probs"].detach()

                    anchor_delta = old_log_prob - base_log_probs
                    if has_pace_token_lambda:
                        token_lambda = model_inputs["pace_token_lambda"].detach()
                        advantages = teacher_log_probs - teacher_base_log_probs - token_lambda * anchor_delta
                    else:
                        advantages = teacher_log_probs - teacher_base_log_probs - lambda_vals * anchor_delta
                    # advantages = teacher_log_probs - teacher_base_log_probs   # extreme update
                    advantages = advantages.detach()

                    policy_mask = response_mask
                    if "cross_token_opd_mask" in model_inputs:
                        cross_token_mask = model_inputs["cross_token_opd_mask"].to(dtype=response_mask.dtype)
                        policy_mask = response_mask * cross_token_mask
                        micro_batch_metrics["actor/opd_cross_token_mask_ratio"] = (
                            cross_token_mask.sum() / response_mask.sum().clamp(min=1)
                        ).item()
                    if has_pace_token_lambda:
                        micro_batch_metrics["actor/pace_token_lambda_mean"] = (
                            token_lambda * policy_mask
                        ).sum().div(policy_mask.sum().clamp(min=1)).item()

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    calculate_entropy = entropy_coeff != 0

                    if has_topk_opd:
                        _, _, log_prob = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            return_topk_log_probs=True,
                        )
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )

                    add_masked_advantage_metrics(
                        micro_batch_metrics,
                        advantages=advantages,
                        response_mask=policy_mask,
                        prefix="actor/opd_advantages",
                    )

                    opd_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=policy_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=None,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    policy_loss = opd_loss * opd_weight

                    loss = policy_loss * loss_scale_factor

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    did_backward = True

                    def _agg_topk_or_2d(loss_mat):
                        if loss_mat.dim() == 3:
                            loss_mat = loss_mat.sum(dim=-1)
                        return agg_loss(
                            loss_mat=loss_mat,
                            loss_mask=policy_mask,
                            loss_agg_mode=loss_agg_mode,
                        )

                    student_logp = _agg_topk_or_2d(log_prob)
                    teacher_logp = _agg_topk_or_2d(teacher_log_probs)
                    teacher_base_logp = _agg_topk_or_2d(teacher_base_log_probs)
                    teacher_delta_logp = _agg_topk_or_2d(teacher_log_probs - teacher_base_log_probs)

                    micro_batch_metrics["actor/opd_loss"] = opd_loss.detach().item() * loss_scale_factor
                    micro_batch_metrics["actor/opd_weighted_loss"] = policy_loss.detach().item() * loss_scale_factor
                    micro_batch_metrics["actor/opd_student_log_prob"] = student_logp.detach().item() * loss_scale_factor
                    micro_batch_metrics["actor/opd_teacher_log_prob"] = teacher_logp.detach().item() * loss_scale_factor
                    micro_batch_metrics["actor/opd_teacher_base_log_prob"] = teacher_base_logp.detach().item() * loss_scale_factor
                    micro_batch_metrics["actor/opd_teacher_delta_log_prob"] = teacher_delta_logp.detach().item() * loss_scale_factor
                    if has_topk_opd:
                        micro_batch_metrics["actor/opd_topk"] = float(model_inputs["student_topk_ids"].shape[-1])

                    append_to_dict(metrics, micro_batch_metrics)

                if did_backward:
                    grad_norm = self._optimizer_step(lr_scale=opd_lr_scale)
                    append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
                else:
                    append_to_dict(metrics, {"actor/grad_norm": 0.0})

        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_mpopd(self, data: DataProto):
        """Update only the student against the detached fused top-k target."""
        required = (
            "student_topk_ids",
            "student_base_topk_log_probs",
            "teacher_base_topk_log_probs",
            "specialized_expert_topk_log_probs",
            "expert_mask",
            "response_mask",
        )
        missing = [key for key in required if key not in data.batch]
        if missing:
            raise ValueError(f"MP-OPD update batch is missing tensors: {missing}")
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        mp_cfg = data.meta_info["mp_opd_config"]
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            *required[:-1],
        ]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch
        data = data.select(
            batch_keys=select_keys,
            non_tensor_batch_keys=["multi_modal_inputs"] if has_multi_modal_inputs else [],
        )
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                self.actor_optimizer.zero_grad()
                grad_accum = max(1, len(micro_batches))
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    target = build_mpopd_target(
                        model_inputs["specialized_expert_topk_log_probs"],
                        model_inputs["teacher_base_topk_log_probs"],
                        model_inputs["student_base_topk_log_probs"],
                        model_inputs["expert_mask"],
                        expert_temperature=float(mp_cfg["expert_temperature"]),
                        token_temperature=float(mp_cfg["token_temperature"]),
                        lambda_value=float(mp_cfg["lambda_value"]),
                        expert_priors=mp_cfg.get("expert_priors"),
                    )
                    if not torch.isfinite(target.target_probs).all():
                        raise FloatingPointError("MP-OPD target contains non-finite probabilities")
                    _, _, student_log_probs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        return_topk_log_probs=True,
                    )
                    loss = compute_mpopd_kl_loss(
                        student_log_probs,
                        target.target_probs,
                        response_mask,
                        target.valid_samples,
                        skip_samples_without_active_experts=bool(
                            mp_cfg.get("skip_samples_without_active_experts", True)
                        ),
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("MP-OPD loss is non-finite")
                    scale = response_mask.shape[0] / self.config.ppo_mini_batch_size if self.config.use_dynamic_bsz else 1 / grad_accum
                    (loss * scale).backward()
                    with torch.no_grad():
                        active = target.valid_samples
                        weights = target.expert_weights
                        valid_weights = weights[active]
                        if valid_weights.numel():
                            weight_entropy = -(valid_weights.clamp_min(1e-12) * valid_weights.clamp_min(1e-12).log()).sum(-1).mean()
                            max_weight = valid_weights.max(-1).values.mean()
                        else:
                            weight_entropy = weights.sum() * 0
                            max_weight = weights.sum() * 0

                        # §17.1 distribution diagnostics.
                        student_topk_probs = student_log_probs.detach().softmax(-1)
                        topk_mass = student_log_probs.detach().exp().sum(-1).mean()
                        target_max_prob = target.target_probs.max(-1).values.mean()

                        # §17.2 expert-signal diagnostics. Restrict to valid experts and
                        # response positions: a conflict is a candidate token where at
                        # least one valid expert raises the log-prob and another lowers it.
                        expert_mask_b = model_inputs["expert_mask"].to(torch.bool)
                        expert_delta = target.expert_delta
                        valid_e = expert_mask_b[:, None, None, :]
                        rm_tk = response_mask[:, :, None].to(expert_delta.dtype)
                        rm_tke = response_mask[:, :, None, None].to(expert_delta.dtype)
                        active_b = active[:, None, None].to(expert_delta.dtype)
                        pos = (expert_delta > 0) & valid_e
                        neg = (expert_delta < 0) & valid_e
                        conflict = pos.any(-1) & neg.any(-1)
                        conflict_denom = (rm_tk * active_b).sum().clamp_min(1.0)
                        conflict_rate = (conflict.to(rm_tk.dtype) * rm_tk * active_b).sum() / conflict_denom
                        valid_count = (rm_tke * valid_e.to(expert_delta.dtype)).sum().clamp_min(1.0)
                        pos_ratio = (pos.to(expert_delta.dtype) * rm_tke * valid_e.to(expert_delta.dtype)).sum() / valid_count
                        neg_ratio = (neg.to(expert_delta.dtype) * rm_tke * valid_e.to(expert_delta.dtype)).sum() / valid_count
                        active_expert_count = expert_mask_b.sum(-1).float().mean()

                        metric_update = {
                            "actor/mpopd_kl_loss": loss.detach().item() * scale,
                            "actor/mpopd_target_entropy": (-(target.target_probs * target.target_log_probs).sum(-1).mean()).item(),
                            "actor/mpopd_student_topk_entropy": (-(student_topk_probs * student_topk_probs.log()).sum(-1).mean()).item(),
                            "actor/mpopd_topk_probability_mass": topk_mass.item(),
                            "actor/mpopd_target_max_probability": target_max_prob.item(),
                            "actor/mpopd_expert_weight_entropy": weight_entropy.item(),
                            "actor/mpopd_max_expert_weight": max_weight.item(),
                            "actor/mpopd_fused_delta_mean": target.fused_delta.mean().item(),
                            "actor/mpopd_fused_delta_abs_mean": target.fused_delta.abs().mean().item(),
                            "actor/mpopd_conflict_rate": conflict_rate.item(),
                            "actor/mpopd_positive_delta_ratio": pos_ratio.item(),
                            "actor/mpopd_negative_delta_ratio": neg_ratio.item(),
                            "actor/mpopd_active_expert_count": active_expert_count.item(),
                            "actor/mpopd_all_experts_unavailable_count": (~target.valid_samples).sum().item(),
                            "actor/mpopd_top_k": student_log_probs.shape[-1],
                        }
                        # Per-expert delta / weight, averaged over response positions.
                        expert_names = mp_cfg.get("expert_names", [])
                        for e, name in enumerate(expert_names):
                            if e >= expert_delta.shape[-1]:
                                break
                            e_active = expert_mask_b[:, e]
                            e_denom = (response_mask * e_active[:, None].to(response_mask.dtype)).sum().clamp_min(1.0)
                            e_rm = response_mask[:, :, None] * e_active[:, None, None].to(response_mask.dtype)
                            e_delta = expert_delta[..., e]
                            e_weight = weights[..., e]
                            metric_update[f"actor/mpopd_{name}_delta_mean"] = ((e_delta * e_rm).sum() / e_denom).item()
                            metric_update[f"actor/mpopd_{name}_delta_abs_mean"] = ((e_delta.abs() * e_rm).sum() / e_denom).item()
                            metric_update[f"actor/mpopd_{name}_weight_mean"] = ((e_weight * e_rm).sum() / e_denom).item()
                        append_to_dict(metrics, metric_update)
                grad_norm = self._optimizer_step(lr_scale=float(self.config.get("mpopd_lr_scale", 1.0)))
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
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

        has_topk_opd = "student_topk_ids" in data.batch.keys()
        if has_topk_opd:
            select_keys.append("student_topk_ids")
        
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
                    if has_topk_opd:
                        padded_max_seq_len = mini_batch.batch["attention_mask"].shape[-1]
                        topk_cap = getattr(self.config, "ppo_topk_max_token_len_per_gpu", 0)
                        if not topk_cap or topk_cap <= 0:
                            topk_cap = max(4096, self.config.ppo_max_token_len_per_gpu // 4)
                        topk_cap = max(topk_cap, padded_max_seq_len)
                        max_token_len = min(
                            max_token_len, topk_cap * self.ulysses_sequence_parallel_size
                        )
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
                    if has_topk_opd:
                        entropy, _, log_prob = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            return_topk_log_probs=True,
                        )
                    else:
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
                    if has_topk_opd:
                        micro_batch_metrics["actor/opd_topk"] = float(model_inputs["student_topk_ids"].shape[-1])

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
                        policy_loss = pg_loss - entropy_loss * entropy_coeff   # 0
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss and not has_topk_opd:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef   # 0
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

        self.actor_optimizer.zero_grad()
        return metrics
