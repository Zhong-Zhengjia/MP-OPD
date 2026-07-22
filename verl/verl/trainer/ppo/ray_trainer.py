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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import time
from datetime import datetime
import json
import os
import uuid
import random
import sqlite3
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional
import copy

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, RayAsyncCollectHandle
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    add_macro_average_val_metrics,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    merge_worker_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from save_debug_sample import save_dataproto_single_sample_with_json_preview


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _index_dataproto(data: DataProto, idx):
    if isinstance(idx, torch.Tensor):
        idx_np = idx.detach().cpu().numpy()
        idx_t = idx.detach().cpu().long()
    else:
        idx_np = np.asarray(idx)
        idx_t = torch.from_numpy(idx_np).long()

    batch = data.batch[idx_t]

    non_tensor_batch = {}
    for k, v in data.non_tensor_batch.items():
        non_tensor_batch[k] = np.asarray(v)[idx_np]

    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=copy.deepcopy(data.meta_info),
    )


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
        ref_tokenizer=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
            ref_tokenizer: Optional tokenizer for reference model. If provided and different from tokenizer,
                re-tokenization will be performed before computing ref log probs.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Store ref_tokenizer for re-tokenization when ref model uses different tokenizer
        self.ref_tokenizer = ref_tokenizer
        self.use_ref_retokenization = ref_tokenizer is not None
        if self.use_ref_retokenization:
            from verl.trainer.ppo.ref_input_utils import tokenizers_need_cross_token_bridge

            self.use_cross_token_bridge = tokenizers_need_cross_token_bridge(tokenizer, ref_tokenizer)
        else:
            self.use_cross_token_bridge = False

        if self.use_ref_retokenization:
            return_raw_chat = config.data.get("return_raw_chat", False)
            if not return_raw_chat:
                raise ValueError(
                    "When using a different tokenizer for ref model (ref_tokenizer is provided) "
                    "you must set data.return_raw_chat=True in config to enable re-tokenization. "
                    "This is needed to access the original messages for re-tokenizing with ref model's chat template."
                )
            if self.use_cross_token_bridge:
                print(
                    "Cross-tokenizer bridge enabled for proxy/ref models "
                    "(text response bridge + common-token OPD mask)."
                )

        # training mode
        self.train_mode = self.config.algorithm.get("train_mode", "ppo")
        self.is_hetero_distill = self.train_mode == "heterogeneous_distill"

        hetero_cfg = self.config.algorithm.get("hetero_distill", {})
        print(hetero_cfg)
        self.student_rollout_n = hetero_cfg.get("student_rollout_n", 1)
        if self.is_hetero_distill:
            self.opd_top_k = int(hetero_cfg.get("opd_top_k", 0))
        else:
            self.opd_top_k = int(self.config.algorithm.get("opd_top_k", 0))
        self.opd_parallel_student_base = bool(hetero_cfg.get("opd_parallel_student_base", False))

        # Store base model paths for corrected reward computation
        self.base_model_path = config.actor_rollout_ref.model.get("base_model_path", None)
        self.ref_base_model_path = config.actor_rollout_ref.ref.get("model", None)
        if self.ref_base_model_path is not None:
            self.ref_base_model_path = self.ref_base_model_path.get("base_model_path", None)
        self.use_base_models = self.base_model_path is not None and self.ref_base_model_path is not None
        
        if self.use_base_models:
            print(f"Corrected reward enabled with base models:")
            print(f"  Actor base model: {self.base_model_path}")
            print(f"  Ref base model: {self.ref_base_model_path}")

                
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.teacher_rollout_wg = None

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        pace_val_files = self.config.data.get("pace_val_files", None)
        if pace_val_files:
            self.pace_val_dataset = create_rl_dataset(
                pace_val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
            self.pace_val_dataloader = StatefulDataLoader(
                dataset=self.pace_val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.pace_val_dataset = None
            self.pace_val_dataloader = None

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _prepare_validation_db(self):
        base_dir = self.config.trainer.default_local_dir
        os.makedirs(base_dir, exist_ok=True)

        db_path = os.path.join(base_dir, "validation_results.db")

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_results (
                    global_step INTEGER NOT NULL,
                    question_id TEXT NOT NULL,
                    rollout_id INTEGER NOT NULL,
                    reward INTEGER NOT NULL,
                    data_source TEXT NOT NULL,
                    sample_index TEXT NOT NULL,
                    prompt TEXT,
                    response TEXT NOT NULL,
                    ground_truth TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (global_step, question_id, rollout_id)
                )
                """
            )

        return db_path

    def _to_db_scalar(self, x):
        if isinstance(x, np.generic):
            return x.item()
        return x

    def _to_json_str(self, x):
        x = self._to_db_scalar(x)
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return json.dumps(x, ensure_ascii=False)

    def _save_validation_rollouts_to_db(
        self,
        conn,
        data_batch,
        input_texts,
        output_texts,
        ground_truths,
        rewards,
        rollout_counter,
    ):
        data_sources = data_batch.non_tensor_batch["data_source"]
        indices = data_batch.non_tensor_batch["index"]

        rows = []

        for data_source, index, prompt, response, gt, reward in zip(
            data_sources,
            indices,
            input_texts,
            output_texts,
            ground_truths,
            rewards,
            strict=True,
        ):
            data_source = str(self._to_db_scalar(data_source))
            index = str(self._to_db_scalar(index))
            question_id = f"{data_source}_{index}"

            rollout_id = rollout_counter[question_id]
            rollout_counter[question_id] += 1

            rows.append(
                (
                    int(self.global_steps),
                    question_id,
                    rollout_id,
                    int(round(float(reward))),
                    data_source,
                    index,
                    prompt,
                    response,
                    self._to_json_str(gt),
                )
            )

        if rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO validation_results (
                    global_step,
                    question_id,
                    rollout_id,
                    reward,
                    data_source,
                    sample_index,
                    prompt,
                    response,
                    ground_truth
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        return len(rows)

    def _topk_overlap_ratio(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.unsqueeze(-1).eq(b.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1) / a.shape[-1]

    def _validate(self, dataloader=None, val_n=None):
        validation_db_path = self._prepare_validation_db()

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_scores = []
        sample_turns = []
        sample_uids = []

        if dataloader is None:
            dataloader = self.val_dataloader
        if val_n is None:
            val_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        if val_n <= 0:
            raise ValueError(f"val_kwargs.n must be positive, got {val_n}")

        validation_rollout_batch_size = int(
            self.config.trainer.get("validation_rollout_batch_size", 1024)
        )
        if validation_rollout_batch_size <= 0:
            raise ValueError(
                f"trainer.validation_rollout_batch_size must be positive, "
                f"got {validation_rollout_batch_size}"
            )

        validation_db_commit_batch_size = int(
            self.config.trainer.get("validation_db_commit_batch_size", 8192)
        )
        validation_db_commit_batch_size = max(1, validation_db_commit_batch_size)

        repeat_chunk_size = min(val_n, validation_rollout_batch_size)
        prompt_chunk_size = max(1, validation_rollout_batch_size // repeat_chunk_size)

        rollout_counter = defaultdict(int)

        expected_rollout_counter = defaultdict(int)

        expected_total_rollouts = 0
        saved_total_rollouts = 0

        conn = sqlite3.connect(validation_db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            conn.execute("BEGIN")
            pending_db_rows = 0

            for test_data in dataloader:
                base_test_batch = DataProto.from_single_dict(test_data)
                base_batch_size = len(base_test_batch.batch["input_ids"])

                if "uid" not in base_test_batch.non_tensor_batch:
                    base_test_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(base_batch_size)],
                        dtype=object,
                    )

                if (
                    self.config.reward_model.enable
                    and base_test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
                ):
                    conn.commit()
                    return {}

                base_data_sources = base_test_batch.non_tensor_batch["data_source"]
                base_indices = base_test_batch.non_tensor_batch["index"]

                for data_source, index in zip(base_data_sources, base_indices, strict=True):
                    data_source = str(self._to_db_scalar(data_source))
                    index = str(self._to_db_scalar(index))
                    question_id = f"{data_source}_{index}"

                    expected_rollout_counter[question_id] += val_n
                    expected_total_rollouts += val_n

                for prompt_start in range(0, base_batch_size, prompt_chunk_size):
                    prompt_end = min(prompt_start + prompt_chunk_size, base_batch_size)

                    base_prompt_batch = base_test_batch[prompt_start:prompt_end]

                    for repeat_start in range(0, val_n, repeat_chunk_size):
                        cur_repeat = min(repeat_chunk_size, val_n - repeat_start)

                        test_batch = base_prompt_batch.repeat(
                            repeat_times=cur_repeat,
                            interleave=True,
                        )

                        current_batch_size = len(test_batch.batch["input_ids"])
                        expected_current_batch_size = (prompt_end - prompt_start) * cur_repeat
                        assert current_batch_size == expected_current_batch_size, (
                            f"Unexpected repeated batch size: "
                            f"{current_batch_size=} vs {expected_current_batch_size=}, "
                            f"{prompt_start=}, {prompt_end=}, "
                            f"{repeat_start=}, {cur_repeat=}"
                        )

                        input_ids = test_batch.batch["input_ids"]
                        input_texts = [
                            self.tokenizer.decode(ids, skip_special_tokens=True)
                            for ids in input_ids
                        ]

                        sample_uids.extend(test_batch.non_tensor_batch["uid"])

                        ground_truths = [
                            item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                            for item in test_batch
                        ]

                        test_gen_batch = self._get_gen_batch(test_batch)
                        test_gen_batch.meta_info = {
                            "eos_token_id": self.tokenizer.eos_token_id,
                            "pad_token_id": self.tokenizer.pad_token_id,
                            "recompute_log_prob": False,
                            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                            "validate": True,
                            "global_steps": self.global_steps,
                        }

                        size_divisor = (
                            self.actor_rollout_wg.world_size
                            if not self.async_rollout_mode
                            else self.config.actor_rollout_ref.rollout.agent.num_workers
                        )

                        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                            test_gen_batch,
                            size_divisor,
                        )

                        if not self.async_rollout_mode:
                            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                                test_gen_batch_padded
                            )
                        else:
                            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                                test_gen_batch_padded
                            )

                        test_output_gen_batch = unpad_dataproto(
                            test_output_gen_batch_padded,
                            pad_size=pad_size,
                        )

                        output_ids = test_output_gen_batch.batch["responses"]
                        output_texts = [
                            self.tokenizer.decode(ids, skip_special_tokens=True)
                            for ids in output_ids
                        ]

                        assert len(output_texts) == current_batch_size, (
                            f"Generated output size mismatch: "
                            f"{len(output_texts)=}, {current_batch_size=}"
                        )

                        test_batch.meta_info.pop("timing", None)
                        test_output_gen_batch.meta_info.pop("timing", None)

                        test_batch = test_batch.union(test_output_gen_batch)
                        test_batch = self._ensure_response_mask(test_batch)
                        test_batch.meta_info["validate"] = True

                        if self.val_reward_fn is None:
                            raise ValueError("val_reward_fn must be provided for validation.")

                        result = self.val_reward_fn(test_batch, return_dict=True)
                        reward_tensor = result["reward_tensor"]
                        scores = reward_tensor.sum(-1).cpu().tolist()

                        assert len(scores) == current_batch_size, (
                            f"Reward size mismatch: {len(scores)=}, {current_batch_size=}"
                        )

                        sample_scores.extend(scores)

                        inserted_rows = self._save_validation_rollouts_to_db(
                            conn=conn,
                            data_batch=test_batch,
                            input_texts=input_texts,
                            output_texts=output_texts,
                            ground_truths=ground_truths,
                            rewards=scores,
                            rollout_counter=rollout_counter,
                        )

                        assert inserted_rows == current_batch_size, (
                            f"Inserted rows mismatch: "
                            f"{inserted_rows=}, {current_batch_size=}"
                        )

                        saved_total_rollouts += inserted_rows
                        pending_db_rows += inserted_rows

                        response_lengths = test_batch.batch["response_mask"].sum(dim=-1).float()
                        batch_avg_response_len = response_lengths.mean().item()

                        now = datetime.now()
                        print(
                            f"[validation] {now.strftime('%Y-%m-%d %H:%M:%S')} | inserted_rows={inserted_rows}, "
                            f"saved_rollouts={saved_total_rollouts}, "
                            f"avg_response_len={batch_avg_response_len:.2f}",
                            flush=True,
                        )

                        if pending_db_rows >= validation_db_commit_batch_size:
                            conn.commit()
                            conn.execute("BEGIN")
                            pending_db_rows = 0

                        reward_extra_infos_dict["reward"].extend(scores)

                        if "reward_extra_info" in result:
                            for key, lst in result["reward_extra_info"].items():
                                assert len(lst) == current_batch_size, (
                                    f"reward_extra_info size mismatch for key={key}: "
                                    f"{len(lst)=}, {current_batch_size=}"
                                )
                                reward_extra_infos_dict[key].extend(lst)

                        if "__num_turns__" in test_batch.non_tensor_batch:
                            sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

                        data_source_lst.append(
                            test_batch.non_tensor_batch.get(
                                "data_source",
                                ["unknown"] * reward_tensor.shape[0],
                            )
                        )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

        assert saved_total_rollouts == expected_total_rollouts, (
            f"Validation rollout count mismatch: "
            f"{saved_total_rollouts=} vs {expected_total_rollouts=}"
        )

        for question_id, expected_count in expected_rollout_counter.items():
            actual_count = rollout_counter[question_id]
            assert actual_count == expected_count, (
                f"Question rollout count mismatch for {question_id}: "
                f"{actual_count=} vs {expected_count=}"
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            )

        if len(sample_scores) == 0:
            return {}

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
        )

        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(
                    [int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()]
                )
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"

                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        aggregate_group = self.config.trainer.get("val_aggregate_group", None)
        aggregate_sources = self.config.trainer.get("val_aggregate_sources", None)
        if aggregate_group and aggregate_sources:
            if isinstance(aggregate_sources, str):
                sources = [source.strip() for source in aggregate_sources.split(",") if source.strip()]
            else:
                sources = list(aggregate_sources)
            aggregate_var = self.config.trainer.get("val_aggregate_var", "reward")
            metric_dict = add_macro_average_val_metrics(
                metric_dict,
                group_name=aggregate_group,
                sources=sources,
                var_name=aggregate_var,
            )

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict


    def init_workers(self):
        """Initialize distributed training workers using Ray backend."""
        self.resource_pool_manager.create_resource_pool()

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_role = "actor_rollout_ref"
            ref_path = OmegaConf.select(self.config, "actor_rollout_ref.ref.model.path")
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=actor_rollout_role,
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.teacher_rollout_wg = None

        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _get_best_metric_name(self):
        metric_name = self.config.trainer.get("best_metric_name", None)
        assert metric_name is not None, (
            "trainer.best_metric_name must be set, "
            "e.g. val-core/gsm8k/acc/mean@1"
        )
        return metric_name


    def _get_best_metric_mode(self):
        mode = self.config.trainer.get("best_metric_mode", "max")
        assert mode in ["max", "min"], "trainer.best_metric_mode must be 'max' or 'min'"
        return mode


    def _is_better_metric(self, current, best):
        if best is None:
            return True
        mode = self._get_best_metric_mode()
        if mode == "max":
            return current > best
        return current < best

    def _save_checkpoint(self, metrics=None):
        import os
        import shutil
        import torch
        from verl.utils.fs import local_mkdir_safe

        base_dir = self.config.trainer.default_local_dir
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(os.getcwd(), base_dir)
        local_mkdir_safe(base_dir)

        best_metric_name = self._get_best_metric_name()

        if not hasattr(self, "_best_valid_metric"):
            self._best_valid_metric = None
        if not hasattr(self, "_best_valid_step"):
            self._best_valid_step = None

        current_metric = None
        is_best = False

        if metrics is not None and best_metric_name in metrics:
            current_metric = float(metrics[best_metric_name])
            is_best = self._is_better_metric(current_metric, self._best_valid_metric)
        else:
            print(
                f"Warning: best metric '{best_metric_name}' not found in metrics. "
                f"Will save as last checkpoint."
            )

        ckpt_name = "best_valid" if is_best else "last"
        local_ckpt_folder = os.path.join(base_dir, ckpt_name)

        print(
            f"Saving checkpoint to {local_ckpt_folder}, "
            f"global_step={self.global_steps}, is_best={is_best}, "
            f"{best_metric_name}={current_metric}"
        )

        # 清掉旧目录，只保留一个 best_valid 和一个 last
        if os.path.exists(local_ckpt_folder):
            shutil.rmtree(local_ckpt_folder)
        local_mkdir_safe(local_ckpt_folder)

        # actor
        actor_local_path = os.path.join(local_ckpt_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, ckpt_name, "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=None,
        )

        # critic
        if self.use_critic:
            critic_local_path = os.path.join(local_ckpt_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, ckpt_name, str(Role.Critic))
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=None,
            )

        # dataloader state
        dataloader_local_path = os.path.join(local_ckpt_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        meta = {
            "global_steps": int(self.global_steps),
            "is_best": bool(is_best),
            "metric_name": best_metric_name,
            "metric_value": current_metric,
        }
        if hasattr(self, "pace_enabled"):
            meta["pace_state"] = self._pace_state_dict()
        torch.save(meta, os.path.join(local_ckpt_folder, "meta.pt"))

        local_latest_checkpointed_iteration = os.path.join(base_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))
        if is_best:
            self._best_valid_metric = current_metric
            self._best_valid_step = int(self.global_steps)

            best_ckpt_txt = os.path.join(base_dir, "best_checkpoint.txt")
            with open(best_ckpt_txt, "w") as f:
                f.write(str(self.global_steps))

            print(
                f"Updated BEST checkpoint: "
                f"step={self.global_steps}, {best_metric_name}={current_metric}"
            )
        else:
            print(f"Updated LAST checkpoint: step={self.global_steps}")

    def _checkpoint_root_dir(self):
        base_dir = self.config.trainer.default_local_dir
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(os.getcwd(), base_dir)
        return base_dir

    def _is_checkpoint_dir(self, path):
        return path is not None and os.path.isdir(path) and os.path.isdir(os.path.join(path, "actor"))

    def _resolve_resume_path(self):
        resume_mode = self.config.trainer.get("resume_mode", "auto")
        resume_from_path = self.config.trainer.get("resume_from_path", None)
        base_dir = self._checkpoint_root_dir()

        if resume_mode == "disable":
            return None

        if resume_mode not in ["auto", "resume_path"]:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

        if resume_mode == "resume_path" and resume_from_path is None:
            raise ValueError("trainer.resume_from_path must be set when resume_mode='resume_path'")

        search_root = resume_from_path if resume_from_path is not None else base_dir
        if search_root is not None and not os.path.isabs(search_root):
            search_root = os.path.join(os.getcwd(), search_root)

        if self._is_checkpoint_dir(search_root):
            return search_root

        if search_root is not None and os.path.isdir(search_root):
            last_path = os.path.join(search_root, "last")
            if self._is_checkpoint_dir(last_path):
                return last_path

            best_path = os.path.join(search_root, "best_valid")
            if self._is_checkpoint_dir(best_path):
                return best_path

            latest_path = find_latest_ckpt_path(search_root)
            if self._is_checkpoint_dir(latest_path):
                return latest_path

        if resume_mode == "resume_path":
            raise FileNotFoundError(f"Could not find a resumable checkpoint under {search_root}")

        return None

    def _load_best_checkpoint_state(self):
        import torch

        base_dir = self._checkpoint_root_dir()
        best_meta_path = os.path.join(base_dir, "best_valid", "meta.pt")
        if not os.path.exists(best_meta_path):
            self._best_valid_metric = None
            self._best_valid_step = None
            return

        best_meta = torch.load(best_meta_path, map_location="cpu", weights_only=False)
        self._best_valid_metric = best_meta.get("metric_value", None)
        self._best_valid_step = best_meta.get("global_steps", None)

    def _load_checkpoint(self):
        import torch

        checkpoint_path = self._resolve_resume_path()
        self._load_best_checkpoint_state()

        if checkpoint_path is None:
            print("No checkpoint found. Starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")

        actor_local_path = os.path.join(checkpoint_path, "actor")
        actor_remote_path = None
        self.actor_rollout_wg.load_checkpoint(
            actor_local_path,
            actor_remote_path,
            del_local_after_load=self.config.trainer.get("del_local_ckpt_after_load", False),
        )

        if self.use_critic:
            critic_local_path = os.path.join(checkpoint_path, str(Role.Critic))
            if os.path.isdir(critic_local_path):
                self.critic_wg.load_checkpoint(
                    critic_local_path,
                    None,
                    del_local_after_load=self.config.trainer.get("del_local_ckpt_after_load", False),
                )
            else:
                print(f"Warning: critic checkpoint not found at {critic_local_path}; critic will start from init state.")

        dataloader_path = os.path.join(checkpoint_path, "data.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"Loaded dataloader state from {dataloader_path}")
        else:
            print(f"Warning: dataloader state not found at {dataloader_path}; dataloader will start from scratch.")

        meta_path = os.path.join(checkpoint_path, "meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            self.global_steps = int(meta.get("global_steps", self.global_steps))
            if hasattr(self, "pace_enabled"):
                self._load_pace_state_dict(meta.get("pace_state", {}))
            print(f"Loaded trainer state: global_steps={self.global_steps}")
        else:
            print(f"Warning: trainer meta not found at {meta_path}; global_steps remains {self.global_steps}.")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _is_empty_dataproto(self, batch: DataProto) -> bool:
        if batch is None:
            return True
        if not hasattr(batch, "batch") or batch.batch is None:
            return True
        if "input_ids" not in batch.batch:
            return True
        input_ids = batch.batch["input_ids"]
        if input_ids is None:
            return True
        if not hasattr(input_ids, "shape"):
            return True
        if input_ids.shape[0] == 0:
            return True
        return False

    def _extract_response_texts_from_batch(self, batch: DataProto):
        responses = batch.batch["responses"]
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]

    def _compute_binary_correctness_from_reward_tensor(self, reward_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reward_tensor: [bsz, resp_len] or [bsz]
        Returns:
            correctness: [bsz] bool tensor
        """
        if reward_tensor.dim() > 1:
            scores = reward_tensor.sum(dim=-1)
        else:
            scores = reward_tensor
        return scores > 0

    def _repeat_base_batch_for_rollout_n(self, base_batch: DataProto, rollout_n: int) -> DataProto:
        return base_batch.repeat(repeat_times=rollout_n, interleave=True)

    def _ensure_response_mask(self, data: DataProto):
        if "response_mask" not in data.batch:
            response_len = data.batch["responses"].shape[-1]
            data.batch["response_mask"] = data.batch["attention_mask"][:, -response_len:]
        return data

    def _init_hetero_train_state(self):
        hetero_cfg = self.config.algorithm.get("hetero_distill", {})
        pace_cfg = self.config.actor_rollout_ref.actor.policy_loss

        self.student_rollout_n = hetero_cfg.get("student_rollout_n", 8)

        self.grpo_update_batch_size = hetero_cfg.get(
            "grpo_update_batch_size",
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
        )
        self.opd_update_batch_size = hetero_cfg.get(
            "opd_update_batch_size",
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
        )

        assert self.grpo_update_batch_size % self.student_rollout_n == 0
        assert self.opd_update_batch_size % self.student_rollout_n == 0

        self.actor_update_steps = 0
        self.grpo_update_steps = 0
        self.opd_update_steps = 0
        self.pace_enabled = bool(pace_cfg.get("pace_enable", False))
        self.pace_micro_enabled = self.pace_enabled and bool(pace_cfg.get("pace_micro_enable", False))
        self.pace_macro_enabled = self.pace_enabled and bool(pace_cfg.get("pace_macro_enable", False))
        self.pace_lambda0 = float(pace_cfg.get("pace_lambda0_init", pace_cfg.get("lambda_vals", 1.0)))
        self.pace_m = 0.0
        self.pace_v = 0.0
        self.pace_prev_reward = None

    def _pace_config(self):
        return self.config.actor_rollout_ref.actor.policy_loss

    def _pace_state_dict(self) -> dict:
        return {
            "enabled": self.pace_enabled,
            "lambda0": self.pace_lambda0,
            "m": self.pace_m,
            "v": self.pace_v,
            "prev_reward": self.pace_prev_reward,
        }

    def _load_pace_state_dict(self, state: dict) -> None:
        if not self.pace_enabled or not state:
            return
        self.pace_lambda0 = float(state.get("lambda0", self.pace_lambda0))
        self.pace_m = float(state.get("m", self.pace_m))
        self.pace_v = float(state.get("v", self.pace_v))
        prev_reward = state.get("prev_reward", self.pace_prev_reward)
        self.pace_prev_reward = None if prev_reward is None else float(prev_reward)

    def _maybe_update_pace_from_validation(self, metrics: dict) -> None:
        """Run held-out validation at PACE's interval, before an OPD update."""
        if not self.pace_enabled or not self.pace_macro_enabled:
            return

        pace_cfg = self._pace_config()
        if pace_cfg.get("pace_reward_source", "rollout") != "validation":
            return
        if self.global_steps % int(pace_cfg.get("pace_update_interval", 1)) != 0:
            metrics["pace/controller_update"] = 0.0
            return

        if not hasattr(self, "pace_val_dataloader") or self.pace_val_dataloader is None:
            raise ValueError("PACE validation dataloader is not initialized. Please provide data.pace_val_files in config.")

        pace_val_n = int(pace_cfg.get("pace_validation_n", 8))
        validation_metrics = self._validate(dataloader=self.pace_val_dataloader, val_n=pace_val_n)
        
        # Add prefix so it doesn't pollute the main metrics
        metrics.update({f"pace-val/{k}": v for k, v in validation_metrics.items()})
        
        metric_name = pace_cfg.get("pace_validation_metric", "")
        if not metric_name:
            raise ValueError("pace_validation_metric must be set when pace_reward_source='validation'")
            
        if metric_name not in validation_metrics:
            available = ", ".join(sorted(validation_metrics))
            raise KeyError(
                f"PACE validation reward metric '{metric_name}' (pace_validation_metric) "
                f"was not produced by _validate(). Available: {available}"
            )
        metrics["pace/progress_source_validation"] = 1.0
        metrics["pace/validation_reward"] = float(validation_metrics[metric_name])
        self._update_pace_controller(metrics["pace/validation_reward"], metrics)

    def _update_pace_controller(self, reward: float, metrics: dict) -> None:
        """Update PACE's global lambda baseline from a sampled rollout reward."""
        if not self.pace_enabled:
            return

        pace_cfg = self._pace_config()
        metrics["pace/lambda0"] = self.pace_lambda0
        metrics["pace/macro_enabled"] = float(self.pace_macro_enabled)
        if not self.pace_macro_enabled:
            return

        interval = int(pace_cfg.get("pace_update_interval", 1))
        if self.global_steps % interval != 0:
            metrics["pace/controller_update"] = 0.0
            return

        reward = float(reward)
        metrics["pace/reward"] = reward
        if self.pace_prev_reward is None:
            self.pace_prev_reward = reward
            metrics["pace/controller_update"] = 0.0
            return

        epsilon = float(pace_cfg.get("pace_epsilon", 1e-8))
        delta = (reward - self.pace_prev_reward) / (abs(self.pace_prev_reward) + epsilon)
        beta1 = float(pace_cfg.get("pace_beta1", 0.9))
        beta2 = float(pace_cfg.get("pace_beta2", 0.99))
        self.pace_m = beta1 * self.pace_m + (1.0 - beta1) * delta
        self.pace_v = beta2 * self.pace_v + (1.0 - beta2) * delta * delta
        # SNR is the signed, noise-normalized reward-improvement trend:
        # positive => stable improvement; negative => degradation. The exponent
        # maps that trend multiplicatively onto lambda0; it intentionally matches
        # the proposal without an extra clip. lambda_min/max bound lambda0 itself.
        snr = self.pace_m / (np.sqrt(self.pace_v) + epsilon)
        exponent = -float(pace_cfg.get("pace_eta", 0.1)) * snr
        self.pace_lambda0 = float(
            np.clip(
                self.pace_lambda0 * np.exp(exponent),
                float(pace_cfg.get("pace_lambda_min", 0.1)),
                float(pace_cfg.get("pace_lambda_max", 10.0)),
            )
        )
        self.pace_prev_reward = reward
        metrics.update(
            {
                "pace/controller_update": 1.0,
                "pace/reward_delta": delta,
                "pace/reward_snr": snr,
                "pace/lambda0": self.pace_lambda0,
                "pace/m": self.pace_m,
                "pace/v": self.pace_v,
            }
        )

    def _apply_pace_micro_reweight(self, batch: DataProto) -> dict[str, float]:
        """Attach batch-conserving token lambda values for the PUST anchor term."""
        if not self.pace_enabled or (not self.pace_micro_enabled and not self.pace_macro_enabled):
            return {}

        pace_cfg = self._pace_config()
        response_mask = batch.batch["response_mask"].float()
        # response_mask excludes padding. Cross-token PUST additionally excludes
        # primary positions that cannot be aligned to a proxy token. Only these
        # positions participate in OPD loss or in PACE batch statistics.
        policy_mask = response_mask
        if "cross_token_opd_mask" in batch.batch:
            policy_mask = policy_mask * batch.batch["cross_token_opd_mask"].to(dtype=policy_mask.dtype)

        if not self.pace_micro_enabled:
            token_lambda = torch.full_like(policy_mask, self.pace_lambda0)
            batch.batch["pace_token_lambda"] = token_lambda
            return {
                "pace/enabled": 1.0,
                "pace/micro_enabled": 0.0,
                "pace/lambda0": self.pace_lambda0,
                "pace/token_lambda_mean": self.pace_lambda0,
            }

        epsilon = float(pace_cfg.get("pace_epsilon", 1e-8))
        # Select only loss-bearing positions: padding/alignment-excluded tokens
        # must not affect MinMax ranges, mean importance, or the lambda budget.
        valid = policy_mask > 0
        if not torch.any(valid):
            batch.batch["pace_token_lambda"] = torch.full_like(policy_mask, self.pace_lambda0)
            return {
                "pace/enabled": 1.0,
                "pace/micro_enabled": 1.0,
                "pace/empty_valid_mask": 1.0,
                "pace/lambda0": self.pace_lambda0,
            }

        entropy = batch.batch["entropys"].detach().float()
        # TODO(PACE): for a shared tokenizer, replace this scalar proxy with
        # KL(pi_student || pi_teacher) from full distributions when available.
        # Cross-tokenizer PUST only aligns log-probabilities of corresponding
        # token positions; its teacher and student vocabularies differ, so the
        # full distributions are defined over different event spaces and an
        # exact token-level KL cannot be computed without a vocabulary mapping.
        disagreement = (batch.batch["old_log_probs"] - batch.batch["teacher_log_probs"]).detach().float().abs()

        def _minmax(values: torch.Tensor) -> torch.Tensor:
            selected = values[valid]
            return (values - selected.min()) / (selected.max() - selected.min() + epsilon)

        entropy_hat = _minmax(entropy)
        disagreement_hat = _minmax(disagreement)
        importance = entropy_hat + disagreement_hat - entropy_hat * disagreement_hat # Reference: https://arxiv.org/pdf/2604.14084
        mean_importance = importance[valid].mean()
        # Initialize excluded positions to one; they never enter the loss.
        # For valid positions, g_t=(1-s_t+eps)/(1-mean(s)+eps), so its batch
        # mean is exactly one up to floating-point rounding. This line alone
        # does not ensure conservation; the following valid-position assignment
        # and denominator containing mean_importance do.
        modulation = torch.ones_like(importance)
        modulation[valid] = (1.0 - importance[valid] + epsilon) / (1.0 - mean_importance + epsilon)
        
        # Clip and re-normalize to prevent extreme amplification
        max_mod = float(pace_cfg.get("pace_micro_max_modulation", 5.0)) # TODO: To be checked
        modulation[valid] = torch.clamp(modulation[valid], max=max_mod)
        mod_mean = modulation[valid].mean()
        modulation[valid] = modulation[valid] / (mod_mean + epsilon)
        
        token_lambda = self.pace_lambda0 * modulation
        batch.batch["pace_token_lambda"] = token_lambda.to(dtype=batch.batch["old_log_probs"].dtype)

        valid_lambda = token_lambda[valid].float()
        res = {
            "pace/enabled": 1.0,
            "pace/micro_enabled": 1.0,
            "pace/lambda0": self.pace_lambda0,
            "pace/importance_mean": mean_importance.item(),
            "pace/modulation_mean": modulation[valid].mean().item(),
            "pace/modulation_std": modulation[valid].std(unbiased=False).item(),
            "pace/token_lambda_mean": valid_lambda.mean().item(),
            "pace/token_lambda_max": valid_lambda.max().item(),
            "pace/token_lambda_min": valid_lambda.min().item(),
            "pace/disagreement_proxy_mean": disagreement[valid].mean().item(),
            "pace/entropy_mean": entropy[valid].mean().item(),
        }
        if valid_lambda.numel() > 0:
            res["pace/token_lambda_p90"] = torch.quantile(valid_lambda, 0.90).item()
            res["pace/token_lambda_p99"] = torch.quantile(valid_lambda, 0.99).item()
        return res


    def _build_hetero_train_batches(
        self,
        student_batch: DataProto,
        student_reward_tensor: torch.Tensor,
    ):
        import numpy as np
        import torch

        n_s = self.student_rollout_n

        student_batch = self._ensure_response_mask(student_batch)

        s_bsz = len(student_batch)
        assert s_bsz % n_s == 0

        num_groups = s_bsz // n_s

        if "uid" not in student_batch.non_tensor_batch:
            student_batch.non_tensor_batch["uid"] = np.repeat(
                np.arange(num_groups).astype(str),
                n_s,
            )

        if "reward" not in student_batch.batch:
            student_reward = self._compute_binary_correctness_from_reward_tensor(student_reward_tensor).float()
            student_batch.batch["reward"] = student_reward

        s_reward_flat = student_batch.batch["reward"].detach().float()

        if s_reward_flat.dim() > 1:
            s_reward_flat = s_reward_flat.view(s_reward_flat.shape[0], -1).max(dim=-1).values

        s_reward = s_reward_flat.view(num_groups, n_s)
        s_correct = s_reward > 0

        s_all_correct = s_correct.all(dim=-1)
        s_all_wrong = (~s_correct).all(dim=-1)
        s_mixed = ~(s_all_correct | s_all_wrong)

        grpo_group_mask = s_mixed

        # opd_group_mask = (~s_all_correct)

        grpo_groups = torch.nonzero(grpo_group_mask, as_tuple=False).flatten()
        # opd_groups = torch.nonzero(opd_group_mask, as_tuple=False).flatten()

        def expand_student_indices(groups):
            if groups.numel() == 0:
                return torch.empty(0, dtype=torch.long)
            offsets = torch.arange(n_s, device=groups.device).view(1, n_s)
            return (groups.view(-1, 1) * n_s + offsets).reshape(-1).long()

        grpo_idx = expand_student_indices(grpo_groups).cpu()
        # opd_idx = expand_student_indices(opd_groups).cpu()
        opd_idx = torch.arange(s_bsz, dtype=torch.long)

        grpo_batch = None
        opd_batch = None

        if grpo_idx.numel() > 0:
            grpo_batch = _index_dataproto(student_batch, grpo_idx)
            grpo_reward = student_reward_tensor.detach().cpu()[grpo_idx]
            grpo_batch.batch["token_level_rewards"] = grpo_reward
            grpo_batch.batch["token_level_scores"] = grpo_reward

            if "old_log_probs" not in grpo_batch.batch and "rollout_log_probs" in grpo_batch.batch:
                grpo_batch.batch["old_log_probs"] = grpo_batch.batch["rollout_log_probs"]

        if opd_idx.numel() > 0:
            opd_batch = _index_dataproto(student_batch, opd_idx)

        stats = {
            "hetero/student_all_correct_group_count": int(s_all_correct.sum().item()),
            "hetero/student_mixed_group_count": int(s_mixed.sum().item()),
            "hetero/student_all_wrong_group_count": int(s_all_wrong.sum().item()),

            "hetero/grpo_candidate_group_count": int(grpo_groups.numel()),
            "hetero/opd_candidate_group_count": int(num_groups),
            "hetero/opd_candidate_row_count": int(opd_idx.numel())
        }

        return grpo_batch, opd_batch, stats


    def _get_hetero_update_mode(self):
        hd_cfg = self.config.algorithm.hetero_distill

        use_grpo = bool(hd_cfg.get("use_grpo", True))
        use_opd = bool(hd_cfg.get("use_opd", True))
        update_mode = hd_cfg.get("update_mode", "both")
        print('[DEBUG] update_mode: ', update_mode, type(update_mode))

        counter = max(self.global_steps - 1, 0)

        if update_mode == "both":
            return use_grpo, use_opd, "both"

        if update_mode == "alt":
            assert use_grpo and use_opd

            opd_steps = int(hd_cfg.get("opd_steps", 10))
            grpo_steps = int(hd_cfg.get("grpo_steps", 50))

            assert opd_steps > 0
            assert grpo_steps > 0

            cycle = opd_steps + grpo_steps
            pos = counter % cycle
            phase = "opd" if pos < opd_steps else "grpo"

            return phase == "grpo", phase == "opd", phase

        if update_mode == "warmup":
            assert use_grpo and use_opd

            warmup_steps = int(hd_cfg.get("warmup_steps", 10))
            assert warmup_steps > 0

            phase = "opd" if counter < warmup_steps else "grpo"

            return phase == "grpo", phase == "opd", phase

        raise ValueError(f"Unknown hetero update_mode: {update_mode}")


    def _add_entropy_metrics(self, metrics, batch, entropys, prefix: str):
        import torch
        with torch.no_grad():
            ent = entropys.detach().float()
            mask = None
            for mask_key in ["response_mask", "loss_mask", "attention_mask"]:
                if mask_key in batch.batch:
                    mask = batch.batch[mask_key]
                    break
            if mask is not None:
                mask = mask.to(ent.device)
                if mask.shape != ent.shape:
                    if mask.dim() == ent.dim() and mask.shape[0] == ent.shape[0]:
                        mask = mask[:, -ent.shape[-1]:]
                    else:
                        mask = None
            if mask is not None:
                mask = mask.bool()
                valid_ent = ent[mask]
            else:
                valid_ent = ent.reshape(-1)
            if valid_ent.numel() == 0:
                metrics[f"{prefix}/entropy_valid_tokens"] = 0
                return
            metrics[f"{prefix}/entropy_mean"] = valid_ent.mean().item()
            # metrics[f"{prefix}/entropy_std"] = valid_ent.std(unbiased=False).item()
            # metrics[f"{prefix}/entropy_min"] = valid_ent.min().item()
            # metrics[f"{prefix}/entropy_max"] = valid_ent.max().item()
            # metrics[f"{prefix}/entropy_valid_tokens"] = int(valid_ent.numel())


    def _build_union_topk_ids(self, actor_topk_ids, ref_topk_ids, chunk_size=262144, compact_ids=True):
        assert actor_topk_ids.shape == ref_topk_ids.shape

        device = actor_topk_ids.device
        orig_shape = actor_topk_ids.shape[:-1]
        k = actor_topk_ids.shape[-1]
        m = k * 2

        actor_flat = actor_topk_ids.reshape(-1, k)
        ref_flat = ref_topk_ids.reshape(-1, k)
        n = actor_flat.shape[0]

        out_dtype = torch.int32 if compact_ids else actor_topk_ids.dtype

        union_ids = torch.empty(
            n,
            m,
            dtype=out_dtype,
            device=device,
        )
        union_mask = torch.empty(
            n,
            m,
            dtype=torch.bool,
            device=device,
        )

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            ids = torch.cat(
                [
                    actor_flat[start:end],
                    ref_flat[start:end],
                ],
                dim=-1,
            )

            sorted_ids, order = torch.sort(ids, dim=-1)

            keep_sorted = torch.ones_like(sorted_ids, dtype=torch.bool)
            keep_sorted[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]

            keep = torch.empty_like(keep_sorted)
            keep.scatter_(dim=-1, index=order, src=keep_sorted)

            union_ids[start:end].copy_(ids.to(out_dtype))
            union_mask[start:end].copy_(keep)

        union_ids = union_ids.reshape(*orig_shape, m)
        union_mask = union_mask.reshape(*orig_shape, m)

        return union_ids, union_mask


    def _add_topk_overlap_metrics(
        self,
        metrics,
        batch,
        actor_topk_ids,
        ref_topk_ids,
        union_topk_mask,
        prefix="opd",
    ):
        with torch.no_grad():
            k = actor_topk_ids.shape[-1]
            token_shape = actor_topk_ids.shape[:-1]

            union_count = union_topk_mask.sum(dim=-1).float()
            intersection_count = (2.0 * k - union_count).clamp(min=0.0, max=float(k))

            top1_match = actor_topk_ids[..., 0] == ref_topk_ids[..., 0]
            teacher_top1_in_student_topk = (actor_topk_ids == ref_topk_ids[..., :1]).any(dim=-1)
            student_top1_in_teacher_topk = (ref_topk_ids == actor_topk_ids[..., :1]).any(dim=-1)

            valid_mask = None
            for key in ("response_mask", "loss_mask"):
                if key in batch.batch and tuple(batch.batch[key].shape) == tuple(token_shape):
                    valid_mask = batch.batch[key].bool()
                    break

            if valid_mask is None:
                valid_mask = torch.ones(token_shape, dtype=torch.bool, device=actor_topk_ids.device)

            valid_mask = valid_mask & torch.isfinite(intersection_count)

            metrics[f"{prefix}/topk_intersection_count"] = intersection_count[valid_mask].float().mean().item()
            metrics[f"{prefix}/topk_union_count"] = union_count[valid_mask].float().mean().item()
            metrics[f"{prefix}/top1_match_rate"] = top1_match[valid_mask].float().mean().item()
            metrics[f"{prefix}/teacher_top1_in_student_topk_rate"] = teacher_top1_in_student_topk[valid_mask].float().mean().item()
            metrics[f"{prefix}/student_top1_in_teacher_topk_rate"] = student_top1_in_teacher_topk[valid_mask].float().mean().item()


    def _prepare_opd_batch(
        self,
        batch: DataProto,
        *,
        include_base: bool,
        ref_on_actor: bool = False,
        ref_output_key: str = "teacher_log_probs",
        fetch_ref_externally: bool = False,
    ) -> tuple[DataProto, object | None, dict]:
        """Attach OPD log-prob tensors via prepare_opd_log_probs (+ optional external ref RPC)."""
        cross_token_stats: dict = {}
        if self.use_ref_retokenization:
            from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs

            apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
            batch, cross_token_stats = prepare_ref_model_inputs(
                batch=batch,
                ref_tokenizer=self.ref_tokenizer,
                primary_tokenizer=self.tokenizer,
                cross_token_bridge=self.use_cross_token_bridge,
                apply_chat_template_kwargs=apply_chat_template_kwargs,
            )

        opd_top_k = self.opd_top_k
        if self.use_cross_token_bridge and opd_top_k > 0:
            raise ValueError(
                "cross-tokenizer PUST (Phase 0/1) does not support opd_top_k > 0. "
                "Set algorithm.hetero_distill.opd_top_k=0 or use the same tokenizer."
            )
        if opd_top_k > 0:
            batch.batch.pop("old_log_probs", None)

        batch.meta_info["top_k"] = opd_top_k
        batch.meta_info["opd_include_base"] = include_base
        batch.meta_info["opd_parallel_student_base"] = self.opd_parallel_student_base
        if ref_on_actor:
            batch.meta_info["is_lora"] = True
            batch.meta_info["opd_ref_on_actor"] = True
            batch.meta_info["opd_ref_output_key"] = ref_output_key

        if opd_top_k > 0 and include_base:
            print(f"[DEBUG] OPD top-k update (only_stu), K={opd_top_k}")

        lp_output = self.actor_rollout_wg.prepare_opd_log_probs(batch)
        timing_metrics = lp_output.meta_info.get("metrics", {})
        if timing_metrics:
            timing_metrics = reduce_metrics(timing_metrics)

        batch.batch["old_log_probs"] = lp_output.batch["old_log_probs"]
        if "entropys" in lp_output.batch:
            batch.batch["entropys"] = lp_output.batch["entropys"]

        if include_base:
            batch.batch["base_log_probs"] = lp_output.batch["base_log_probs"]
            batch.batch["teacher_log_probs"] = lp_output.batch["teacher_log_probs"]
            batch.batch["teacher_base_log_probs"] = lp_output.batch["teacher_base_log_probs"]

        if self.use_cross_token_bridge:
            from verl.trainer.ppo.ref_input_utils import (
                align_teacher_log_probs_to_primary,
                maybe_apply_sequence_level_opd_fallback,
            )

            batch, align_stats = align_teacher_log_probs_to_primary(batch)
            cross_token_stats.update(align_stats)

            hetero_cfg = self.config.algorithm.get("hetero_distill", {})
            fallback_threshold = float(hetero_cfg.get("cross_token_common_ratio_threshold", 0.8))
            batch, fallback_stats = maybe_apply_sequence_level_opd_fallback(
                batch,
                common_ratio_threshold=fallback_threshold,
            )
            cross_token_stats.update(fallback_stats)

        if opd_top_k > 0:
            batch.batch["student_topk_ids"] = lp_output.batch["student_topk_ids"]
            if ref_on_actor and ref_output_key in lp_output.batch:
                batch.batch[ref_output_key] = lp_output.batch[ref_output_key]
            elif fetch_ref_externally:
                ref_batch = batch
                ref_batch.meta_info["opd_ref_output_key"] = ref_output_key
                ref_lp = self.ref_policy_wg.compute_ref_log_probs_on_ids(ref_batch)
                batch = batch.union(ref_lp)

        entropys = lp_output.batch["entropys"] if "entropys" in lp_output.batch else None
        timing_metrics.update(cross_token_stats)
        return batch, entropys, timing_metrics

    def _prepare_opd_update_batch(self, update_batch: DataProto) -> tuple[DataProto, dict]:
        """PUST / hetero OPD update: student + base + teacher + teacher-base log probs."""
        update_batch, _, timing_metrics = self._prepare_opd_batch(
            update_batch,
            include_base=True,
        )
        return update_batch, timing_metrics

    def fit_heterogeneous(self):
        from omegaconf import OmegaConf
        from pprint import pprint
        from tqdm import tqdm
        from verl.utils.tracking import Tracking

        if not hasattr(self, "grpo_buffer"):
            self._init_hetero_train_state()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything (restores model, dataloader, global_steps)
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="HeteroDistill Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                is_last_step = self.global_steps >= self.total_training_steps


                if self._is_empty_dataproto(batch):
                    metrics = {
                        "hetero/skip_empty_base_batch": 1,
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1
                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return
                    continue

                # add uid
                if "uid" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch["input_ids"]))], dtype=object
                    )

                student_gen_batch = self._repeat_base_batch_for_rollout_n(batch, self.student_rollout_n)
                if self._is_empty_dataproto(student_gen_batch):
                    metrics["hetero/skip_empty_student_gen_batch"] = 1

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return
                    continue
                student_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "do_sample": True,
                    "validate": False,
                    "global_steps": self.global_steps,
                }

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        student_rollout_batch = self.actor_rollout_wg.generate_sequences(student_gen_batch)

                    student_batch = student_rollout_batch
                    student_batch = self._ensure_response_mask(student_batch)
                    student_batch.meta_info["global_token_num"] = torch.sum(
                        student_batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    student_response_len = student_batch.batch["response_mask"].sum(dim=-1).float().mean().item()
                    metrics["hetero/student_response_length_mean"] = student_response_len

                    # ======Rollout Debug Start ======
                    student_batch_size = len(student_batch.batch["prompts"])
                    print(f'[DEBUG] Student rollout batch size: {student_batch_size}')
                    print(f'[DEBUG] Student average response length: {student_response_len}')
                    print(f"[DEBUG] Student rollout time cost: {timing_raw.get('gen', 0.0)} s")

                    i= 0
                    prompt_ids = student_batch.batch["prompts"][i]
                    prompt_length = prompt_ids.shape[0]
                    attention_mask = student_batch.batch["attention_mask"][i]
                    response_ids = student_batch.batch["responses"][i]
                    valid_response_length = attention_mask[prompt_length:].sum().item()
                    valid_response_ids = response_ids[:valid_response_length]
                    response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                    print("[DEBUG] Student response_text:")
                    print(repr(response_text))

                    now = datetime.now()
                    print('[DEBUG] current time: ', now.strftime('%Y-%m-%d %H:%M:%S'))
                    # ======Rollout Debug End ======

                    # =========================
                    # 3) reward / correctness
                    # =========================
                    with marked_timer("reward", timing_raw, color="yellow"):
                        student_reward_tensor, student_reward_extra_infos = compute_reward(
                            student_batch, self.reward_fn
                        )
                    student_reward = self._compute_binary_correctness_from_reward_tensor(
                        student_reward_tensor
                    ).float()
                    student_batch.batch["reward"] = student_reward

                    from collections import defaultdict

                    student_reward_flat = student_reward.detach().float().view(-1).cpu()
                    uids = student_batch.non_tensor_batch["uid"]

                    uid_to_indices = defaultdict(list)
                    for idx, uid in enumerate(uids):
                        uid_to_indices[str(uid)].append(idx)

                    group_stds = []
                    for indices in uid_to_indices.values():
                        group_rewards = student_reward_flat[indices]
                        group_stds.append(group_rewards.std(unbiased=False))

                    if len(group_stds) > 0:
                        student_rollout_correct_group_mean_std = torch.stack(group_stds).mean().item()
                    else:
                        student_rollout_correct_group_mean_std = 0.0

                    metrics["hetero/student_rollout_correct_mean"] = student_reward_flat.mean().item()
                    metrics["hetero/student_rollout_correct_std"] = student_reward.std(unbiased=False).item()
                    metrics["hetero/student_rollout_correct_group_mean_std"] = student_rollout_correct_group_mean_std
                    if self.pace_enabled and self.pace_macro_enabled:
                        if self._pace_config().get("pace_reward_source", "rollout") == "rollout":
                            metrics["pace/progress_source_rollout"] = 1.0
                            self._update_pace_controller(metrics["hetero/student_rollout_correct_mean"], metrics)
                        else:
                            # Validation is deliberately deferred until after rollout
                            # batch construction and immediately before OPD preparation.
                            # That makes the updated lambda0 apply to this OPD update.
                            metrics["pace/progress_source_rollout"] = 0.0

                    # =========================
                    # 4) build distillation batch
                    # =========================
                    with marked_timer("build_hetero", timing_raw, color="cyan"):
                        grpo_batch, opd_batch, build_stats = self._build_hetero_train_batches(
                            student_batch=student_batch,
                            student_reward_tensor=student_reward_tensor,
                        )
                    metrics.update(build_stats)

                    do_grpo, do_opd, update_phase = self._get_hetero_update_mode()
                    metrics["hetero/update_phase_is_grpo"] = int(update_phase == "grpo")
                    metrics["hetero/update_phase_is_opd"] = int(update_phase == "opd")

                    # =========================
                    # GRPO update
                    # =========================
                    grpo_has_candidate = grpo_batch is not None and len(grpo_batch) > 0

                    if do_grpo:
                        if grpo_has_candidate:

                            if grpo_has_candidate:
                                grpo_batch = compute_advantage(
                                    data=grpo_batch,
                                    adv_estimator=AdvantageEstimator.GRPO,
                                    gamma=self.config.algorithm.get("gamma", 1.0),
                                    lam=self.config.algorithm.get("lam", 1.0),
                                    num_repeat=self.student_rollout_n,
                                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                                    config=self.config.algorithm,
                                )

                            update_batch = grpo_batch

                            if update_batch is not None and len(update_batch) > 0:
                                print("[DEBUG] GRPO UPDATE START.")

                                with marked_timer("grpo_prep", timing_raw, color="blue"):
                                    old_log_prob_output = self.actor_rollout_wg.compute_log_prob(update_batch)
                                    update_batch.batch["old_log_probs"] = old_log_prob_output.batch["old_log_probs"]
                                    update_batch.batch['entropys'] = old_log_prob_output.batch["entropys"]

                                    base_log_probs_output = self.actor_rollout_wg.compute_base_log_prob(update_batch)
                                    update_batch.batch["base_log_probs"] = base_log_probs_output.batch["base_log_probs"]

                                self._add_entropy_metrics(
                                    metrics=metrics,
                                    batch=update_batch,
                                    entropys=old_log_prob_output.batch["entropys"],
                                    prefix="grpo",
                                )

                                with marked_timer("update_grpo", timing_raw, color="red"):
                                    actor_output = self.actor_rollout_wg.update_actor_grpo(update_batch)
                                actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(merge_worker_metrics(actor_metrics, prefix="grpo"))

                                self.grpo_update_steps += 1
                                self.actor_update_steps += 1
                        else:
                            metrics["hetero/grpo_skip_no_candidate"] = 1
                    else:
                        metrics["hetero/grpo_skip_by_schedule"] = 1

                    # =========================
                    # OPD update
                    # =========================
                    if do_opd:
                        if opd_batch is not None and len(opd_batch) > 0:
                            update_batch = opd_batch

                            if update_batch is not None and len(update_batch) > 0:
                                print("[DEBUG] OPD UPDATE START.")

                                self._maybe_update_pace_from_validation(metrics)
                                with marked_timer("opd_prep", timing_raw, color="olive"):
                                    update_batch, opd_prep_timing = self._prepare_opd_update_batch(update_batch)
                                metrics.update(merge_worker_metrics(opd_prep_timing, prefix="opd"))
                                metrics.update(self._apply_pace_micro_reweight(update_batch))
                                self._add_entropy_metrics(
                                    metrics=metrics,
                                    batch=update_batch,
                                    entropys=update_batch.batch["entropys"],
                                    prefix="opd",
                                )
                                if self.opd_top_k > 0:
                                    metrics["opd/top_k"] = self.opd_top_k

                                with marked_timer("update_opd", timing_raw, color="purple"):
                                    actor_output = self.actor_rollout_wg.update_actor_opd(update_batch)
                                actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(merge_worker_metrics(actor_metrics, prefix="opd"))

                                self.opd_update_steps += 1
                                self.actor_update_steps += 1
                        else:
                            metrics["hetero/opd_skip_no_candidate"] = 1
                    else:
                        metrics["hetero/opd_skip_by_schedule"] = 1

                    metrics["training/grpo_update_steps"] = self.grpo_update_steps
                    metrics["training/opd_update_steps"] = self.opd_update_steps
                    metrics["training/actor_update_steps"] = self.actor_update_steps

                    # =========================
                    # 6) validate / save / log
                    # =========================
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics = self._validate()
                        pprint(f"Validation metrics: {val_metrics}")
                        if is_last_step:
                            last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint(metrics=metrics)

                steps_duration = timing_raw.get("step", 0.0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_timing_metrics(batch=student_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(batch=student_batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self.is_hetero_distill:
            return self.fit_heterogeneous()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"val_metrics is empty"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                # generate policy model output
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        if self.opd_top_k > 0:
                            assert not bypass_recomputing_logprobs, (
                                "top-k OPD requires decoupled log-prob mode (bypass_mode=false)"
                            )
                            assert self.use_reference_policy, "top-k OPD requires a reference (teacher) policy"
                            with marked_timer("opd_topk_log_probs", timing_raw, color="blue"):
                                batch, entropys, opd_prep_timing = self._prepare_opd_batch(
                                    batch,
                                    include_base=False,
                                    ref_on_actor=self.ref_in_actor,
                                    ref_output_key="ref_log_prob",
                                    fetch_ref_externally=self.use_reference_policy and not self.ref_in_actor,
                                )
                                metrics.update(merge_worker_metrics(opd_prep_timing, prefix="opd"))
                                metrics["opd/top_k"] = self.opd_top_k
                                if entropys is not None:
                                    response_masks = batch.batch["response_mask"]
                                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                    entropy_agg = agg_loss(
                                        loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                    )
                                    metrics["actor/entropy"] = entropy_agg.detach().item()
                        else:
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(
                                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                            if self.use_reference_policy:
                                # compute reference log_prob
                                with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                    # Get apply_chat_template_kwargs from config if available
                                    apply_chat_template_kwargs = self.config.data.get(
                                        "apply_chat_template_kwargs", {}
                                    )

                                    # If ref model uses different tokenizer/prompt template, re-tokenize inputs for ref model
                                    if self.use_ref_retokenization:
                                        from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs

                                        batch, _ = prepare_ref_model_inputs(
                                            batch=batch,
                                            ref_tokenizer=self.ref_tokenizer,
                                            primary_tokenizer=self.tokenizer,
                                            cross_token_bridge=self.use_cross_token_bridge,
                                            apply_chat_template_kwargs=apply_chat_template_kwargs,
                                        )

                                        if not self.ref_in_actor:
                                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                        else:
                                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                        batch = batch.union(ref_log_prob)
                                        if self.use_cross_token_bridge:
                                            from verl.trainer.ppo.ref_input_utils import (
                                                align_ref_side_log_probs_to_primary,
                                            )

                                            batch, _ = align_ref_side_log_probs_to_primary(
                                                batch, keys=("ref_log_prob",)
                                            )

                                    else:
                                        # Standard ref model log prob computation
                                        if not self.ref_in_actor:
                                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                        else:
                                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                        batch = batch.union(ref_log_prob)

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in batch.batch.keys()'
               

                    if not hasattr(self, "_debug_sample_saved"):
                        self._debug_sample_saved = False

                    if not self._debug_sample_saved:
                        save_dataproto_single_sample_with_json_preview(
                            batch=batch,
                            save_path="./debug_data/sample_step{}_idx0.pt".format(self.global_steps),
                            json_path="./debug_data/sample_step{}_idx0_preview.json".format(self.global_steps),
                            idx=0,
                        )
                        self._debug_sample_saved = True

                    # Compute base model log probs for corrected reward computation
                    # This computes: base_log_prob from actor's base model (using input_ids)
                    # and base_ref_log_prob from ref's base model (using ref_input_ids)
                    if self.use_base_models:
                        with marked_timer("base_log_probs", timing_raw, color="green"):
                            # First compute base_ref_log_prob using ref's base model
                            # This uses ref_input_ids which may be present in batch
                            if not self.ref_in_actor:
                                base_ref_log_prob = self.ref_policy_wg.compute_base_ref_log_prob(batch)
                            else:
                                base_ref_log_prob = self.actor_rollout_wg.compute_base_ref_log_prob(batch)
                            batch = batch.union(base_ref_log_prob)
                            
                            # Now compute base_log_prob using actor's base model with input_ids
                            # We need to temporarily remove ref_input_ids to ensure compute_log_prob uses input_ids
                            ref_input_tensors = {}
                            if "ref_input_ids" in batch.batch:
                                ref_input_tensors["ref_input_ids"] = batch.batch.pop("ref_input_ids")
                            if "ref_attention_mask" in batch.batch:
                                ref_input_tensors["ref_attention_mask"] = batch.batch.pop("ref_attention_mask")
                            if "ref_position_ids" in batch.batch:
                                ref_input_tensors["ref_position_ids"] = batch.batch.pop("ref_position_ids")
                            
                            # Compute base_log_prob using actor's base model with input_ids
                            base_log_prob = self.actor_rollout_wg.compute_base_log_prob(batch)
                            batch = batch.union(base_log_prob)
                            
                            # Restore ref_input_ids tensors back to batch
                            for key, tensor in ref_input_tensors.items():
                                batch.batch[key] = tensor

                            if self.use_cross_token_bridge:
                                from verl.trainer.ppo.ref_input_utils import (
                                    align_ref_side_log_probs_to_primary,
                                )

                                batch, _ = align_ref_side_log_probs_to_primary(
                                    batch, keys=("base_ref_log_prob",)
                                )
                            
                            print(f"Computed base log probs for corrected reward: "
                                  f"base_log_prob shape={batch.batch['base_log_prob'].shape}, "
                                  f"base_ref_log_prob shape={batch.batch['base_ref_log_prob'].shape}") 
                    
                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(merge_worker_metrics(critic_output_metrics))

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(merge_worker_metrics(actor_output_metrics))

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint(metrics=metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
