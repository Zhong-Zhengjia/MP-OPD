我正在基于 verl 写一个异质系统模型训练的框架，这个框架包括一个 student model 和一个 teacher model。两者会各自对同一个 sample 去 rollout 轨迹（各自 8 条）。

然后我们会基于这 16 条轨迹采用两种训练策略。首先，我们将这 16 条轨迹分成 student rollouts 和 teacher rollouts。

- 对于 student rollouts 而言，如果它的 8 条轨迹全部做对了，则跳过这个 sample。
- 如果 student rollout 对错参半，则这个 8 条轨迹后续使用 grpo 训练。
- 如果 student rollout 的 8 条轨迹全错，而 teacher rollouts 中有对的轨迹，则进行 On-Policy Distillation (OPD)。
    - 所谓 OPD，是让 teacher model 在 student model 的错误轨迹上重新推理一遍（teacher forcing 推理），然后在每个 token 上计算 student model 和 teacher model 的 reverse kl 作为训练 loss。$\text{KL}\left(\pi_\theta \| \pi_{\text{teacher}}\right) = \mathbb{E}_{x \sim \pi_\theta} \left[ \log \pi_\theta(x_{t+1}|x_{1..t}) - \log \pi_{\text{teacher}}(x_{t+1}|x_{1..t}) \right]$
- 如果 student rollout 的 8 条轨迹全错，而 teacher rollouts 的轨迹也全错，则同样跳过这个 sample。

这样 teacher model 实际上就只为模型提供了新的推理思路（因为只有 student model 全错的时候才进行 OPD），然后 student model 在获得了新的推理思路后，还会主动在其他的问题上进行尝试，类似一种 learn & play 的训练方式

因此，在模型 rollout 轨迹之后，你需要对轨迹做判别处理，并构造成训练的 batch 用来更新模型权重。下面是训练主算法的代码，主要涉及资源的分配和轨迹 rollout。

```python
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


def union_gen_and_rollout_batch(gen_batch: DataProto, rollout_batch: DataProto) -> DataProto:
    prefer_rollout_tensor_keys = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "prompts",
    }

    gen_tensor_keys = list(gen_batch.batch.keys())
    rollout_tensor_keys = list(rollout_batch.batch.keys())

    overlap = set(gen_tensor_keys) & set(rollout_tensor_keys)

    illegal_conflicts = []
    for k in overlap:
        if k in prefer_rollout_tensor_keys:
            continue
        if not gen_batch.batch[k].equal(rollout_batch.batch[k]):
            illegal_conflicts.append(k)

    assert len(illegal_conflicts) == 0, (
        f"Unexpected conflicting keys during union: {illegal_conflicts}"
    )

    kept_tensor_keys = [k for k in gen_tensor_keys if k not in prefer_rollout_tensor_keys]

    gen_batch_trimmed = gen_batch.select(
        batch_keys=kept_tensor_keys,
        non_tensor_batch_keys=list(gen_batch.non_tensor_batch.keys()),
    )

    return gen_batch_trimmed.union(rollout_batch)


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
            return_raw_chat = config.data.get("return_raw_chat", False)
            if not return_raw_chat:
                raise ValueError(
                    "When using a different tokenizer for ref model (ref_tokenizer is provided) "
                    "you must set data.return_raw_chat=True in config to enable re-tokenization. "
                    "This is needed to access the original messages for re-tokenizing with ref model's chat template."
                )

        # training mode
        self.train_mode = self.config.algorithm.get("train_mode", "ppo")
        self.is_hetero_distill = self.train_mode == "heterogeneous_distill"

        hetero_cfg = self.config.algorithm.get("hetero_distill", {})
        self.student_rollout_n = hetero_cfg.get("student_rollout_n", 1)
        self.teacher_rollout_n = hetero_cfg.get("teacher_rollout_n", 1)
        self.use_sdft = hetero_cfg.get("use_sdft", True)
        self.use_icl_opd = hetero_cfg.get("use_icl_opd", True)
        self.sdft_weight = hetero_cfg.get("sdft_weight", 1.0)
        self.icl_opd_weight = hetero_cfg.get("icl_opd_weight", 1.0)
        self.sample_demo_strategy = hetero_cfg.get("sample_demo_strategy", "random")

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

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
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

        train_mode = self.config.algorithm.get("train_mode", None)

        def _spawn_single_worker_group(role, ray_cls_with_init, prefix_name=None):
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            wg = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                **wg_kwargs,
            )
            key = prefix_name or str(role)
            spawned = wg.spawn(prefix_set={key})
            return spawned[key]

        # heterogeneous_distill: do NOT colocate different roles
        if train_mode == "heterogeneous_distill":
            if self.hybrid_engine:
                actor_rollout_role = "actor_rollout_ref"
                actor_rollout_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ActorRollout],
                    config=self.config.actor_rollout_ref,
                    role=actor_rollout_role,
                )
                all_wg[str(Role.ActorRollout)] = _spawn_single_worker_group(
                    Role.ActorRollout, actor_rollout_cls, prefix_name=str(Role.ActorRollout)
                )
            else:
                raise NotImplementedError

            teacher_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.TeacherRollout],
                config=self.config,
                role=str(Role.TeacherRollout),
            )
            all_wg[str(Role.TeacherRollout)] = _spawn_single_worker_group(
                Role.TeacherRollout, teacher_rollout_cls
            )

            if self.use_critic:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
                critic_cfg = omega_conf_to_dataclass(self.config.critic)
                critic_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.Critic],
                    config=critic_cfg,
                )
                critic_wg = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=critic_cls,
                    **wg_kwargs,
                )
                all_wg[str(Role.Critic)] = critic_wg.spawn(prefix_set={str(Role.Critic)})[str(Role.Critic)]

            if self.use_reference_policy and not self.ref_in_actor:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
                ref_policy_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RefPolicy],
                    config=self.config.actor_rollout_ref,
                    role=str(Role.RefPolicy),
                )
                ref_wg = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=ref_policy_cls,
                    **wg_kwargs,
                )
                all_wg[str(Role.RefPolicy)] = ref_wg.spawn(prefix_set={str(Role.RefPolicy)})[str(Role.RefPolicy)]

            if self.use_rm:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel],
                    config=self.config.reward_model,
                )
                rm_wg = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=rm_cls,
                    **wg_kwargs,
                )
                all_wg[str(Role.RewardModel)] = rm_wg.spawn(prefix_set={str(Role.RewardModel)})[str(Role.RewardModel)]

        else:
            self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

            if self.hybrid_engine:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                actor_rollout_role = str(Role.ActorRollout)
                actor_rollout_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ActorRollout],
                    config=self.config.actor_rollout_ref,
                    role=actor_rollout_role,
                )
                self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
            else:
                raise NotImplementedError

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
        if train_mode == "heterogeneous_distill":
            self.teacher_rollout_wg = all_wg[str(Role.TeacherRollout)]
            self.teacher_rollout_wg.init_model()

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
            max_ckpt_to_keep=1,
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
                max_ckpt_to_keep=1,
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

    def _build_hetero_distill_batch(
        self,
        base_batch: DataProto,
        student_batch: DataProto,
        teacher_batch: DataProto,
        student_reward_tensor: torch.Tensor,
        teacher_reward_tensor: torch.Tensor,
    ) -> DataProto:
        import numpy as np
        import torch
        import uuid

        if base_batch is None or student_batch is None or teacher_batch is None:
            return None
        if "input_ids" not in base_batch.batch:
            return None
        if "responses" not in student_batch.batch or "responses" not in teacher_batch.batch:
            return None
        if "input_ids" not in student_batch.batch:
            return None
        if base_batch.batch["input_ids"].shape[0] == 0:
            return None
        if student_batch.batch["responses"].shape[0] == 0 or teacher_batch.batch["responses"].shape[0] == 0:
            return None
        if student_batch.batch["input_ids"].shape[0] == 0:
            return None
        if student_reward_tensor is None or teacher_reward_tensor is None:
            return None
        if student_reward_tensor.numel() == 0 or teacher_reward_tensor.numel() == 0:
            return None

        student_correct = self._compute_binary_correctness_from_reward_tensor(student_reward_tensor).cpu()
        teacher_correct = self._compute_binary_correctness_from_reward_tensor(teacher_reward_tensor).cpu()

        base_bs = len(base_batch.batch["input_ids"])
        student_n = self.student_rollout_n
        teacher_n = self.teacher_rollout_n

        if student_n <= 0 or teacher_n <= 0:
            return None

        expected_student = base_bs * student_n
        expected_teacher = base_bs * teacher_n

        if student_reward_tensor.shape[0] != expected_student:
            return None
        if teacher_reward_tensor.shape[0] != expected_teacher:
            return None
        if student_batch.batch["responses"].shape[0] != expected_student:
            return None
        if teacher_batch.batch["responses"].shape[0] != expected_teacher:
            return None
        if student_batch.batch["input_ids"].shape[0] != expected_student:
            return None

        # Optional tensor fields in student_batch
        has_student_attention_mask = "attention_mask" in student_batch.batch
        has_student_position_ids = "position_ids" in student_batch.batch
        has_student_response_mask = "response_mask" in student_batch.batch

        student_resp_texts = self._extract_response_texts_from_batch(student_batch)
        teacher_resp_texts = self._extract_response_texts_from_batch(teacher_batch)

        prompt_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in base_batch.batch["input_ids"]
        ]

        if "uid" in base_batch.non_tensor_batch:
            uids = list(base_batch.non_tensor_batch["uid"])
        else:
            uids = [str(uuid.uuid4()) for _ in range(base_bs)]

        critic_icl_opd = getattr(self.config.trainer, "critic_icl_opd", False)

        tensor_input_ids = []
        tensor_attention_mask = []
        tensor_position_ids = []
        tensor_responses = []
        tensor_response_mask = []

        non_tensor_distill_type = []
        non_tensor_demo_text = []
        non_tensor_wrong_response_text = []
        non_tensor_prompt_text = []
        non_tensor_uid = []

        def choose_one(items):
            if len(items) == 0:
                return None
            if self.sample_demo_strategy == "random":
                idx = np.random.randint(0, len(items))
                return items[idx]
            return items[0]

        def build_wrong_trajectory_from_student(j):
            """
            Directly reuse student wrong trajectory tensors from student_batch.
            Assumes student_batch.batch["input_ids"] already contains full prompt + response token ids.
            """
            input_ids = student_batch.batch["input_ids"][j]

            if has_student_attention_mask:
                attention_mask = student_batch.batch["attention_mask"][j]
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            if has_student_position_ids:
                position_ids = student_batch.batch["position_ids"][j]
            else:
                position_ids = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

            responses = student_batch.batch["responses"][j]

            if has_student_response_mask:
                response_mask = student_batch.batch["response_mask"][j]
            else:
                response_mask = torch.ones_like(responses, dtype=torch.long)

            if response_mask.numel() == 0 or response_mask.sum().item() == 0:
                return None

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            }

        def append_sample(encoded_item, distill_type, demo_text, wrong_text, prompt_text, uid):
            tensor_input_ids.append(encoded_item["input_ids"])
            tensor_attention_mask.append(encoded_item["attention_mask"])
            tensor_position_ids.append(encoded_item["position_ids"])
            tensor_responses.append(encoded_item["responses"])
            tensor_response_mask.append(encoded_item["response_mask"])

            non_tensor_distill_type.append(distill_type)
            non_tensor_demo_text.append(demo_text)
            non_tensor_wrong_response_text.append(wrong_text)
            non_tensor_prompt_text.append(prompt_text)
            non_tensor_uid.append(uid)

        for i in range(base_bs):
            s_l = i * student_n
            s_r = (i + 1) * student_n
            t_l = i * teacher_n
            t_r = (i + 1) * teacher_n

            student_correct_texts = [
                student_resp_texts[j] for j in range(s_l, s_r) if bool(student_correct[j].item())
            ]
            student_wrong_items = [
                (j, student_resp_texts[j]) for j in range(s_l, s_r) if not bool(student_correct[j].item())
            ]
            teacher_correct_texts = [
                teacher_resp_texts[j] for j in range(t_l, t_r) if bool(teacher_correct[j].item())
            ]

            student_has_correct = len(student_correct_texts) > 0
            student_has_wrong = len(student_wrong_items) > 0
            teacher_has_correct = len(teacher_correct_texts) > 0

            # 1) student 全对：跳过
            if not student_has_wrong:
                continue

            # 2) student 和 teacher 全错：跳过
            if (not student_has_correct) and (not teacher_has_correct):
                continue

            prompt_text = prompt_texts[i]
            uid = uids[i]

            # Directly build all wrong student trajectories once, then reuse for sdft / icl_opd
            encoded_wrong_items = []
            for j, wrong_text in student_wrong_items:
                encoded_item = build_wrong_trajectory_from_student(j)
                if encoded_item is None:
                    continue
                encoded_wrong_items.append((wrong_text, encoded_item))

            if len(encoded_wrong_items) == 0:
                continue

            # SDFT samples: use student correct demonstration on student wrong trajectories
            if self.use_sdft and len(student_correct_texts) > 0:
                demo_text = choose_one(student_correct_texts)
                for wrong_text, encoded_item in encoded_wrong_items:
                    append_sample(
                        encoded_item=encoded_item,
                        distill_type="sdft",
                        demo_text=demo_text,
                        wrong_text=wrong_text,
                        prompt_text=prompt_text,
                        uid=uid,
                    )

            # ICL-OPD samples: use teacher correct demonstration on student wrong trajectories
            # If critic_icl_opd=True, teacher only guides when student is all wrong.
            allow_teacher_guide = False
            if self.use_icl_opd and len(teacher_correct_texts) > 0:
                if critic_icl_opd:
                    allow_teacher_guide = not student_has_correct
                else:
                    allow_teacher_guide = True

            if allow_teacher_guide:
                demo_text = choose_one(teacher_correct_texts)
                for wrong_text, encoded_item in encoded_wrong_items:
                    append_sample(
                        encoded_item=encoded_item,
                        distill_type="icl_opd",
                        demo_text=demo_text,
                        wrong_text=wrong_text,
                        prompt_text=prompt_text,
                        uid=uid,
                    )

        if len(tensor_input_ids) == 0:
            return None

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_input_len = max(x.shape[0] for x in tensor_input_ids)
        max_resp_len = max(x.shape[0] for x in tensor_responses)

        def pad_1d(x, max_len, pad_value=0):
            if x.shape[0] == max_len:
                return x
            pad = torch.full((max_len - x.shape[0],), pad_value, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_1d(x, max_input_len, pad_id) for x in tensor_input_ids], dim=0)
        attention_mask = torch.stack([pad_1d(x, max_input_len, 0) for x in tensor_attention_mask], dim=0)
        position_ids = torch.stack([pad_1d(x, max_input_len, 0) for x in tensor_position_ids], dim=0)
        responses = torch.stack([pad_1d(x, max_resp_len, pad_id) for x in tensor_responses], dim=0)
        response_mask = torch.stack([pad_1d(x, max_resp_len, 0) for x in tensor_response_mask], dim=0)

        distill_batch = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            },
            non_tensors={
                "distill_type": np.array(non_tensor_distill_type, dtype=object),
                "demo_text": np.array(non_tensor_demo_text, dtype=object),
                "wrong_response_text": np.array(non_tensor_wrong_response_text, dtype=object),
                "prompt_text": np.array(non_tensor_prompt_text, dtype=object),
                "uid": np.array(non_tensor_uid, dtype=object),
            },
            meta_info={
                "distill_mode": "heterogeneous_distill",
                "sdft_weight": self.sdft_weight,
                "icl_opd_weight": self.icl_opd_weight,
                "critic_icl_opd": critic_icl_opd,
            },
        )

        multiple = 6
        bsz = len(distill_batch)
        keep = (bsz // multiple) * multiple
        if keep == 0:
            return None
        if keep != bsz:
            distill_batch = distill_batch[:keep]

        print(f'[DEBUG] Hetero_distill_batch_size: {len(distill_batch)}')
        return distill_batch

    def fit_heterogeneous_distill(self):
        from omegaconf import OmegaConf
        from pprint import pprint
        from tqdm import tqdm
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint first
        # self._load_checkpoint()

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

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
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

                

                # =========================
                # 1) student and teacher rollout
                # =========================
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


                teacher_gen_batch = self._repeat_base_batch_for_rollout_n(batch, self.teacher_rollout_n)
                if self._is_empty_dataproto(teacher_gen_batch):
                    metrics["hetero/skip_empty_teacher_gen_batch"] = 1

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
                teacher_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "do_sample": True,
                    "validate": False,
                    "global_steps": self.global_steps,
                }


                rollout_start = time.time()
                student_future = self.actor_rollout_wg.generate_sequences_async(student_gen_batch)
                teacher_future = self.teacher_rollout_wg.generate_sequences_async(teacher_gen_batch)
                
                student_rollout_batch, teacher_rollout_batch = RayAsyncCollectHandle.gather(
                    [student_future, teacher_future]
                )
                rollout_end = time.time()

                print(f"[DEBUG] Parallel student+teacher rollout time cost: {rollout_end-rollout_start} s")

                student_batch = union_gen_and_rollout_batch(student_gen_batch, student_rollout_batch)

                # =====================================
                student_average_response_length = 0 
                student_batch_size = len(student_batch.batch["prompts"])
                print(f'[DEBUG] Student rollout batch size: {student_batch_size}')
                for i in range(student_batch_size):
                    prompt_ids = student_batch.batch["prompts"][i]
                    prompt_length = prompt_ids.shape[0]
                    attention_mask = student_batch.batch["attention_mask"][i]
                    valid_response_length = attention_mask[prompt_length:].sum().item()
                    student_average_response_length += valid_response_length
                print(f'[DEBUG] Student average response length: {student_average_response_length/student_batch_size}')

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
                # =====================================

                if "response_mask" not in student_batch.batch:
                    student_batch.batch["response_mask"] = compute_response_mask(student_batch)

                teacher_batch = union_gen_and_rollout_batch(teacher_gen_batch, teacher_rollout_batch)

                # ===================================== 
                teacher_average_response_length = 0
                teacher_batch_size = len(teacher_batch.batch["prompts"])
                print(f'[DEBUG] Teacher rollout batch size: {teacher_batch_size}')
                for i in range(teacher_batch_size):
                    prompt_ids = teacher_batch.batch["prompts"][i]
                    prompt_length = prompt_ids.shape[0]
                    attention_mask = teacher_batch.batch["attention_mask"][i]
                    valid_response_length = attention_mask[prompt_length:].sum().item()
                    teacher_average_response_length += valid_response_length
                print(f'[DEBUG] Teacher average response length: {teacher_average_response_length/teacher_batch_size}')

                i= 0
                prompt_ids = teacher_batch.batch["prompts"][i]
                prompt_length = prompt_ids.shape[0]
                attention_mask = teacher_batch.batch["attention_mask"][i]
                response_ids = teacher_batch.batch["responses"][i]
                valid_response_length = attention_mask[prompt_length:].sum().item()
                valid_response_ids = response_ids[:valid_response_length]
                response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                print("[DEBUG] Teacher response_text:")
                print(repr(response_text))
                # =====================================

                if "response_mask" not in teacher_batch.batch:
                    teacher_batch.batch["response_mask"] = compute_response_mask(teacher_batch)

                # =========================
                # 3) reward / correctness
                # =========================
                student_reward_tensor, student_reward_extra_infos = compute_reward(student_batch, self.reward_fn)
                teacher_reward_tensor, teacher_reward_extra_infos = compute_reward(teacher_batch, self.reward_fn)

                # =========================
                # 4) build distillation batch
                # =========================
                distill_batch = self._build_hetero_distill_batch(
                    base_batch=batch,
                    student_batch=student_batch,
                    teacher_batch=teacher_batch,
                    student_reward_tensor=student_reward_tensor,
                    teacher_reward_tensor=teacher_reward_tensor,
                )

                student_correct = self._compute_binary_correctness_from_reward_tensor(student_reward_tensor)
                teacher_correct = self._compute_binary_correctness_from_reward_tensor(teacher_reward_tensor)

                metrics["hetero/student_rollout_correct_rate"] = student_correct.float().mean().item()
                metrics["hetero/teacher_rollout_correct_rate"] = teacher_correct.float().mean().item()
                metrics["hetero/student_rollout_count"] = int(student_correct.numel())
                metrics["hetero/teacher_rollout_count"] = int(teacher_correct.numel())

                if distill_batch is None or len(distill_batch.batch["input_ids"]) == 0:
                    metrics["hetero/distill_sample_count"] = 0

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return
                    continue

                metrics["hetero/distill_sample_count"] = len(distill_batch.batch["input_ids"])
                distill_types = list(distill_batch.non_tensor_batch["distill_type"])
                metrics["hetero/sdft_sample_count"] = sum(1 for x in distill_types if x == "sdft")
                metrics["hetero/icl_opd_sample_count"] = sum(1 for x in distill_types if x == "icl_opd")

                # =========================
                # 5) update student by distillation
                # =========================
                actor_output = self.actor_rollout_wg.update_actor_distill(distill_batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                # =========================
                # 6) validate / save / log
                # =========================
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    val_metrics = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint(metrics=metrics)

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
```

可以看到之前的这个代码处理轨迹的逻辑是有不同的，之前虽然 student 和 teacher 都进行 rollout，但是其实是根据轨迹中的正确轨迹和错误轨迹进行组合，并根据正确轨迹的来源打上 sdft (对 student model 使用正确轨迹作为 context 构造 self-teacher 来做 opd) 和 icl-opd （加上正确轨迹让 teacher model 做 opd）。

此外，如果使用现在构想的新的算法，则不需要使用 union_gen_and_rollout_batch 来处理 batch 的融合了，不需要在 non_tensor_batches 中添加 demo_text 和 wrong_response 了。

这里我需要你帮我修改这个代码，同样需要构造两个训练 batch，一个是 grpo 的 batch，一个是 opd 的 batch（这里不用 icl-opd 了，我发现加上正确轨迹之后，模型反而会放弃思考，得分反而下降）。后续我会根据不同的训练 batch 采取不同的训练方法。

在 `_build_hetero_distill_batch()` 函数的最后，我还加入了下面的代码：

```python
        multiple = 6
        bsz = len(distill_batch)
        keep = (bsz // multiple) * multiple
        if keep == 0:
            return None
        if keep != bsz:
            distill_batch = distill_batch[:keep]
```

这是因为我给 teacher rollout worker 单独分配了两卡的资源，所以 student model 这里实际上只有 6 卡，这里你可能还需要分析一下，如果我 grpo 的一个 group 有 8 条轨迹，训练会不会出问题。

下面是我的训练参数设置：

```bash
python3 -m verl.trainer.main_ppo \
    +algorithm.train_mode=heterogeneous_distill \
    +algorithm.hetero_distill.student_rollout_n=8 \
    +algorithm.hetero_distill.teacher_rollout_n=8 \
    +algorithm.hetero_distill.use_sdft=true \
    +algorithm.hetero_distill.use_icl_opd=true \
    +algorithm.hetero_distill.sdft_weight=1.0 \
    +algorithm.hetero_distill.icl_opd_weight=1.0 \
    +algorithm.hetero_distill.sample_demo_strategy=random \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=288 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=3245 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path=$student_model_path \
    +actor_rollout_ref.model.base_model_path=$student_model_path \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=576 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=false \
    actor_rollout_ref.actor.policy_loss.multi_teacher_distill=false \
    teacher_rollout.model.path=$teacher_model_path \
    teacher_rollout.model.use_remove_padding=true \
    teacher_rollout.actor.optim.lr_warmup_steps_ratio=0.0 \
    teacher_rollout.actor.ppo_mini_batch_size=8 \
    teacher_rollout.actor.ppo_micro_batch_size_per_gpu=1 \
    teacher_rollout.actor.ppo_max_token_len_per_gpu=9216 \
    teacher_rollout.actor.fsdp_config.param_offload=true \
    teacher_rollout.actor.fsdp_config.optimizer_offload=true \
    teacher_rollout.model.enable_gradient_checkpointing=true \
    teacher_rollout.rollout.log_prob_micro_batch_size_per_gpu=4 \
    teacher_rollout.rollout.tensor_model_parallel_size=2 \
    teacher_rollout.rollout.data_parallel_size=1 \
    teacher_rollout.rollout.pipeline_model_parallel_size=1 \
    teacher_rollout.rollout.name=vllm \
    teacher_rollout.rollout.mode=sync \
    teacher_rollout.rollout.free_cache_engine=true \
    teacher_rollout.rollout.gpu_memory_utilization=0.92 \
    teacher_rollout.rollout.n=1 \
    teacher_rollout.rollout.max_num_batched_tokens=9216 \
    teacher_rollout.rollout.temperature=1.0 \
    teacher_rollout.rollout.top_p=1.0 \
    teacher_rollout.rollout.val_kwargs.do_sample=True \
    teacher_rollout.rollout.val_kwargs.temperature=1.0 \
    teacher_rollout.rollout.val_kwargs.top_p=1.0 \
    teacher_rollout.rollout.val_kwargs.n=1 \
    teacher_rollout.rollout.calculate_log_probs=false \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_is_threshold=null \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=true \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
    trainer.project_name='heterogeneous-distillation' \
    trainer.experiment_name="${output_model_name}_${method}_${today}" \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=$n_node \
    +trainer.critic_icl_opd=false \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.save_freq=10 \
    +trainer.best_metric_name="val-core/DeepMath-103K/reward/mean@1" \
    +trainer.best_metric_mode="max" \
    trainer.default_local_dir=$output_path \
    trainer.test_freq=10 \
    trainer.total_epochs=2 $@
```

然后还有 actor_rollout_wg 的 generate_sequences 的代码（这里是同步版本，实际应用的时候采用异步可以加速 rollout 速度）：

```python
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        # ===============================
        # input_ids = prompts.batch["input_ids"]
        # print('[DEBUG] Student Prompts', self.tokenizer.decode(input_ids[0], skip_special_tokens=False))

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            self._run_coro_blocking(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            self._run_coro_blocking(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output
```

以及 vllm rollout engine 的 generate_sequences 代码：

```python
    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
```

同时我保证训练的时候是同族模型，也就是 tokenizer 是相同的。

现在，请你分析有哪些代码需要改动，该如何改动。尤其需要帮我分析一下在构造训练 batch 的时候，该怎么处理 rollout_n=8 和显卡数量 = 6 之间的关系。你也可以帮我分析一下如何设置 batch_size 等信息更好。如果你需要更多的代码，请告诉我，我会给你补充。