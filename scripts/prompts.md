我正在写一个模型训练算法，下面是这个代码的模型参数更新部分的代码

```python
    def _forward_micro_batch_logits(self, micro_batch) -> torch.Tensor:
        """
        Return logits on response positions only.

        Args:
            micro_batch: dict containing
                - input_ids
                - attention_mask
                - position_ids
                - responses

        Returns:
            logits: (bsz, response_len, vocab_size)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                return_dict=True,
            )

            logits = output.logits
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_len, vocab_size)
            return logits

    def _build_distill_inputs_for_micro_batch(self, micro_batch, tokenizer):
        """
        Build tokenized target inputs for teacher/student-base with demonstration context.

        Returns:
            dict with:
                input_ids
                attention_mask
                position_ids
                responses
                response_mask
                distill_types
        """

        prompt_texts = micro_batch.non_tensor_batch["prompt_text"]
        demo_texts = micro_batch.non_tensor_batch["demo_text"]
        wrong_response_texts = micro_batch.non_tensor_batch["wrong_response_text"]
        distill_types = micro_batch.non_tensor_batch["distill_type"]

        target_input_ids_list = []
        target_attention_mask_list = []
        target_position_ids_list = []
        target_responses_list = []
        target_response_mask_list = []

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        for prompt_text, demo_text, wrong_resp_text in zip(prompt_texts, demo_texts, wrong_response_texts):
            prefix_text = (
                f"{prompt_text}\n\n"
                f"Correct solution:\n{demo_text}\n\n"
                f"Correctly solve the original question:\n"
            )
            full_text = prefix_text + wrong_resp_text

            encoded_prefix = tokenizer(
                prefix_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded_full = tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
            )

            prefix_ids = encoded_prefix["input_ids"][0]
            full_ids = encoded_full["input_ids"][0]
            response_ids = full_ids[prefix_ids.shape[0]:]

            if response_ids.numel() == 0:
                # fallback: use tokenizer on wrong response directly
                encoded_resp = tokenizer(
                    wrong_resp_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
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
            "distill_types": distill_types,
        }

    def _kl_divergence_with_logits(self, student_logits, target_logits, response_mask):
        """
        Args:
            student_logits: (bsz, resp_len, vocab)
            target_logits: (bsz, resp_len, vocab)
            response_mask:  (bsz, resp_len)

        Returns:
            scalar loss
            token_kl: (bsz, resp_len)
        """
        import torch
        import torch.nn.functional as F

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)

        token_kl = F.kl_div(student_log_probs, target_probs, reduction="none").sum(dim=-1)
        masked_kl = token_kl * response_mask.float()

        denom = response_mask.float().sum().clamp_min(1.0)
        loss = masked_kl.sum() / denom
        return loss, token_kl

    @GPUMemoryLogger(role="dp actor distill", logger=logger)
    def update_policy_distill(self, data: DataProto, teacher_actor, student_base_actor, tokenizer):
        """
        Heterogeneous distillation update.

        data contains:
            tensor batch:
                - input_ids: original prompt + wrong response
                - attention_mask
                - position_ids
                - responses
                - response_mask
            non_tensor batch:
                - distill_type: "sdft" | "icl_opd"
                - demo_text
                - wrong_response_text
                - prompt_text
                - uid

        teacher_actor: frozen teacher model wrapper (DataParallelPPOActor)
        student_base_actor: frozen student base model wrapper (DataParallelPPOActor)
        tokenizer: tokenizer used to build in-context target inputs
        """
        import torch
        import torch.nn.functional as F

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

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                    gradient_accumulation = len(micro_batches)
                else:
                    gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    response_mask = micro_batch.batch["response_mask"].float()

                    # 1. student logits on original wrong trajectory
                    student_logits = self._forward_micro_batch_logits(micro_batch.batch)

                    # 2. build target inputs with demonstration context
                    target_inputs = self._build_distill_inputs_for_micro_batch(micro_batch, tokenizer)

                    target_input_ids = target_inputs["input_ids"].to(get_device_id())
                    target_attention_mask = target_inputs["attention_mask"].to(get_device_id())
                    target_position_ids = target_inputs["position_ids"].to(get_device_id())
                    target_responses = target_inputs["responses"].to(get_device_id())
                    target_response_mask = target_inputs["response_mask"].to(get_device_id())
                    distill_types = list(target_inputs["distill_types"])

                    batch_size = target_input_ids.shape[0]
                    vocab_size = student_logits.shape[-1]
                    target_logits = torch.zeros(
                        batch_size,
                        target_responses.shape[1],
                        vocab_size,
                        device=student_logits.device,
                        dtype=student_logits.dtype,
                    )

                    sdft_mask = torch.zeros(batch_size, device=student_logits.device, dtype=torch.float32)
                    icl_opd_mask = torch.zeros(batch_size, device=student_logits.device, dtype=torch.float32)

                    # 3. route each sample to target model
                    sdft_indices = [i for i, t in enumerate(distill_types) if t == "sdft"]
                    icl_indices = [i for i, t in enumerate(distill_types) if t == "icl_opd"]

                    # helper to gather a sub-batch
                    def gather_sub_batch(indices):
                        return {
                            "input_ids": target_input_ids[indices],
                            "attention_mask": target_attention_mask[indices],
                            "position_ids": target_position_ids[indices],
                            "responses": target_responses[indices],
                        }

                    # 3.1 student-base logits for SDFT
                    if len(sdft_indices) > 0:
                        sdft_mask[sdft_indices] = 1.0
                        sub_batch = gather_sub_batch(sdft_indices)

                        with torch.no_grad():
                            target_module = student_base_actor.actor_module
                            target_module.eval()
                            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                                output = target_module(
                                    input_ids=sub_batch["input_ids"],
                                    attention_mask=sub_batch["attention_mask"],
                                    position_ids=sub_batch["position_ids"],
                                    use_cache=False,
                                    return_dict=True,
                                )
                                sub_logits = output.logits[:, -sub_batch["responses"].shape[1] - 1 : -1, :]

                        target_logits[sdft_indices, : sub_logits.shape[1], :] = sub_logits

                    # 3.2 teacher logits for ICL-OPD
                    if len(icl_indices) > 0:
                        icl_opd_mask[icl_indices] = 1.0
                        sub_batch = gather_sub_batch(icl_indices)

                        with torch.no_grad():
                            target_module = teacher_actor.actor_module
                            target_module.eval()
                            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                                output = target_module(
                                    input_ids=sub_batch["input_ids"],
                                    attention_mask=sub_batch["attention_mask"],
                                    position_ids=sub_batch["position_ids"],
                                    use_cache=False,
                                    return_dict=True,
                                )
                                sub_logits = output.logits[:, -sub_batch["responses"].shape[1] - 1 : -1, :]

                        target_logits[icl_indices, : sub_logits.shape[1], :] = sub_logits

                    # 4. align lengths if necessary
                    resp_len = min(student_logits.shape[1], target_logits.shape[1], response_mask.shape[1], target_response_mask.shape[1])
                    student_logits = student_logits[:, :resp_len, :]
                    target_logits = target_logits[:, :resp_len, :]
                    response_mask_local = response_mask[:, :resp_len] * target_response_mask[:, :resp_len].float()

                    # 5. KL loss
                    token_student_log_probs = F.log_softmax(student_logits, dim=-1)
                    token_target_probs = F.softmax(target_logits, dim=-1)
                    token_kl = F.kl_div(token_student_log_probs, token_target_probs, reduction="none").sum(dim=-1)

                    sdft_token_mask = response_mask_local * sdft_mask.unsqueeze(-1)
                    icl_opd_token_mask = response_mask_local * icl_opd_mask.unsqueeze(-1)

                    sdft_denom = sdft_token_mask.sum().clamp_min(1.0)
                    icl_opd_denom = icl_opd_token_mask.sum().clamp_min(1.0)

                    sdft_loss = (token_kl * sdft_token_mask).sum() / sdft_denom
                    icl_opd_loss = (token_kl * icl_opd_token_mask).sum() / icl_opd_denom

                    total_loss = sdft_weight * sdft_loss + icl_opd_weight * icl_opd_loss
                    total_loss = total_loss / max(gradient_accumulation, 1)

                    if self.scaler is not None:
                        self.scaler.scale(total_loss).backward()
                    else:
                        total_loss.backward()

                    micro_metrics = {
                        "distill/loss": total_loss.detach().item(),
                        "distill/sdft_loss": sdft_loss.detach().item(),
                        "distill/icl_opd_loss": icl_opd_loss.detach().item(),
                        "distill/sdft_samples": float(len(sdft_indices)),
                        "distill/icl_opd_samples": float(len(icl_indices)),
                        "distill/mean_token_kl": (token_kl * response_mask_local).sum().detach().item()
                        / response_mask_local.sum().clamp_min(1.0).item(),
                    }
                    append_to_dict(metrics, micro_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
```

可以看到，这个代码主要使用 `update_policy_distill()` 函数来更新模型参数。我需要你帮我分析以下几个问题，以保证这些代码能准确完成我的目标。

首先，这个代码的目标是利用异质模型进行在线蒸馏来提高模型的能力。这里一共用到了三个模型，分别是 student model， student base model（参数和student model 一致，但是不更新） 和 teacher mdoel （和 student model 不一致，也不更新）。整个算法分成 rollout 阶段和参数更新阶段，rollout 阶段的代码实现在算法的主循环中。这里主要展示模型参数更新的部分。

在 rollout 阶段，对于每个问题，student model 和 teacher model 都会 rollout n 次。然后我们会判断这些 rollout 是否正确。如果 student 的轨迹全部正确，则不做任何处理。如果 student 的轨迹有错误，则需要在错误的轨迹上进行 On-policy 的蒸馏。当 student 的轨迹有错误的时候，则查看 student 和 teacher 的轨迹是否有正确的，如果 student 的轨迹有正确的，则进行 SDFT 的在线蒸馏。如果 teacher 轨迹有正确的，则进行 ICL-OPD 的蒸馏。

SDFT (self-distillation finetuning) 指 student base model 以正确的 student 轨迹作为 demonstration 在 student 错误的轨迹上重新推理，在每个 token 上计算 student base model 和 student model 的 kl loss，然后来更新 student model。 ICL-OPD (in-context-learning on-policy distillation) 则是 teacher model 以正确的 teacher 轨迹作为 demonstration 在 student 错误的轨迹上重新推理，在每个 token 上计算 kl loss 来监督更新 student model。

现在，请帮我分析一下上面的代码，是否完成了我的要求，我主要有以下几个问题想要确认：
- 上述的代码，是逐 token 获得 logits 并计算 kl loss 的吗？为什么我看到是在 microbatch 上计算 logits 
- 模型更新的方向应该是 student model 的 logits 分布向 teacher model 或者 student base model 的 logits 分布更新，这个代码的计算逻辑正确吗（更新方向正确吗）？为什么代码计算的时候是用的 log_probs 来计算，而不是直接用分布。