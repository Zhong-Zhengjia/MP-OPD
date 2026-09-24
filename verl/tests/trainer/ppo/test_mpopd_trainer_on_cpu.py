# Copyright 2026 MP-OPD contributors
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

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.config.algorithm import MPOPDConfig
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.expert_prompt_utils import FOCUS_MARKER
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "fake"

    def __init__(self, vocab=None):
        self.vocab = dict(vocab or {"clean": 10, FOCUS_MARKER: 901})
        self._next_id = max(self.vocab.values()) + 1

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        result = []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = self._next_id
                self._next_id += 1
            result.append(self.vocab[token])
        return result

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=False, **kwargs):
        del kwargs
        text = " ".join(f"{item['role']} {item['content']}" for item in messages)
        if add_generation_prompt:
            text += " assistant"
        return self._tokenize(text) if tokenize else text

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        input_ids = torch.tensor([self._tokenize(text)], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return self._tokenize(text)


def _object_array(items):
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def _rollout_batch(*, experts_available=True):
    contexts = (
        {"chasing": {"enabled": True, "evidence_available": True, "evidence": ["clicked item"]}}
        if experts_available
        else {"chasing": {"enabled": False}}
    )
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[10, 11, 71, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
            "prompts": torch.tensor([[10, 11]], dtype=torch.long),
            "responses": torch.tensor([[71, 0]], dtype=torch.long),
        },
        non_tensors={
            "raw_prompt": _object_array([[{"role": "user", "content": "clean request"}]]),
            "expert_contexts": _object_array([contexts]),
        },
    )


class FakeWorkerGroup:
    def __init__(self):
        self.calls = []
        self.clean_ids = None

    def generate_sequences(self, batch):
        self.calls.append("generate_sequences")
        # HF rollout returns tensors only; the trainer must restore driver-only
        # prompt metadata before packing the specialized expert prompts.
        return DataProto(batch=batch.batch.clone(), meta_info=dict(batch.meta_info))

    def prepare_mpopd_log_probs(self, batch):
        self.calls.append("prepare_mpopd_log_probs")
        self.clean_ids = batch.batch["input_ids"].clone()
        assert "expert_contexts" not in batch.non_tensor_batch
        assert 901 not in self.clean_ids
        assert batch.meta_info["top_k"] == 2
        assert batch.meta_info["num_experts"] == 1
        shape = (len(batch), batch.batch["responses"].shape[1], 2)
        expert_shape = (*shape, 1)
        return DataProto.from_dict(
            tensors={
                "student_topk_ids": torch.zeros(shape, dtype=torch.long),
                "old_log_probs": torch.zeros(shape),
                "student_base_topk_log_probs": torch.zeros(shape),
                "teacher_base_topk_log_probs": torch.zeros(shape),
                "specialized_expert_topk_log_probs": torch.zeros(expert_shape),
            },
            meta_info={"metrics": {"timing/mpopd_prepare_total_s": [0.25]}},
        )

    def update_actor_mpopd(self, batch):
        self.calls.append("update_actor_mpopd")
        unavailable = (~batch.batch["expert_mask"].any(dim=-1)).sum().item()
        return DataProto(meta_info={"metrics": {"actor/mpopd_all_experts_unavailable_count": [unavailable]}})


def _fake_trainer(*, experts_available=True):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.train_mode = "multi_prompt_distill"
    trainer.is_mp_opd = True
    trainer.tokenizer = FakeTokenizer()
    trainer.ref_tokenizer = trainer.tokenizer
    trainer.mp_opd_config = MPOPDConfig(
        expert_names=["chasing"],
        expert_instructions={"chasing": "focus recent"},
        top_k=2,
        expert_temperature=0.7,
        token_temperature=1.2,
        lambda_value=0.5,
        expert_priors=[1.0],
        max_expert_prompt_length=64,
    )
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"temperature": 1.0}}})
    trainer.actor_rollout_wg = FakeWorkerGroup()
    trainer.reward_fn = SimpleNamespace(calls=[])
    trainer.rollout_batch = _rollout_batch(experts_available=experts_available)
    return trainer


def test_step_call_order_and_does_not_compute_reward():
    trainer = _fake_trainer()

    metrics = trainer._run_mpopd_step(trainer.rollout_batch, metrics={}, timing_raw={})

    assert trainer.actor_rollout_wg.calls == [
        "generate_sequences",
        "prepare_mpopd_log_probs",
        "update_actor_mpopd",
    ]
    assert trainer.reward_fn.calls == []
    assert metrics["actor/mpopd_all_experts_unavailable_count"] == 0


def test_update_preparation_preserves_clean_inputs_and_forwards_algorithm_metadata():
    trainer = _fake_trainer()
    batch = trainer._ensure_response_mask(trainer.rollout_batch)
    original_ids = batch.batch["input_ids"].clone()

    update_batch, prep_metrics = trainer._prepare_mpopd_update_batch(batch)

    assert torch.equal(update_batch.batch["input_ids"], original_ids)
    assert torch.equal(trainer.actor_rollout_wg.clean_ids, original_ids)
    assert "expert_contexts" not in update_batch.non_tensor_batch
    assert update_batch.meta_info["mpopd_expert_temperature"] == 0.7
    assert update_batch.meta_info["mpopd_token_temperature"] == 1.2
    assert update_batch.meta_info["mpopd_lambda_value"] == 0.5
    assert update_batch.meta_info["mpopd_expert_priors"] == [1.0]
    assert prep_metrics["timing/mpopd_prepare_total_s"] == 0.25


def test_all_unavailable_sample_is_reported_once():
    trainer = _fake_trainer(experts_available=False)

    metrics = trainer._run_mpopd_step(trainer.rollout_batch, metrics={}, timing_raw={})

    assert metrics["actor/mpopd_all_experts_unavailable_count"] == 1


def test_fit_routes_only_multi_prompt_mode_to_mpopd(monkeypatch):
    mp_trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    mp_trainer.is_mp_opd = True
    mp_trainer._fit_mpopd = lambda: "mpopd"
    assert mp_trainer.fit() == "mpopd"

    legacy_trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    legacy_trainer.is_mp_opd = False
    legacy_trainer.grpo_buffer = []
    legacy_trainer.config = OmegaConf.create(
        {"trainer": {"project_name": "p", "experiment_name": "e", "logger": []}}
    )
    monkeypatch.setattr(
        "verl.utils.tracking.Tracking",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("entered legacy route")),
    )
    with pytest.raises(RuntimeError, match="entered legacy route"):
        legacy_trainer.fit()


def _validation_trainer(*, vocab_matches=True, cross_bridge=False):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.mp_opd_config = MPOPDConfig(top_k=2)
    trainer.tokenizer = FakeTokenizer({"a": 1})
    trainer.ref_tokenizer = FakeTokenizer({"a": 1} if vocab_matches else {"b": 1})
    trainer.use_cross_token_bridge = cross_bridge
    trainer.config = OmegaConf.create(
        {
            "data": {"return_raw_chat": True},
            "actor_rollout_ref": {
                "model": {"base_model_path": "student-base"},
                "ref": {"model": {"path": "teacher", "base_model_path": "teacher-base"}},
            },
        }
    )
    return trainer


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda trainer: setattr(trainer, "mp_opd_config", SimpleNamespace(top_k=0)), "top_k"),
        (lambda trainer: trainer.config.data.update({"return_raw_chat": False}), "return_raw_chat"),
        (lambda trainer: trainer.config.actor_rollout_ref.model.update({"base_model_path": None}), "base_model"),
        (lambda trainer: trainer.config.actor_rollout_ref.ref.model.update({"path": None}), "reference expert"),
        (lambda trainer: setattr(trainer, "ref_tokenizer", None), "reference tokenizer"),
        (lambda trainer: setattr(trainer, "use_cross_token_bridge", True), "cross-token"),
        (
            lambda trainer: setattr(trainer, "ref_tokenizer", FakeTokenizer({"different": 9})),
            "vocabularies",
        ),
    ],
)
def test_mpopd_setup_validation_rejects_unsupported_states(mutation, message):
    trainer = _validation_trainer()
    mutation(trainer)

    with pytest.raises(ValueError, match=message):
        trainer._validate_mpopd_setup()


def test_mpopd_setup_validation_accepts_identical_vocabularies():
    _validation_trainer()._validate_mpopd_setup()


def test_trainer_initializes_dedicated_mode_without_critic_or_standalone_ref(monkeypatch):
    monkeypatch.setattr(RayPPOTrainer, "_create_dataloader", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.ValidationGenerationsLogger",
        lambda **kwargs: SimpleNamespace(),
    )
    config = OmegaConf.create(
        {
            "algorithm": {
                "train_mode": "multi_prompt_distill",
                "mp_opd": OmegaConf.structured(MPOPDConfig(top_k=2)),
                "use_kl_in_reward": False,
            },
            "data": {"return_raw_chat": True},
            "actor_rollout_ref": {
                "hybrid_engine": True,
                "model": {"base_model_path": "student-base", "lora_rank": 0},
                "ref": {"model": {"path": "teacher"}},
            },
            "critic": {"enable": True},
            "trainer": {"project_name": "p", "experiment_name": "e", "device": "cpu"},
        }
    )
    tokenizer = FakeTokenizer({"a": 1})

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        ref_tokenizer=FakeTokenizer({"a": 1}),
        role_worker_mapping={Role.ActorRollout: object, Role.RefPolicy: object},
        resource_pool_manager=SimpleNamespace(),
    )

    assert trainer.train_mode == "multi_prompt_distill"
    assert trainer.is_mp_opd is True
    assert trainer.use_critic is False
    assert trainer.ref_in_actor is True


def test_task_runner_registers_reference_policy_for_mpopd(monkeypatch):
    runner = TaskRunner()
    monkeypatch.setattr("verl.trainer.main_ppo.ray.remote", lambda cls: cls)
    config = OmegaConf.create(
        {
            "algorithm": {"train_mode": "multi_prompt_distill", "use_kl_in_reward": False},
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
        }
    )

    runner.add_ref_policy_worker(config, object)

    assert Role.RefPolicy in runner.role_worker_mapping
