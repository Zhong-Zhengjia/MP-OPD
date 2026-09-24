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

from contextlib import nullcontext
from types import MethodType

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import build_mpopd_target, compute_mpopd_kl_loss
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def test_only_student_gets_gradient():
    student = torch.nn.Parameter(torch.tensor([[[0.0, 0.0]]]))
    target = torch.tensor([[[0.9, 0.1]]])

    loss = compute_mpopd_kl_loss(
        student,
        target,
        torch.ones(1, 1),
        torch.tensor([True]),
    )
    loss.backward()

    assert student.grad is not None
    assert student.grad[0, 0, 0] < 0
    assert target.grad is None


def test_invalid_sample_has_zero_gradient():
    student = torch.zeros(1, 1, 2, requires_grad=True)

    loss = compute_mpopd_kl_loss(
        student,
        torch.tensor([[[0.5, 0.5]]]),
        torch.ones(1, 1),
        torch.tensor([False]),
    )
    loss.backward()

    assert torch.equal(student.grad, torch.zeros_like(student))


def test_target_is_detached():
    target = build_mpopd_target(
        torch.zeros(1, 1, 2, 1, requires_grad=True),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
    )

    assert not target.target_probs.requires_grad


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.step_lrs = []

    def step(self, closure=None):
        self.step_lrs.append([group["lr"] for group in self.param_groups])
        return super().step(closure=closure)


class _StudentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.selected_logits = torch.nn.Parameter(torch.tensor([[[0.0, 0.0]]]))


def _update_batch(*, all_unavailable=False, non_finite=False):
    batch_size = 2
    response_length = 2
    top_k = 2
    num_experts = 2
    input_ids = torch.tensor([[1, 2, 11, 12], [3, 4, 21, 22]])
    specialized = torch.zeros(batch_size, response_length, top_k, num_experts, requires_grad=True)
    with torch.no_grad():
        specialized[0, :, 0, 0] = 2.0
        specialized[0, :, 0, 1] = 1.0
        if non_finite:
            specialized[0, 0, 0, 0] = torch.nan
    teacher_base = torch.zeros(batch_size, response_length, top_k, requires_grad=True)
    student_base = torch.zeros(batch_size, response_length, top_k, requires_grad=True)
    expert_mask = torch.tensor([[True, True], [False, False]])
    if all_unavailable:
        expert_mask.zero_()

    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(input_ids.shape[1]).expand(batch_size, -1).clone(),
            "responses": input_ids[:, -response_length:].clone(),
            "response_mask": torch.ones(batch_size, response_length),
            "student_topk_ids": torch.tensor(
                [
                    [[0, 1], [0, 1]],
                    [[0, 1], [0, 1]],
                ]
            ),
            "old_log_probs": torch.zeros(batch_size, response_length, top_k),
            "student_base_topk_log_probs": student_base,
            "teacher_base_topk_log_probs": teacher_base,
            "specialized_expert_topk_log_probs": specialized,
            "expert_mask": expert_mask,
        },
        meta_info={
            "temperature": 1.0,
            "mpopd_expert_temperature": 1.0,
            "mpopd_token_temperature": 1.0,
            "mpopd_lambda_value": 1.0,
            "mpopd_expert_priors": None,
        },
    )
    return batch, specialized, teacher_base, student_base


def _actor(monkeypatch):
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_id", lambda: "cpu")
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = OmegaConf.create(
        {
            "ppo_mini_batch_size": 2,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_max_token_len_per_gpu": 128,
            "ppo_epochs": 1,
            "use_dynamic_bsz": False,
            "mpopd_lr_scale": 0.5,
            "grad_clip": 1.0,
        }
    )
    actor.actor_module = _StudentModule()
    actor.actor_optimizer = _RecordingSGD(actor.actor_module.parameters(), lr=0.1)
    actor.scaler = None
    actor.ulysses_sequence_parallel_size = 1
    actor.forward_calls = 0

    def fake_forward(self, model_inputs, **kwargs):
        self.forward_calls += 1
        batch_size, response_length, top_k = model_inputs["student_topk_ids"].shape
        selected = self.actor_module.selected_logits.expand(batch_size, response_length, top_k)
        return None, None, selected

    actor._forward_micro_batch = MethodType(fake_forward, actor)
    return actor


def test_update_policy_mpopd_updates_only_student_and_reports_metrics(monkeypatch):
    actor = _actor(monkeypatch)
    batch, specialized, teacher_base, student_base = _update_batch()
    before = actor.actor_module.selected_logits.detach().clone()

    metrics = actor.update_policy_mpopd(batch)

    assert not torch.equal(actor.actor_module.selected_logits.detach(), before)
    assert actor.forward_calls == 2
    assert actor.actor_optimizer.step_lrs == [[0.05]]
    assert specialized.grad is None
    assert teacher_base.grad is None
    assert student_base.grad is None
    assert set(metrics) >= {
        "actor/mpopd_kl_loss",
        "actor/mpopd_target_entropy",
        "actor/mpopd_student_topk_entropy",
        "actor/mpopd_expert_weight_entropy",
        "actor/mpopd_max_expert_weight",
        "actor/mpopd_fused_delta_mean",
        "actor/mpopd_fused_delta_abs_mean",
        "actor/mpopd_conflict_rate",
        "actor/mpopd_all_experts_unavailable_count",
        "actor/mpopd_top_k",
        "actor/grad_norm",
    }
    assert metrics["actor/mpopd_all_experts_unavailable_count"] == [1.0]


def test_actor_rejects_non_finite_target(monkeypatch):
    actor = _actor(monkeypatch)
    batch, _, _, _ = _update_batch(non_finite=True)

    with pytest.raises(FloatingPointError, match="MP-OPD target"):
        actor.update_policy_mpopd(batch)

    assert actor.actor_optimizer.step_lrs == []


def test_all_unavailable_batch_skips_backward_and_optimizer(monkeypatch):
    actor = _actor(monkeypatch)
    batch, _, _, _ = _update_batch(all_unavailable=True)

    metrics = actor.update_policy_mpopd(batch)

    assert actor.forward_calls == 0
    assert actor.actor_optimizer.step_lrs == []
    assert actor.actor_module.selected_logits.grad is None
    assert metrics["actor/mpopd_all_experts_unavailable_count"] == [2.0]
    assert metrics["actor/grad_norm"] == [0.0]


class _FakeUpdateActor:
    def __init__(self):
        self.calls = 0

    def update_policy_mpopd(self, data):
        self.calls += 1
        return {"actor/mpopd_kl_loss": [0.25]}


def _worker_for_update():
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_offload_param = False
    worker._is_offload_optimizer = False
    worker._world_size = 1
    worker.ulysses_sharding_manager = nullcontext()
    worker.actor = _FakeUpdateActor()
    worker.base_policy = object()
    worker.ref_policy = object()
    worker.actor_lr_scheduler = None
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "temperature": 1.0,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 128,
                "log_prob_use_dynamic_bsz": False,
            },
            "actor": {
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_max_token_len_per_gpu": 128,
                "use_dynamic_bsz": False,
            },
        }
    )
    return worker


def test_update_actor_mpopd_calls_student_update_without_base_ref_policy():
    worker = _worker_for_update()
    worker.base_ref_policy = None
    batch, _, _, _ = _update_batch()

    output = worker.update_actor_mpopd(batch)

    assert worker.actor.calls == 1
    assert output.meta_info["metrics"]["actor/mpopd_kl_loss"] == 0.25


@pytest.mark.parametrize("missing", ["base_policy", "ref_policy"])
def test_update_actor_mpopd_requires_frozen_probability_sources(missing):
    worker = _worker_for_update()
    setattr(worker, missing, None)
    batch, _, _, _ = _update_batch()

    with pytest.raises(RuntimeError, match="MP-OPD"):
        worker.update_actor_mpopd(batch)
