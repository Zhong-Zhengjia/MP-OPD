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

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.expert_prompt_utils import ExpertPromptInputs
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    build_flat_expert_dataproto,
    restore_specialized_expert_log_probs,
)


def _packed(mask=None):
    if mask is None:
        mask = torch.tensor([[True, False, True], [False, True, False]])
    flat_indices = mask.nonzero(as_tuple=False)
    flat_count = flat_indices.shape[0]
    response_length = 2
    sequence_length = 5
    input_ids = torch.arange(flat_count * sequence_length, dtype=torch.long).reshape(flat_count, sequence_length) + 50
    responses = input_ids[:, -response_length:].clone()
    return ExpertPromptInputs(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(sequence_length).expand(flat_count, -1).clone(),
        responses=responses,
        response_mask=torch.ones(flat_count, response_length, dtype=torch.long),
        expert_mask=mask,
        flat_batch_indices=flat_indices[:, 0],
        flat_expert_indices=flat_indices[:, 1],
    )


def _source_batch(packed=None):
    batch_size = 2
    response_length = 2
    top_k = 2
    clean_input_ids = torch.tensor([[1, 2, 11, 12], [3, 4, 21, 22]])
    tensors = {
        "input_ids": clean_input_ids,
        "attention_mask": torch.ones_like(clean_input_ids),
        "position_ids": torch.arange(clean_input_ids.shape[1]).expand(batch_size, -1).clone(),
        "responses": clean_input_ids[:, -response_length:].clone(),
        "response_mask": torch.ones(batch_size, response_length, dtype=torch.long),
    }
    if packed is not None:
        num_experts = packed.expert_mask.shape[1]
        sequence_length = packed.input_ids.shape[1]
        dense_values = {
            "expert_input_ids": packed.input_ids.new_zeros((batch_size, num_experts, sequence_length)),
            "expert_attention_mask": packed.attention_mask.new_zeros((batch_size, num_experts, sequence_length)),
            "expert_position_ids": packed.position_ids.new_zeros((batch_size, num_experts, sequence_length)),
            "expert_responses": packed.responses.new_zeros((batch_size, num_experts, response_length)),
            "expert_response_mask": packed.response_mask.new_zeros((batch_size, num_experts, response_length)),
        }
        index = (packed.flat_batch_indices, packed.flat_expert_indices)
        dense_values["expert_input_ids"][index] = packed.input_ids
        dense_values["expert_attention_mask"][index] = packed.attention_mask
        dense_values["expert_position_ids"][index] = packed.position_ids
        dense_values["expert_responses"][index] = packed.responses
        dense_values["expert_response_mask"][index] = packed.response_mask
        tensors.update(dense_values)
        tensors["expert_mask"] = packed.expert_mask
    return DataProto.from_dict(
        tensors=tensors,
        meta_info={
            "top_k": top_k,
            "num_experts": 3,
            "expert_forward_micro_batch_size": 2,
        },
    )


def test_build_flat_expert_dataproto_selects_student_ids_and_carries_no_raw_context():
    packed = _packed()
    source = _source_batch()
    student_topk_ids = torch.arange(2 * 2 * 2).reshape(2, 2, 2)

    flat = build_flat_expert_dataproto(source, packed, student_topk_ids)

    assert flat.batch["input_ids"].shape[0] == 3
    assert flat.batch["student_topk_ids"].shape == (3, 2, 2)
    assert torch.equal(
        flat.batch["student_topk_ids"][1],
        student_topk_ids[packed.flat_batch_indices[1]],
    )
    assert set(flat.batch.keys()) == {
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "student_topk_ids",
    }
    assert not flat.non_tensor_batch


def test_restore_scatter_uses_reversible_batch_and_expert_indices():
    packed = _packed()
    flat_log_probs = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2)

    restored = restore_specialized_expert_log_probs(flat_log_probs, packed, batch_size=2, num_experts=3)

    assert restored.shape == (2, 2, 2, 3)
    for flat_idx, (batch_idx, expert_idx) in enumerate(
        zip(packed.flat_batch_indices, packed.flat_expert_indices, strict=True)
    ):
        assert torch.equal(restored[batch_idx, :, :, expert_idx], flat_log_probs[flat_idx])
    for batch_idx, expert_idx in (~packed.expert_mask).nonzero(as_tuple=False):
        assert torch.equal(
            restored[batch_idx, :, :, expert_idx],
            torch.zeros_like(restored[batch_idx, :, :, expert_idx]),
        )


def test_empty_flat_restore_has_expected_shape_and_zeros():
    packed = _packed(torch.zeros(2, 4, dtype=torch.bool))
    flat_log_probs = torch.empty(0, 2, 3)

    restored = restore_specialized_expert_log_probs(flat_log_probs, packed, batch_size=2, num_experts=4)

    assert restored.shape == (2, 2, 3, 4)
    assert torch.equal(restored, torch.zeros_like(restored))


class _FakeActor:
    def __init__(self, calls, *, invalid_ids=False):
        self.calls = calls
        self.invalid_ids = invalid_ids
        self.actor_module = object()

    def compute_topk_ids_and_log_probs(self, data, top_k):
        self.calls.append("actor")
        batch_size, response_length = data.batch["responses"].shape
        ids = torch.arange(top_k).view(1, 1, top_k).expand(batch_size, response_length, -1).clone()
        if self.invalid_ids:
            ids[0, 0, 0] = 100
        entropy = torch.zeros(batch_size, response_length)
        log_probs = torch.full((batch_size, response_length, top_k), -0.25)
        return entropy, ids, log_probs


class _FakeScorer:
    def __init__(self, calls, clean_name, expert_name=None):
        self.calls = calls
        self.clean_name = clean_name
        self.expert_name = expert_name
        self.actor_module = object()
        self.forward_calls = 0

    def compute_topk_log_probs_on_ids(self, data):
        self.forward_calls += 1
        is_expert = bool((data.batch["input_ids"][:, 0] >= 50).all())
        self.calls.append(self.expert_name if is_expert else self.clean_name)
        batch_size, response_length, top_k = data.batch["student_topk_ids"].shape
        base = data.batch["input_ids"][:, 0].float().view(batch_size, 1, 1)
        return base.expand(batch_size, response_length, top_k).clone()


def _worker_and_batch(*, empty_experts=False, invalid_ids=False):
    calls = []
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_ref = True
    worker._is_lora = False
    worker._is_offload_param = False
    worker.world_size = 1
    worker.ulysses_sharding_manager = nullcontext()
    worker.actor = _FakeActor(calls, invalid_ids=invalid_ids)
    worker.base_policy = _FakeScorer(calls, "student_base")
    worker.ref_policy = _FakeScorer(calls, "teacher_base", "specialized_experts")
    worker.ref_vocab_size = 100
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "log_prob_micro_batch_size_per_gpu": 2,
                "log_prob_max_token_len_per_gpu": 128,
                "log_prob_use_dynamic_bsz": False,
                "temperature": 1.0,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 2,
                "log_prob_max_token_len_per_gpu": 128,
                "log_prob_use_dynamic_bsz": False,
            },
        }
    )
    mask = torch.zeros(2, 3, dtype=torch.bool) if empty_experts else None
    packed = _packed(mask)
    return worker, _source_batch(packed), calls


def test_prepare_mpopd_runs_deterministic_shared_teacher_sequence_and_returns_exact_keys():
    worker, batch, calls = _worker_and_batch()

    output = worker.prepare_mpopd_log_probs(batch)

    assert calls == ["actor", "student_base", "teacher_base", "specialized_experts"]
    assert set(output.batch.keys()) == {
        "student_topk_ids",
        "old_log_probs",
        "student_base_topk_log_probs",
        "teacher_base_topk_log_probs",
        "specialized_expert_topk_log_probs",
        "expert_mask",
    }
    assert output.batch["specialized_expert_topk_log_probs"].shape == (2, 2, 2, 3)
    assert torch.equal(output.batch["expert_mask"], batch.batch["expert_mask"])


def test_prepare_mpopd_skips_specialized_forward_for_empty_expert_batch():
    worker, batch, calls = _worker_and_batch(empty_experts=True)

    output = worker.prepare_mpopd_log_probs(batch)

    assert calls == ["actor", "student_base", "teacher_base"]
    assert worker.ref_policy.forward_calls == 1
    assert torch.equal(
        output.batch["specialized_expert_topk_log_probs"],
        torch.zeros_like(output.batch["specialized_expert_topk_log_probs"]),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda worker, data: setattr(worker, "base_policy", None), "student base"),
        (lambda worker, data: setattr(worker, "ref_policy", None), "teacher base"),
        (lambda worker, data: data.meta_info.update(top_k=0), "top_k"),
        (lambda worker, data: data.meta_info.update(num_experts=99), "expert count"),
    ],
)
def test_prepare_mpopd_rejects_invalid_state(mutation, message):
    worker, batch, _ = _worker_and_batch()
    mutation(worker, batch)

    with pytest.raises((RuntimeError, ValueError), match=message):
        worker.prepare_mpopd_log_probs(batch)


def test_selected_id_must_exist_in_teacher_vocabulary():
    worker, batch, _ = _worker_and_batch(invalid_ids=True)

    with pytest.raises(ValueError, match="vocabulary"):
        worker.prepare_mpopd_log_probs(batch)
