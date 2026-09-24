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

import copy

import numpy as np
import torch

from verl import DataProto
from verl.trainer.config.algorithm import MPOPDConfig
from verl.trainer.ppo.expert_prompt_utils import (
    EVIDENCE_MARKER,
    FOCUS_MARKER,
    attach_expert_prompt_tensors,
    build_expert_messages,
    normalize_expert_contexts,
    prepare_expert_prompt_inputs,
)


class FakeTokenizer:
    pad_token_id = 0
    chat_template = "fake"

    def __init__(self):
        self.vocab = {FOCUS_MARKER: 901, EVIDENCE_MARKER: 902}
        self._next_id = 1000

    def _tokenize(self, text):
        ids = []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = self._next_id
                self._next_id += 1
            ids.append(self.vocab[token])
        return ids

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=False, **kwargs):
        del kwargs
        text = " ".join(f"{item['role']} {item['content']}" for item in messages)
        if add_generation_prompt:
            text += " assistant"
        if tokenize:
            return self._tokenize(text)
        return text

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        ids = torch.tensor([self._tokenize(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return self._tokenize(text)


def _object_array(items):
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def _batch(contexts, *, prompts=None, responses=None):
    batch_size = len(contexts)
    if prompts is None:
        prompts = [[{"role": "user", "content": f"clean request {idx}"}] for idx in range(batch_size)]
    if responses is None:
        responses = torch.tensor([[71, 72, 0], [81, 0, 0]], dtype=torch.long)[:batch_size]
    response_mask = responses.ne(0).long()
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[11, 12, 71, 72, 0], [21, 22, 81, 0, 0]], dtype=torch.long)[:batch_size],
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.long)[:batch_size],
            "position_ids": torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 0, 0]], dtype=torch.long)[:batch_size],
            "responses": responses,
            "response_mask": response_mask,
        },
        non_tensors={
            "raw_prompt": _object_array(prompts),
            "expert_contexts": _object_array(contexts),
        },
    )


def test_disabled_differs_from_missing_and_observed_empty_evidence():
    raw = {
        "chasing": {"enabled": False},
        "long_term": {"enabled": True, "evidence_available": False},
        "repurchase": {"enabled": True, "evidence_available": True, "evidence": []},
    }

    contexts, mask = normalize_expert_contexts(
        raw,
        ["chasing", "long_term", "repurchase"],
        {"chasing": "focus recent", "long_term": "focus stable", "repurchase": "focus repeat"},
    )

    assert mask.tolist() == [False, True, True]
    assert contexts[1].evidence_available is False
    assert contexts[2].evidence == []


def test_config_order_wins_over_dict_order():
    raw = {
        "generalized": {"enabled": True, "instruction_override": "g2"},
        "chasing": {"enabled": True},
    }

    contexts, mask = normalize_expert_contexts(
        raw, ["chasing", "generalized"], {"chasing": "c", "generalized": "g"}
    )

    assert [item.instruction for item in contexts] == ["c", "g2"]
    assert mask.tolist() == [True, True]


def test_absent_sample_context_keeps_all_configured_experts_active():
    contexts, mask = normalize_expert_contexts(
        None, ["chasing", "long_term"], {"chasing": "c", "long_term": "l"}
    )

    assert mask.tolist() == [True, True]
    assert all(not item.evidence_available for item in contexts)


def test_build_messages_copies_clean_prompt_and_only_adds_available_evidence():
    raw_prompt = [{"role": "user", "content": "clean request"}]
    original = copy.deepcopy(raw_prompt)
    contexts, _ = normalize_expert_contexts(
        {"chasing": {"evidence_available": False, "evidence": "must not leak"}},
        ["chasing"],
        {"chasing": "focus recent"},
    )

    messages = build_expert_messages(raw_prompt, "chasing", contexts[0])

    assert raw_prompt == original
    assert messages is not raw_prompt
    assert FOCUS_MARKER in messages[-1]["content"]
    assert EVIDENCE_MARKER not in messages[-1]["content"]
    assert "must not leak" not in messages[-1]["content"]


def test_prepare_flattens_valid_experts_in_sample_and_config_order_without_clean_prompt_mutation():
    tokenizer = FakeTokenizer()
    cfg = MPOPDConfig(
        expert_names=["chasing", "long_term", "repurchase"],
        expert_instructions={"chasing": "focus c", "long_term": "focus l", "repurchase": "focus r"},
        max_expert_prompt_length=64,
    )
    batch = _batch(
        [
            {
                "chasing": {"enabled": True},
                "long_term": {"enabled": False},
                "repurchase": {"enabled": True, "evidence_available": True, "evidence": []},
            },
            {
                "chasing": {"enabled": False},
                "long_term": {"enabled": True},
                "repurchase": {"enabled": False},
            },
        ]
    )
    clean_ids = batch.batch["input_ids"].clone()

    packed = prepare_expert_prompt_inputs(batch, tokenizer, cfg)

    assert packed.input_ids.shape[0] == int(packed.expert_mask.sum())
    assert packed.flat_batch_indices.tolist() == [0, 0, 1]
    assert packed.flat_expert_indices.tolist() == [0, 2, 1]
    assert torch.equal(batch.batch["input_ids"], clean_ids)
    assert tokenizer.vocab[FOCUS_MARKER] not in batch.batch["input_ids"]
    assert tokenizer.vocab[EVIDENCE_MARKER] not in batch.batch["input_ids"]
    assert torch.equal(packed.responses, batch.batch["responses"][packed.flat_batch_indices])
    assert torch.equal(packed.response_mask, batch.batch["response_mask"][packed.flat_batch_indices])
    assert torch.equal(packed.input_ids[:, -packed.responses.shape[1] :], packed.responses)
    assert torch.equal(packed.attention_mask[:, -packed.responses.shape[1] :], packed.response_mask)


def test_missing_context_builds_all_experts_but_keeps_all_clean_views_identical():
    tokenizer = FakeTokenizer()
    cfg = MPOPDConfig(max_expert_prompt_length=64)
    batch = _batch([None])
    student_clean_ids = batch.batch["input_ids"].clone()
    student_base_clean_ids = batch.batch["input_ids"].clone()
    teacher_base_clean_ids = batch.batch["input_ids"].clone()

    packed = prepare_expert_prompt_inputs(batch, tokenizer, cfg)

    assert packed.expert_mask.tolist() == [[True, True, True, True]]
    assert packed.input_ids.shape[0] == 4
    assert torch.equal(student_clean_ids, student_base_clean_ids)
    assert torch.equal(student_clean_ids, teacher_base_clean_ids)
    for marker in (FOCUS_MARKER, EVIDENCE_MARKER):
        assert tokenizer.vocab[marker] not in student_clean_ids


def test_overflow_masks_only_the_unbuildable_expert_and_preserves_responses():
    tokenizer = FakeTokenizer()
    cfg = MPOPDConfig(
        expert_names=["short", "long"],
        expert_instructions={"short": "brief", "long": "brief"},
        max_expert_prompt_length=12,
    )
    batch = _batch(
        [
            {
                "short": {"enabled": True},
                "long": {"enabled": True, "instruction_override": "overflow " * 20},
            }
        ]
    )

    packed = prepare_expert_prompt_inputs(batch, tokenizer, cfg)

    assert packed.expert_mask.tolist() == [[True, False]]
    assert packed.flat_batch_indices.tolist() == [0]
    assert torch.equal(packed.responses, batch.batch["responses"][[0]])


def test_all_overflow_returns_empty_flat_batch_and_false_mask():
    tokenizer = FakeTokenizer()
    cfg = MPOPDConfig(
        expert_names=["a", "b"],
        expert_instructions={"a": "overflow " * 20, "b": "overflow " * 20},
        max_expert_prompt_length=5,
    )
    batch = _batch([None])

    packed = prepare_expert_prompt_inputs(batch, tokenizer, cfg)

    assert packed.input_ids.shape[0] == 0
    assert packed.responses.shape == (0, batch.batch["responses"].shape[1])
    assert not packed.expert_mask.any()


def test_attach_uses_dense_transport_without_raw_expert_contexts_or_source_mutation():
    tokenizer = FakeTokenizer()
    cfg = MPOPDConfig(
        expert_names=["a", "b"],
        expert_instructions={"a": "focus a", "b": "focus b"},
        max_expert_prompt_length=64,
    )
    batch = _batch([{"a": {"enabled": True}, "b": {"enabled": False}}])
    packed = prepare_expert_prompt_inputs(batch, tokenizer, cfg)

    attached = attach_expert_prompt_tensors(batch, packed)

    assert "expert_contexts" in batch.non_tensor_batch
    assert "expert_contexts" not in attached.non_tensor_batch
    assert attached.batch["expert_input_ids"].shape[:2] == (1, 2)
    assert attached.batch["expert_mask"].tolist() == [[True, False]]
    assert torch.equal(
        attached.batch["expert_input_ids"][0, 0],
        packed.input_ids[0],
    )
