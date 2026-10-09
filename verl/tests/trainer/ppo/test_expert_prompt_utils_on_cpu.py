import numpy as np
import torch

from verl.trainer.config.algorithm import MPOPDConfig
from verl.trainer.ppo.expert_prompt_utils import (
    build_expert_messages,
    normalize_expert_contexts,
    prepare_expert_prompt_inputs,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return " ".join(message["content"] for message in messages) + " <gen>"

    def __call__(self, text, add_special_tokens, return_tensors):
        return {"input_ids": torch.tensor([[len(text.split()) + 1, 2, 3]])}


def test_disabled_differs_from_missing_and_observed_empty_evidence():
    contexts, mask = normalize_expert_contexts(
        {
            "chasing": {"enabled": False},
            "long_term": {"enabled": True, "evidence_available": False},
            "repurchase": {"enabled": True, "evidence_available": True, "evidence": []},
        },
        ["chasing", "long_term", "repurchase"],
        {"chasing": "recent", "long_term": "stable", "repurchase": "repeat"},
    )
    assert mask.tolist() == [False, True, True]
    assert contexts[1].evidence_available is False
    assert contexts[2].evidence == []


def test_config_order_wins_over_dict_order():
    contexts, mask = normalize_expert_contexts(
        {"generalized": {"enabled": True, "instruction_override": "g2"}, "chasing": {"enabled": True}},
        ["chasing", "generalized"],
        {"chasing": "c", "generalized": "g"},
    )
    assert [item.instruction for item in contexts] == ["c", "g2"]
    assert mask.tolist() == [True, True]


def test_expert_messages_do_not_mutate_clean_prompt():
    clean = [{"role": "user", "content": "clean"}]
    messages = build_expert_messages(
        clean,
        "chasing",
        normalize_expert_contexts(None, ["chasing"], {"chasing": "recent"})[0][0],
    )
    assert clean == [{"role": "user", "content": "clean"}]
    assert "recent" in messages[-1]["content"]


def test_prepare_packs_only_active_experts():
    class Batch:
        pass

    batch = Batch()
    batch.batch = {
        "responses": torch.tensor([[7, 8], [9, 0]]),
        "response_mask": torch.tensor([[1, 1], [1, 0]]),
    }
    raw_prompts = np.empty(2, dtype=object)
    raw_prompts[:] = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    expert_contexts = np.empty(2, dtype=object)
    expert_contexts[:] = [{"chasing": {"enabled": True}}, {"chasing": {"enabled": False}}]
    batch.non_tensor_batch = {"raw_prompt": raw_prompts, "expert_contexts": expert_contexts}
    config = MPOPDConfig(expert_names=["chasing"], expert_instructions={"chasing": "recent"})
    packed = prepare_expert_prompt_inputs(batch, FakeTokenizer(), config)
    assert packed.expert_mask.tolist() == [[True], [False]]
    assert packed.flat_batch_indices.tolist() == [0]


def test_expert_messages_carry_no_duplicate_header_or_dangling_evidence():
    # The instruction already carries the expert's viewpoint header, so the builder
    # must not wrap it in a second ``【MP-OPD ... 专家视角】`` heading, and must not
    # emit the evidence heading when no evidence was observed.
    context = normalize_expert_contexts(None, ["chasing"], {"chasing": "focus recent items"})[0][0]
    messages = build_expert_messages([{"role": "user", "content": "clean"}], "chasing", context)
    content = messages[-1]["content"]
    assert "MP-OPD" not in content
    assert "证据" not in content
    assert "focus recent items" in content


def test_expert_messages_include_evidence_only_when_available():
    raw = {"chasing": {"enabled": True, "evidence_available": True, "evidence": {"k": "v"}}}
    context = normalize_expert_contexts(raw, ["chasing"], {"chasing": "focus recent"})[0][0]
    messages = build_expert_messages([{"role": "user", "content": "clean"}], "chasing", context)
    content = messages[-1]["content"]
    assert "证据" in content
    assert '"k": "v"' in content


def test_prepare_skips_experts_exceeding_max_total_length():
    # An expert whose prompt+response would exceed ``max_total_length`` (e.g. the
    # rollout ``max_model_len``) is dropped so the frozen forward never truncates the
    # response prefix. The same expert is kept when the bound is removed.
    class Batch:
        pass

    batch = Batch()
    batch.batch = {
        "responses": torch.tensor([[7, 8, 9, 10, 11]]),
        "response_mask": torch.ones(1, 5, dtype=torch.long),
    }
    raw_prompts = np.empty(1, dtype=object)
    raw_prompts[:] = [[{"role": "user", "content": "a b c"}]]
    expert_contexts = np.empty(1, dtype=object)
    expert_contexts[:] = [{"chasing": {"enabled": True}}]
    batch.non_tensor_batch = {"raw_prompt": raw_prompts, "expert_contexts": expert_contexts}
    config = MPOPDConfig(expert_names=["chasing"], expert_instructions={"chasing": "recent"})

    bounded = prepare_expert_prompt_inputs(batch, FakeTokenizer(), config, max_total_length=4)
    assert bounded.expert_mask.tolist() == [[False]]
    assert bounded.flat_batch_indices.tolist() == []

    unbounded = prepare_expert_prompt_inputs(batch, FakeTokenizer(), config, max_total_length=None)
    assert unbounded.expert_mask.tolist() == [[True]]
    assert unbounded.flat_batch_indices.tolist() == [0]
