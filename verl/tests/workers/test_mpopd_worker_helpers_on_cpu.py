import torch

from verl import DataProto
from verl.trainer.ppo.expert_prompt_utils import ExpertPromptInputs
from verl.workers.fsdp_workers import (
    build_flat_expert_dataproto,
    restore_specialized_expert_log_probs,
)


def _packed():
    return ExpertPromptInputs(
        input_ids=torch.tensor([[11, 12], [21, 22], [31, 32]]),
        attention_mask=torch.ones(3, 2, dtype=torch.long),
        position_ids=torch.arange(2).repeat(3, 1),
        responses=torch.tensor([[5, 6], [7, 8], [9, 0]]),
        response_mask=torch.tensor([[1, 1], [1, 1], [1, 0]]),
        expert_mask=torch.tensor([[True, False, True, False], [False, True, False, False]]),
        flat_batch_indices=torch.tensor([0, 0, 1]),
        flat_expert_indices=torch.tensor([0, 2, 1]),
    )


def test_flat_expert_batch_repeats_student_topk_ids_without_raw_context():
    source = DataProto.from_dict(
        tensors={
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "position_ids": torch.arange(3).repeat(2, 1),
            "responses": torch.tensor([[5, 6], [7, 8]]),
        },
        non_tensors={"expert_contexts": [{"x": 1}, {"y": 2}]},
    )
    student_topk_ids = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )
    flat = build_flat_expert_dataproto(source, _packed(), student_topk_ids)

    assert flat.batch["input_ids"].shape[0] == 3
    assert torch.equal(flat.batch["student_topk_ids"][1], student_topk_ids[0])
    assert "expert_contexts" not in flat.non_tensor_batch


def test_restore_specialized_log_probs_scatter_preserves_expert_axis():
    packed = _packed()
    flat = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ]
    )
    restored = restore_specialized_expert_log_probs(flat, packed, batch_size=2, num_experts=4)

    assert restored.shape == (2, 1, 2, 4)
    assert torch.equal(restored[0, 0, :, 0], flat[0, 0])
    assert torch.equal(restored[0, 0, :, 2], flat[1, 0])
    assert torch.equal(restored[1, 0, :, 1], flat[2, 0])
    assert torch.equal(restored[..., 3], torch.zeros(2, 1, 2))


def test_restore_empty_expert_batch_returns_zero_tensor():
    packed = _packed()
    empty = torch.empty(0, 2, 3)
    packed = ExpertPromptInputs(
        input_ids=packed.input_ids[:0],
        attention_mask=packed.attention_mask[:0],
        position_ids=packed.position_ids[:0],
        responses=packed.responses[:0],
        response_mask=packed.response_mask[:0],
        expert_mask=torch.zeros(2, 4, dtype=torch.bool),
        flat_batch_indices=torch.empty(0, dtype=torch.long),
        flat_expert_indices=torch.empty(0, dtype=torch.long),
    )
    restored = restore_specialized_expert_log_probs(empty, packed, batch_size=2, num_experts=4)

    assert restored.shape == (2, 2, 3, 4)
    assert torch.equal(restored, torch.zeros_like(restored))
