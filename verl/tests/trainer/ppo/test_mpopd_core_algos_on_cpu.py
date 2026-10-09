import torch

from verl.trainer.ppo.core_algos import build_mpopd_target, compute_mpopd_kl_loss


def test_zero_delta_reduces_to_anchor_distribution():
    anchor = torch.tensor([[[0.0, -1.0, -2.0]]])
    result = build_mpopd_target(
        specialized_expert_log_probs=anchor[..., None].expand(-1, -1, -1, 2),
        teacher_base_log_probs=anchor,
        student_base_log_probs=anchor,
        expert_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    expected = torch.softmax(anchor, dim=-1)
    assert torch.allclose(result.target_probs, expected)
    assert torch.allclose(result.fused_delta, torch.zeros_like(result.fused_delta))


def test_positive_and_negative_delta_change_target_with_signed_fusion():
    base = torch.zeros(1, 1, 2)
    specialized = torch.zeros(1, 1, 2, 2)
    specialized[..., 0, 0] = 3.0
    specialized[..., 1, 1] = -3.0
    result = build_mpopd_target(
        specialized,
        torch.zeros_like(base),
        base,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert result.target_probs[0, 0, 0] > result.anchor_probs[0, 0, 0]
    assert result.target_probs[0, 0, 1] < result.anchor_probs[0, 0, 1]


def test_unavailable_experts_are_masked_and_all_unavailable_is_finite():
    specialized = torch.tensor([[[[2.0, 100.0]]], [[[float("nan"), 0.0]]]])
    teacher = torch.zeros(2, 1, 1)
    student = torch.zeros(2, 1, 1)
    result = build_mpopd_target(
        specialized,
        teacher,
        student,
        torch.tensor([[True, False], [False, False]]),
    )
    assert torch.allclose(result.expert_weights[0, 0, 0], torch.tensor([1.0, 0.0]))
    assert result.valid_samples.tolist() == [True, False]
    assert torch.isfinite(result.target_probs).all()


def test_kl_is_zero_when_student_matches_target_and_invalid_rows_have_no_gradient():
    student = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    target = torch.full((1, 2, 2), 0.5)
    loss = compute_mpopd_kl_loss(
        student,
        target,
        torch.ones(1, 2),
        torch.tensor([True]),
    )
    assert torch.allclose(loss, torch.tensor(0.0))

    invalid_student = torch.zeros(1, 1, 2, requires_grad=True)
    invalid_loss = compute_mpopd_kl_loss(
        invalid_student,
        torch.full((1, 1, 2), 0.5),
        torch.ones(1, 1),
        torch.tensor([False]),
    )
    invalid_loss.backward()
    assert torch.equal(invalid_student.grad, torch.zeros_like(invalid_student))


def test_target_is_detached():
    result = build_mpopd_target(
        torch.zeros(1, 1, 2, 1, requires_grad=True),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert not result.target_probs.requires_grad


def test_strict_mode_raises_when_any_sample_lacks_active_experts():
    # ``skip_samples_without_active_experts=False`` treats a sample with no active
    # expert as a configuration error rather than silently producing a zero-gradient
    # row, so a batch containing one should raise immediately.
    student = torch.zeros(1, 1, 2, requires_grad=True)
    target = torch.full((1, 1, 2), 0.5)
    try:
        compute_mpopd_kl_loss(
            student,
            target,
            torch.ones(1, 1),
            torch.tensor([False]),
            skip_samples_without_active_experts=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("strict mode should raise when a sample has no active experts")


def test_strict_mode_passes_when_all_samples_have_active_experts():
    student = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    target = torch.full((1, 1, 2), 0.5)
    loss = compute_mpopd_kl_loss(
        student,
        target,
        torch.ones(1, 1),
        torch.tensor([True]),
        skip_samples_without_active_experts=False,
    )
    assert torch.allclose(loss, torch.tensor(0.0))
