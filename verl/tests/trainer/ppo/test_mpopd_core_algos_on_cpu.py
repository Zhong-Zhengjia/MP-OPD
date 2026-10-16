import torch

from verl.trainer.ppo.core_algos import build_mpopd_target, compute_mpopd_kl_loss


def test_zero_delta_reduces_to_anchor_distribution():
    # When specialized == teacher, K-normalization makes delta exactly zero
    # (log_softmax(x) - log_softmax(x) = 0), so the target collapses to the
    # anchor distribution regardless of the normalization choice.
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


def test_k_normalized_delta_is_shift_invariant():
    # K-normalization renormalizes each expert's log-probs within the top-k
    # set via log_softmax. Two experts whose raw log-probs differ by a
    # constant across all K tokens produce identical delta, because
    # log_softmax cancels any additive constant. This is the property that
    # puts delta in the same ±log(K) scale as the anchor.
    base = torch.zeros(1, 1, 2)
    specialized = torch.zeros(1, 1, 2, 2)
    specialized[..., 0, 0] = 3.0  # expert 0: [3, 0]
    specialized[..., 0, 1] = 0.0
    specialized[..., 1, 0] = 5.0  # expert 1: [5, 2] — same shape, shifted +2
    specialized[..., 1, 1] = 2.0
    result = build_mpopd_target(
        specialized,
        torch.zeros_like(base),
        base,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert torch.allclose(result.expert_delta[..., 0], result.expert_delta[..., 1])


def test_positive_and_negative_delta_change_target_with_signed_fusion():
    # Experts with different K-spreads (3 vs 1) produce genuinely different
    # K-normalized deltas, so the signed fusion weights them differently.
    # Both still push token 0 up and token 1 down — the direction is preserved
    # by K-normalization, but the magnitude reflects the within-K spread.
    base = torch.zeros(1, 1, 2)
    specialized = torch.zeros(1, 1, 2, 2)
    specialized[..., 0, 0] = 3.0  # expert 0: spread 3 → strong delta
    specialized[..., 1, 1] = -1.0  # expert 1: spread 1 → weaker delta
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


def test_reverse_kl_is_zero_when_student_matches_target():
    # Reverse KL(student || target): when student and target are identical
    # (both uniform [0.5, 0.5]), KL = Σ s_k * (log s_k - log t_k) = 0.
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


def test_reverse_kl_pushes_student_toward_target_mode():
    # Reverse KL is mode-seeking: when the target concentrates on token 0
    # (0.9 vs 0.1) and the student is uniform, the gradient on the
    # high-probability token should be negative (gradient descent increases
    # its logit, raising its probability toward the target).
    student = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
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
