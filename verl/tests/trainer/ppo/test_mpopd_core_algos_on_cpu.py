"""Unit tests for the method_new MP-OPD target/loss (K+1 out bucket, full-vocab delta).

These tests run on CPU without torch distributed. Inputs are constructed as
valid full-vocab ``log_softmax`` values: each "vocab" is the K candidates plus a
synthetic out token, so the anchor distribution ``a`` genuinely sums to 1 over
the K+1 set and ``log1mexp`` recovers a real complement mass.
"""

import math

import torch

from verl.trainer.ppo.core_algos import (
    build_mpopd_target,
    compute_mpopd_kl_loss,
    compute_mpopd_token_kl,
)
from verl.utils.torch_functional import log1mexp


def _anchor_from_logits(logits: torch.Tensor):
    """Build a valid K+1 anchor from K logits.

    Treats the K logits as the top-k slice of a (K+1)-word vocab whose out logit
    is 0, so ``log_softmax`` over K+1 yields a proper distribution. Returns
    ``(log_a_k, log_a_out)`` where ``exp(log_a_k).sum + exp(log_a_out) == 1``.
    """
    out_logit = torch.zeros(*logits.shape[:2], 1)
    full = torch.cat([logits, out_logit], dim=-1)
    log_probs = torch.log_softmax(full, dim=-1)
    k = logits.shape[-1]
    return log_probs[..., :k], log_probs[..., k]


def _log_softmax_kplus1(logits: torch.Tensor):
    """Full-vocab log_softmax over a (K+1) vocab, returning only the K top-k slices.

    ``logits`` is [B,T,K]; the out logit is fixed at 0 so the returned K values
    are a valid full-vocab ``log_softmax`` (their exp sums to <= 1).
    """
    out_logit = torch.zeros(*logits.shape[:2], 1)
    full = torch.cat([logits, out_logit], dim=-1)
    return torch.log_softmax(full, dim=-1)[..., : logits.shape[-1]]


def _expert_log_softmax(specialized_logits: torch.Tensor):
    """Per-expert full-vocab log_softmax over a (K+1) vocab, K slices only.

    ``specialized_logits`` is [B,T,K,E]; the out logit is 0 per expert, so each
    expert's K values are a valid full-vocab ``log_softmax``.
    """
    # Append a [B,T,1,E] out-logit slice (zeros) along the K axis to get
    # [B,T,K+1,E], log_softmax over dim=2, then take the first K slices.
    out = torch.zeros(*specialized_logits.shape[:2], 1, specialized_logits.shape[3])
    full = torch.cat([specialized_logits, out], dim=2)
    return torch.log_softmax(full, dim=2)[..., : specialized_logits.shape[2], :]


# --------------------------------------------------------------------------
# build_mpopd_target
# --------------------------------------------------------------------------


def test_g_zero_target_equals_anchor_over_kplus1():
    # method_new §8: when the fused expert delta is zero (specialized == teacher),
    # the target collapses to the anchor distribution a on the K+1 set.
    anchor_logits = torch.tensor([[[1.0, -1.0, 0.5]]])
    log_a_k, log_a_out = _anchor_from_logits(anchor_logits)
    teacher = _log_softmax_kplus1(anchor_logits)  # [B,T,K]
    specialized = teacher.unsqueeze(-1).expand(-1, -1, -1, 2)  # == teacher → delta 0
    result = build_mpopd_target(
        specialized_expert_log_probs=specialized,
        teacher_base_log_probs=teacher,
        student_base_log_probs=log_a_k,
        student_base_out_log_probs=log_a_out,
        expert_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    expected = torch.exp(torch.cat([log_a_k, log_a_out.unsqueeze(-1)], dim=-1))
    assert torch.allclose(result.target_probs, expected, atol=1e-6)
    assert torch.allclose(result.fused_delta, torch.zeros_like(result.fused_delta), atol=1e-6)


def test_target_and_anchor_sum_to_one_over_kplus1():
    anchor_logits = torch.randn(2, 3, 4)
    log_a_k, log_a_out = _anchor_from_logits(anchor_logits)
    teacher = _log_softmax_kplus1(anchor_logits)
    specialized = _expert_log_softmax(teacher.unsqueeze(-1).expand(-1, -1, -1, 2) + 0.3)
    result = build_mpopd_target(
        specialized_expert_log_probs=specialized,
        teacher_base_log_probs=teacher,
        student_base_log_probs=log_a_k,
        student_base_out_log_probs=log_a_out,
        expert_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    assert torch.allclose(result.target_probs.sum(-1), torch.ones(2, 3), atol=1e-5)
    assert torch.allclose(result.anchor_probs.sum(-1), torch.ones(2, 3), atol=1e-5)


def test_opposite_deltas_cancel_to_anchor():
    # Two experts with exactly opposite deltas (+d, -d) get equal softmax weights
    # (softmax of equal |delta|), so the signed fusion cancels and the target
    # returns to the anchor (method_new §8: g=0 -> q=a).
    # NB: a constant logit shift does NOT yield opposite deltas here, because
    # the out logit is fixed at 0 -- shifting all K logits by +c moves mass into
    # K from the out bucket, while -c moves it out, and those are not negatives
    # of each other. We instead set specialized = teacher +/- d directly so the
    # per-expert deltas are exactly d and -d.
    anchor_logits = torch.tensor([[[1.0, -1.0, 0.5, 0.2]]])
    log_a_k, log_a_out = _anchor_from_logits(anchor_logits)
    teacher = _log_softmax_kplus1(anchor_logits)
    d = 0.1
    # [B,T,K,E]: expert 0 = teacher + d, expert 1 = teacher - d (small d keeps
    # exp(specialized).sum over K < 1, i.e. the out mass stays non-negative).
    specialized = torch.stack([teacher + d, teacher - d], dim=-1)
    result = build_mpopd_target(
        specialized,
        teacher,
        log_a_k,
        log_a_out,
        torch.ones(1, 2, dtype=torch.bool),
    )
    expected = torch.exp(torch.cat([log_a_k, log_a_out.unsqueeze(-1)], dim=-1))
    assert torch.allclose(result.fused_delta, torch.zeros_like(result.fused_delta), atol=1e-6)
    assert torch.allclose(result.target_probs, expected, atol=1e-6)


def test_positive_delta_raises_target_token():
    # A positive fused delta on token 0 should raise its target probability
    # above its anchor probability (method_new §8).
    anchor_logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    log_a_k, log_a_out = _anchor_from_logits(anchor_logits)
    teacher = _log_softmax_kplus1(anchor_logits)
    specialized = teacher.unsqueeze(-1).expand(-1, -1, -1, 2).clone()
    # both experts shift mass onto token 0
    lift = torch.tensor([3.0, 0.0, 0.0, 0.0])
    specialized[..., 0] = _log_softmax_kplus1(anchor_logits + lift)
    specialized[..., 1] = _log_softmax_kplus1(anchor_logits + lift)
    result = build_mpopd_target(
        specialized,
        teacher,
        log_a_k,
        log_a_out,
        torch.ones(1, 2, dtype=torch.bool),
        lambda_value=0.05,  # amplify the full-vocab-scale delta
    )
    assert result.target_probs[0, 0, 0] > result.anchor_probs[0, 0, 0]


def test_expert_weights_sum_to_one_per_candidate():
    anchor_logits = torch.randn(1, 2, 3)
    log_a_k, log_a_out = _anchor_from_logits(anchor_logits)
    teacher = _log_softmax_kplus1(anchor_logits)
    specialized = torch.stack(
        [_log_softmax_kplus1(anchor_logits + 1.0), _log_softmax_kplus1(anchor_logits - 1.0)], dim=-1
    )
    result = build_mpopd_target(
        specialized, teacher, log_a_k, log_a_out, torch.ones(1, 2, dtype=torch.bool)
    )
    assert torch.allclose(result.expert_weights.sum(-1), torch.ones(1, 2, 3), atol=1e-6)


def test_unavailable_experts_masked_and_all_unavailable_is_finite():
    specialized = torch.tensor([[[[2.0, 100.0]]], [[[float("nan"), 0.0]]]])
    teacher = _log_softmax_kplus1(torch.zeros(2, 1, 1))
    log_a_k, log_a_out = _anchor_from_logits(torch.zeros(2, 1, 1))
    result = build_mpopd_target(
        specialized, teacher, log_a_k, log_a_out,
        torch.tensor([[True, False], [False, False]]),
    )
    assert torch.allclose(result.expert_weights[0, 0, 0], torch.tensor([1.0, 0.0]))
    assert result.valid_samples.tolist() == [True, False]
    assert torch.isfinite(result.target_probs).all()


def test_target_is_detached():
    anchor_logits = torch.tensor([[[1.0, -1.0]]])
    log_a_k, log_a_out = _anchor_from_logits(anchor_logits)
    teacher = _log_softmax_kplus1(anchor_logits)
    result = build_mpopd_target(
        torch.zeros(1, 1, 2, 1, requires_grad=True),
        teacher,
        log_a_k,
        log_a_out,
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert not result.target_probs.requires_grad
    assert not result.target_log_probs.requires_grad
    assert not result.expert_weights.requires_grad


def test_delta_normalize_full_vs_k():
    # Both modes produce a valid K+1 target. The function does the
    # K-renormalization internally for mode='k', so both modes receive the same
    # raw full-vocab inputs. A per-token perturbation (not a constant shift) is
    # required: a constant shift cancels in the K-softmax so the k-mode delta
    # would be zero, defeating the "nonzero delta" check.
    anchor_logits = torch.tensor([[[1.0, -1.0, 0.5]]])  # K=3
    log_a_k, log_a_out = _anchor_from_logits(anchor_logits)
    teacher = _log_softmax_kplus1(anchor_logits)
    # per-token shifts that differ across tokens AND across experts
    shift_e0 = torch.tensor([[[0.5, -0.2, 0.1]]])
    shift_e1 = torch.tensor([[[0.1, 0.3, -0.4]]])
    spec_logits = torch.stack(
        [anchor_logits + shift_e0, anchor_logits + shift_e1], dim=-1
    )  # [1,1,3,2]
    specialized = _expert_log_softmax(spec_logits)  # log_softmax over K+1, [1,1,3,2]
    for mode in ("full", "k"):
        result = build_mpopd_target(
            specialized_expert_log_probs=specialized,
            teacher_base_log_probs=teacher,
            student_base_log_probs=log_a_k,
            student_base_out_log_probs=log_a_out,
            expert_mask=torch.ones(1, 2, dtype=torch.bool),
            delta_normalize=mode,
        )
        assert torch.isfinite(result.target_probs).all()
        assert torch.allclose(result.target_probs.sum(-1), torch.ones(1, 1), atol=1e-5)
        # expert != teacher (per-token perturbation) -> fused delta is nonzero
        assert result.fused_delta.abs().sum() > 0
    # In 'k' mode the delta lives in ±log(K) (renormalized within top-k); in
    # 'full' mode it is the raw full-vocab difference. The two magnitudes differ.
    full_res = build_mpopd_target(
        specialized, teacher, log_a_k, log_a_out, torch.ones(1, 2, dtype=torch.bool), delta_normalize="full"
    )
    k_res = build_mpopd_target(
        specialized, teacher, log_a_k, log_a_out, torch.ones(1, 2, dtype=torch.bool), delta_normalize="k"
    )
    assert k_res.expert_delta.abs().mean() != full_res.expert_delta.abs().mean()


# --------------------------------------------------------------------------
# compute_mpopd_kl_loss / compute_mpopd_token_kl
# --------------------------------------------------------------------------


def _uniform_student(B=1, T=1, K=2):
    # A uniform full-vocab log_softmax over a (K+1) vocab: top-k log probs are
    # log(1/(K+1)) and the out mass is the same, so p is uniform over K+1.
    log_p = torch.full((B, T, K), -math.log(K + 1))
    log_p_out = torch.full((B, T), -math.log(K + 1))
    return log_p, log_p_out


def test_reverse_kl_is_zero_when_student_matches_target():
    log_p_k, log_p_out = _uniform_student(K=2)
    log_p_k.requires_grad_(True)
    log_p_out.requires_grad_(True)
    target = torch.full((1, 1, 3), 1 / 3)  # uniform over K+1
    loss = compute_mpopd_kl_loss(log_p_k, log_p_out, target, torch.ones(1, 1), torch.tensor([True]))
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_student_out_term_has_gradient():
    # method_new §8: the student's out-bucket term must carry gradient.
    log_p_k, log_p_out = _uniform_student(K=3)
    log_p_k.requires_grad_(True)
    log_p_out.requires_grad_(True)
    # target concentrated on token 0
    target = torch.tensor([[[0.7, 0.1, 0.1, 0.1]]])
    loss = compute_mpopd_kl_loss(log_p_k, log_p_out, target, torch.ones(1, 1), torch.tensor([True]))
    loss.backward()
    assert log_p_out.grad is not None
    assert log_p_out.grad.abs().sum() > 0


def test_reverse_kl_pushes_student_toward_target_mode():
    # Reverse KL is mode-seeking: at a uniform student, a target concentrated on
    # token 0 should have a negative gradient on log p_0 (gradient descent raises
    # token 0). The unconstrained gradient is p_0*(1 + log(p_0/q_0)); this is
    # negative iff p_0/q_0 < 1/e, so the target must be concentrated enough that
    # the mode token's ratio clears that threshold (q_0=0.9 with p_0=1/3 sits
    # just above 1/e and gives a near-zero positive gradient -- use 0.99).
    log_p_k, log_p_out = _uniform_student(K=2)
    log_p_k.requires_grad_(True)
    target = torch.tensor([[[0.99, 0.005, 0.005]]])
    loss = compute_mpopd_kl_loss(log_p_k, log_p_out, target, torch.ones(1, 1), torch.tensor([True]))
    loss.backward()
    assert log_p_k.grad is not None
    # gradient descent raises token 0's logit → its grad should be negative
    assert log_p_k.grad[0, 0, 0] < 0


def test_invalid_sample_has_zero_gradient():
    log_p_k, log_p_out = _uniform_student(K=2)
    log_p_k.requires_grad_(True)
    target = torch.full((1, 1, 3), 1 / 3)
    loss = compute_mpopd_kl_loss(log_p_k, log_p_out, target, torch.ones(1, 1), torch.tensor([False]))
    loss.backward()
    assert torch.equal(log_p_k.grad, torch.zeros_like(log_p_k.grad))


def test_strict_mode_raises_when_any_sample_lacks_active_experts():
    log_p_k, log_p_out = _uniform_student(K=2)
    target = torch.full((1, 1, 3), 1 / 3)
    try:
        compute_mpopd_kl_loss(
            log_p_k, log_p_out, target, torch.ones(1, 1), torch.tensor([False]),
            skip_samples_without_active_experts=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("strict mode should raise when a sample has no active experts")


def test_token_kl_shape_and_padding():
    # token_kl is per-position [B,T]; padding positions must not affect the
    # reduced loss (mask zeroes them out).
    log_p_k, log_p_out = _uniform_student(B=1, T=2, K=2)
    log_p_k.requires_grad_(True)
    target = torch.full((1, 2, 3), 1 / 3)
    token_kl = compute_mpopd_token_kl(log_p_k, log_p_out, target)
    assert token_kl.shape == (1, 2)
    mask = torch.tensor([[1.0, 0.0]])
    loss = compute_mpopd_kl_loss(log_p_k, log_p_out, target, mask, torch.tensor([True]))
    # only the first position contributes and it matches the target → 0
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


# --------------------------------------------------------------------------
# log1mexp
# --------------------------------------------------------------------------


def test_log1mexp_matches_naive_and_is_stable():
    x = torch.tensor([-1e-8, -0.1, -0.5, -0.6931, -1.0, -3.0, -20.0])
    out = log1mexp(x)
    naive = torch.log(1.0 - torch.exp(x))
    # match the naive formula where it is numerically safe
    assert torch.allclose(out[1:], naive[1:], atol=1e-5)
    # extreme: x -> 0. The naive log(1 - exp(x)) suffers catastrophic cancellation
    # (exp(-1e-30) rounds to 1.0 -> log(0) -> -inf). The stable log1mexp uses
    # log(-expm1(x)) = log(1e-30) = -69.08, which is the TRUE (finite) value.
    assert torch.isfinite(log1mexp(torch.tensor([-1e-30])))
    assert torch.allclose(log1mexp(torch.tensor([-1e-30])), torch.tensor(-69.0776), atol=1e-3)
    # x -> -inf: 1 - exp(x) -> 1, log -> 0
    assert torch.allclose(log1mexp(torch.tensor([-50.0])), torch.tensor(0.0), atol=1e-6)


def test_log1mexp_rejects_positive_input():
    try:
        log1mexp(torch.tensor([0.1]))
    except ValueError:
        pass
    else:
        raise AssertionError("log1mexp should reject x > 0")
