"""Actor-level MP-OPD loss tests on CPU (method_new K+1 API)."""

import math

import torch

from verl.trainer.ppo.core_algos import build_mpopd_target, compute_mpopd_kl_loss


def _anchor_from_logits(logits):
    out_logit = torch.zeros(*logits.shape[:2], 1)
    full = torch.cat([logits, out_logit], dim=-1)
    log_probs = torch.log_softmax(full, dim=-1)
    k = logits.shape[-1]
    return log_probs[..., :k], log_probs[..., k]


def _uniform_student(B=1, T=1, K=2):
    log_p = torch.full((B, T, K), -math.log(K + 1))
    log_p_out = torch.full((B, T), -math.log(K + 1))
    return log_p, log_p_out


def test_only_student_log_probs_receive_gradient():
    log_p_k, log_p_out = _uniform_student(K=2)
    log_p_k.requires_grad_(True)
    log_p_out.requires_grad_(True)
    # Reverse KL is mode-seeking: at a uniform student the unconstrained
    # gradient is p_0*(1 + log(p_0/q_0)), negative iff p_0/q_0 < 1/e. The target
    # must be concentrated enough to clear that threshold (q_0=0.9 with p_0=1/3
    # sits just above 1/e and gives a near-zero positive gradient -- use 0.99).
    target = torch.tensor([[[0.99, 0.005, 0.005]]])
    loss = compute_mpopd_kl_loss(log_p_k, log_p_out, target, torch.ones(1, 1), torch.tensor([True]))
    loss.backward()

    assert log_p_k.grad is not None
    assert log_p_k.grad[0, 0, 0] < 0
    assert log_p_out.grad is not None
    assert log_p_out.grad.abs().sum() > 0


def test_invalid_sample_has_zero_gradient():
    log_p_k, log_p_out = _uniform_student(K=2)
    log_p_k.requires_grad_(True)
    loss = compute_mpopd_kl_loss(
        log_p_k,
        log_p_out,
        torch.full((1, 1, 3), 1 / 3),
        torch.ones(1, 1),
        torch.tensor([False]),
    )
    loss.backward()
    assert torch.equal(log_p_k.grad, torch.zeros_like(log_p_k.grad))


def test_mpopd_target_is_detached_from_all_input_tensors():
    anchor_logits = torch.tensor([[[1.0, -1.0]]])
    log_a_k, log_a_out = _anchor_from_logits(anchor_logits)
    target = build_mpopd_target(
        torch.zeros(1, 1, 2, 2, requires_grad=True),
        torch.zeros(1, 1, 2, requires_grad=True),
        log_a_k,
        log_a_out,
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert not target.target_probs.requires_grad
    assert not target.target_log_probs.requires_grad
    assert not target.expert_weights.requires_grad


def test_all_unavailable_target_is_finite_and_loss_is_zero():
    anchor_logits = torch.zeros(1, 2, 3)
    log_a_k, log_a_out = _anchor_from_logits(anchor_logits)
    target = build_mpopd_target(
        torch.zeros(1, 2, 3, 2),
        torch.zeros(1, 2, 3),
        log_a_k,
        log_a_out,
        torch.zeros(1, 2, dtype=torch.bool),
    )
    log_p_k, log_p_out = _uniform_student(B=1, T=2, K=3)
    log_p_k.requires_grad_(True)
    loss = compute_mpopd_kl_loss(log_p_k, log_p_out, target.target_probs, torch.ones(1, 2), target.valid_samples)
    assert torch.isfinite(target.target_probs).all()
    # valid_samples all False → denominator 0 → zero-gradient loss
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(log_p_k.grad, torch.zeros_like(log_p_k.grad))
