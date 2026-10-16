import torch

from verl.trainer.ppo.core_algos import build_mpopd_target, compute_mpopd_kl_loss


def test_only_student_log_probs_receive_gradient():
    # Reverse KL(student || target): gradient flows only through the student
    # (target is detached). When the target concentrates on token 0 (0.9 vs
    # 0.1) and the student is uniform, the gradient on token 0 is negative
    # (gradient descent raises its logit toward the target mode).
    student_log_probs = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    target = torch.tensor([[[0.9, 0.1]]])
    loss = compute_mpopd_kl_loss(
        student_log_probs,
        target,
        torch.ones(1, 1),
        torch.tensor([True]),
    )
    loss.backward()

    assert student_log_probs.grad is not None
    assert student_log_probs.grad[0, 0, 0] < 0


def test_invalid_sample_has_zero_gradient():
    student_log_probs = torch.zeros(1, 1, 2, requires_grad=True)
    loss = compute_mpopd_kl_loss(
        student_log_probs,
        torch.tensor([[[0.5, 0.5]]]),
        torch.ones(1, 1),
        torch.tensor([False]),
    )
    loss.backward()

    assert torch.equal(student_log_probs.grad, torch.zeros_like(student_log_probs))


def test_mpopd_target_is_detached_from_all_input_tensors():
    specialized = torch.zeros(1, 1, 2, 2, requires_grad=True)
    target = build_mpopd_target(
        specialized,
        torch.zeros(1, 1, 2, requires_grad=True),
        torch.zeros(1, 1, 2, requires_grad=True),
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert not target.target_probs.requires_grad
    assert not target.target_log_probs.requires_grad
    assert not target.expert_weights.requires_grad


def test_all_unavailable_target_is_finite_and_loss_is_zero():
    target = build_mpopd_target(
        torch.zeros(1, 2, 3, 2),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, dtype=torch.bool),
    )
    student_log_probs = torch.zeros(1, 2, 3, requires_grad=True)
    loss = compute_mpopd_kl_loss(
        student_log_probs,
        target.target_probs,
        torch.ones(1, 2),
        target.valid_samples,
    )
    assert torch.isfinite(target.target_probs).all()
    assert loss.item() == 0.0
