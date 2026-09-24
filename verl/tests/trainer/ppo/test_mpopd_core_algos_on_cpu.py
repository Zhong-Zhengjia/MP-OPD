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

import pytest
import torch

from verl.trainer.ppo.core_algos import build_mpopd_target, compute_mpopd_kl_loss


def test_absolute_routes_but_signed_delta_fuses():
    specialized = torch.tensor([[[[3.0, -4.0], [1.0, 1.0]]]])

    target = build_mpopd_target(
        specialized,
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.tensor([[True, True]]),
    )

    assert target.expert_weights[0, 0, 0, 1] > target.expert_weights[0, 0, 0, 0]
    assert target.fused_delta[0, 0, 0] < 0
    assert target.target_probs[0, 0, 0] < target.anchor_probs[0, 0, 0]


def test_zero_delta_returns_anchor():
    anchor_log_probs = torch.randn(2, 3, 4)

    target = build_mpopd_target(
        torch.zeros(2, 3, 4, 2),
        torch.zeros(2, 3, 4),
        anchor_log_probs,
        torch.ones(2, 2, dtype=torch.bool),
    )

    assert torch.allclose(target.target_probs, torch.softmax(anchor_log_probs, dim=-1))


def test_all_unavailable_is_finite_zero_weight_and_invalid():
    anchor_log_probs = torch.randn(1, 2, 3)

    target = build_mpopd_target(
        torch.randn(1, 2, 3, 2),
        torch.zeros(1, 2, 3),
        anchor_log_probs,
        torch.zeros(1, 2, dtype=torch.bool),
    )

    assert torch.isfinite(target.target_probs).all()
    assert torch.equal(target.expert_weights, torch.zeros_like(target.expert_weights))
    assert torch.equal(target.fused_delta, torch.zeros_like(target.fused_delta))
    assert torch.allclose(target.target_probs, torch.softmax(anchor_log_probs, dim=-1))
    assert target.valid_samples.tolist() == [False]


def test_disabled_non_finite_expert_cannot_contaminate_fused_target():
    target = build_mpopd_target(
        torch.tensor([[[[torch.nan, 1.0], [torch.inf, 0.0]]]]),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.tensor([[False, True]]),
    )

    assert torch.isfinite(target.fused_delta).all()
    assert torch.isfinite(target.target_probs).all()


def test_target_invariants_and_priors():
    specialized = torch.tensor([[[[2.0, 2.0], [0.0, 0.0]]]], requires_grad=True)

    target = build_mpopd_target(
        specialized,
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.ones(1, 2, dtype=torch.bool),
        expert_priors=torch.tensor([3.0, 1.0]),
    )

    assert target.expert_weights[0, 0, 0, 0] > target.expert_weights[0, 0, 0, 1]
    assert target.target_probs[0, 0, 0] > target.anchor_probs[0, 0, 0]
    assert torch.allclose(target.target_probs.sum(dim=-1), torch.ones(1, 1))
    for value in (
        target.target_probs,
        target.target_log_probs,
        target.anchor_probs,
        target.expert_weights,
        target.expert_delta,
        target.fused_delta,
    ):
        assert not value.requires_grad


def test_temperatures_control_expert_routing_and_token_signal_strength():
    specialized = torch.tensor([[[[4.0, 1.0], [0.0, 0.0]]]])
    teacher_base = torch.zeros(1, 1, 2)
    student_base = torch.zeros(1, 1, 2)
    expert_mask = torch.ones(1, 2, dtype=torch.bool)

    sharp_experts = build_mpopd_target(
        specialized,
        teacher_base,
        student_base,
        expert_mask,
        expert_temperature=0.25,
    )
    soft_experts = build_mpopd_target(
        specialized,
        teacher_base,
        student_base,
        expert_mask,
        expert_temperature=4.0,
    )
    sharp_tokens = build_mpopd_target(
        specialized,
        teacher_base,
        student_base,
        expert_mask,
        token_temperature=0.25,
    )
    soft_tokens = build_mpopd_target(
        specialized,
        teacher_base,
        student_base,
        expert_mask,
        token_temperature=4.0,
    )

    assert sharp_experts.expert_weights[0, 0, 0, 0] > soft_experts.expert_weights[0, 0, 0, 0]
    assert sharp_tokens.target_probs[0, 0, 0] > soft_tokens.target_probs[0, 0, 0]


@pytest.mark.parametrize("field", ["expert_temperature", "token_temperature", "lambda_value"])
def test_positive_scale_parameters_are_required(field):
    kwargs = {field: 0.0}

    with pytest.raises(ValueError, match=field):
        build_mpopd_target(
            torch.zeros(1, 1, 2, 1),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.ones(1, 1, dtype=torch.bool),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("prior", "message"),
    [
        (torch.tensor([1.0]), "expert_priors"),
        (torch.tensor([1.0, 0.0]), "positive"),
    ],
)
def test_expert_priors_must_match_the_expert_axis_and_be_positive(prior, message):
    with pytest.raises(ValueError, match=message):
        build_mpopd_target(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            expert_priors=prior,
        )


def test_kl_zero_at_target_and_masks_invalid_rows():
    target = torch.tensor([[[0.8, 0.2]], [[0.5, 0.5]]])
    student = target.log().clone().requires_grad_(True)

    loss = compute_mpopd_kl_loss(
        student,
        target,
        torch.ones(2, 1),
        torch.tensor([True, False]),
    )

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)
    loss.backward()
    assert torch.allclose(student.grad, torch.zeros_like(student.grad), atol=1e-7)


def test_kl_masks_response_positions_and_averages_only_valid_tokens():
    target = torch.tensor([[[0.75, 0.25], [0.5, 0.5]]])
    student = torch.zeros(1, 2, 2, requires_grad=True)
    expected = 0.75 * torch.log(torch.tensor(1.5)) + 0.25 * torch.log(torch.tensor(0.5))

    loss = compute_mpopd_kl_loss(
        student,
        target,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([True]),
    )

    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.equal(student.grad[:, 1], torch.zeros_like(student.grad[:, 1]))


def test_kl_all_invalid_retains_zero_student_gradient():
    student = torch.randn(1, 2, 3, requires_grad=True)

    loss = compute_mpopd_kl_loss(
        student,
        torch.full((1, 2, 3), 1 / 3),
        torch.ones(1, 2),
        torch.tensor([False]),
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student.grad))
