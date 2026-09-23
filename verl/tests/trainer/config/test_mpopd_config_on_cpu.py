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

from verl.trainer.config.algorithm import AlgoConfig, MPOPDConfig


def test_defaults():
    cfg = MPOPDConfig()

    assert cfg.expert_names == ["chasing", "long_term", "repurchase", "generalized"]
    assert set(cfg.expert_instructions) == set(cfg.expert_names)
    assert all(cfg.expert_instructions[name].strip() for name in cfg.expert_names)
    assert (cfg.top_k, cfg.expert_temperature, cfg.token_temperature) == (10, 1.0, 1.0)
    assert cfg.lambda_value == 1.0
    assert cfg.skip_samples_without_active_experts is True
    assert cfg.max_expert_prompt_length == 4096


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expert_names": []},
        {"expert_names": ["x", "x"]},
        {"top_k": 0},
        {"expert_temperature": 0.0},
        {"token_temperature": -1.0},
        {"lambda_value": 0.0},
        {"expert_forward_micro_batch_size": -1},
        {"max_expert_prompt_length": 0},
        {"expert_priors": [1.0, 0.0, 1.0, 1.0]},
    ],
)
def test_invalid_values(kwargs):
    with pytest.raises(ValueError):
        MPOPDConfig(**kwargs)


def test_prior_count_matches_experts():
    with pytest.raises(ValueError, match="expert_priors"):
        MPOPDConfig(
            expert_names=["a", "b"],
            expert_instructions={"a": "focus a", "b": "focus b"},
            expert_priors=[1.0],
        )


def test_instruction_keys_and_values_are_validated():
    with pytest.raises(ValueError, match="expert_instructions"):
        MPOPDConfig(expert_instructions={"chasing": "only one"})

    with pytest.raises(ValueError, match="non-empty"):
        MPOPDConfig(
            expert_instructions={
                "chasing": "c",
                "long_term": "l",
                "repurchase": "",
                "generalized": "g",
            }
        )


def test_algo_config_exposes_typed_mpopd_defaults():
    cfg = AlgoConfig()

    assert cfg.train_mode == "ppo"
    assert isinstance(cfg.mp_opd, MPOPDConfig)
