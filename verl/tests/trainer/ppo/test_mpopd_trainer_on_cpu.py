from types import SimpleNamespace

import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _FakeWorkerGroup:
    def __init__(self):
        self.calls = []

    def generate_sequences(self, batch):
        self.calls.append("generate_sequences")
        batch.batch["responses"] = torch.tensor([[7, 8]])
        batch.batch["response_mask"] = torch.ones(1, 2, dtype=torch.long)
        return batch

    def prepare_mpopd_log_probs(self, batch):
        self.calls.append("prepare_mpopd_log_probs")
        return DataProto.from_dict(
            tensors={
                "student_topk_ids": torch.zeros(1, 2, 2, dtype=torch.long),
                "old_log_probs": torch.zeros(1, 2, 2),
                "student_base_topk_log_probs": torch.zeros(1, 2, 2),
                "teacher_base_topk_log_probs": torch.zeros(1, 2, 2),
                "specialized_expert_topk_log_probs": torch.zeros(1, 2, 2, 1),
                "expert_mask": torch.ones(1, 1, dtype=torch.bool),
            }
        )

    def update_actor_mpopd(self, batch):
        self.calls.append("update_actor_mpopd")
        return DataProto(meta_info={"metrics": {"actor/mpopd_kl_loss": 0.0}})


def _trainer():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    trainer.actor_rollout_wg = _FakeWorkerGroup()
    trainer.global_steps = 0
    trainer.actor_update_steps = 0
    trainer._ensure_response_mask = lambda batch: batch

    def prepare(batch):
        output = trainer.actor_rollout_wg.prepare_mpopd_log_probs(batch)
        return batch.union(output), {"prepare_s": 0.0}

    trainer._prepare_mpopd_update_batch = prepare
    return trainer


def test_mpopd_step_calls_rollout_prepare_and_update_in_order():
    trainer = _trainer()
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "position_ids": torch.arange(2).repeat(1, 1),
        }
    )
    metrics = trainer._run_mpopd_step(batch, {}, {})

    assert trainer.actor_rollout_wg.calls == [
        "generate_sequences",
        "prepare_mpopd_log_probs",
        "update_actor_mpopd",
    ]
    assert metrics["actor/mpopd_kl_loss"] == 0.0
