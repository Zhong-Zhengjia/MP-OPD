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
    # _run_mpopd_step returns (metrics, rollout_batch); unpack the tuple.
    metrics, _rollout_batch = trainer._run_mpopd_step(batch, {}, {})

    assert trainer.actor_rollout_wg.calls == [
        "generate_sequences",
        "prepare_mpopd_log_probs",
        "update_actor_mpopd",
    ]
    assert metrics["actor/mpopd_kl_loss"] == 0.0


def test_fit_mpopd_saves_checkpoint_on_save_freq(monkeypatch):
    """_fit_mpopd must call _save_checkpoint when save_freq > 0.

    Regression: the MP-OPD loop previously had no save logic at all (unlike the
    regular PPO fit), so passing save_freq was silently ignored and no
    checkpoint was ever written. Verify save fires on the last step and on
    every save_freq boundary.
    """
    import verl.utils.tracking as tracking_mod

    class _FakeTracking:
        def __init__(self, **kwargs):
            self.logs = []

        def log(self, data, step):
            self.logs.append((step, data))

    class _FakeTqdm:
        def __init__(self, total=None, desc=None):
            self.n = 0

        def update(self, n=1):
            self.n += n

        def close(self):
            pass

    def _fake_marked_timer(name, timing_raw, **kw):
        from contextlib import nullcontext

        return nullcontext()

    def _fake_compute_timing_metrics(batch, timing_raw):
        return {}

    trainer = _trainer()
    trainer.total_training_steps = 5
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            total_epochs=5,
            save_freq=2,
            project_name="t",
            experiment_name="t",
            logger=["console"],
        )
    )
    # OmegaConf.to_container(self.config, ...) needs a real (or fake) container
    # call; patch it to return a plain dict since the fake Tracking ignores it.
    import verl.trainer.ppo.ray_trainer as rt_mod
    from omegaconf import OmegaConf

    trainer.config = OmegaConf.create(
        {"trainer": {"total_epochs": 5, "save_freq": 2, "project_name": "t", "experiment_name": "t", "logger": ["console"]}}
    )

    # _fit_mpopd calls DataProto.from_single_dict(batch_dict), which expects a
    # plain dict of tensors (the real StatefulDataLoader yields dicts via
    # collate_fn -> {**tensors, **non_tensors}). Feeding DataProto here would
    # hit "'DataProto' object has no attribute 'items'", so yield a plain dict.
    trainer.train_dataloader = [
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "position_ids": torch.arange(2).repeat(1, 1),
        }
    ]
    trainer._is_empty_dataproto = lambda batch: False

    save_calls = []

    def _fake_save_checkpoint(metrics=None):
        save_calls.append(trainer.global_steps)

    trainer._save_checkpoint = _fake_save_checkpoint
    trainer._load_checkpoint = lambda: None

    # _fit_mpopd does local imports (from verl.utils.tracking import Tracking,
    # from tqdm import tqdm) at call time, so patch the source modules.
    monkeypatch.setattr(tracking_mod, "Tracking", _FakeTracking, raising=False)
    monkeypatch.setattr("tqdm.tqdm", _FakeTqdm, raising=False)
    monkeypatch.setattr(rt_mod, "marked_timer", _fake_marked_timer, raising=False)
    monkeypatch.setattr(rt_mod, "compute_timing_metrics", _fake_compute_timing_metrics, raising=False)

    # Make _run_mpopd_step deterministic and cheap.
    def _fake_run_step(batch, metrics, timing_raw):
        return {"actor/mpopd_kl_loss": float(trainer.global_steps)}, batch

    trainer._run_mpopd_step = _fake_run_step

    trainer._fit_mpopd()

    # save_freq=2, total=5 -> save at steps 1, 3, and 4 (last).
    # global_steps is the pre-increment index, so save fires at 1, 3, 4.
    assert len(save_calls) >= 1, "no checkpoint was saved during _fit_mpopd"
    assert save_calls[-1] == 4, f"last save should be at global_step=4, got {save_calls[-1]}"
