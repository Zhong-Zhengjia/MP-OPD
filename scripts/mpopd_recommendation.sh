#!/usr/bin/env bash
# Launch real MP-OPD recommendation training (full loop, not a smoke test).
#
# This launcher is the supported entry point: it fixes the MP-OPD training mode,
# the four-expert order, the default fusion temperatures, the clean/raw-chat
# retention, console-only logging, and disables reward KL, actor KL loss, and
# validation by default. It selects FSDP for the actor and vLLM for rollout.
#
# Required assets (defaults point at the server checkpoint share; override via env):
#   STUDENT_MODEL_PATH  Hugging Face student checkpoint (updated by training)
#   EXPERT_MODEL_PATH   Frozen teacher/expert checkpoint (shared by all 4 experts)
#   TRAIN_FILES / VAL_FILES  Parquet produced by scripts/prepare_train_parquet.sh
#
# Example (defaults already point at the configured server paths and 4 GPUs):
#   bash scripts/prepare_train_parquet.sh
#   bash scripts/mpopd_recommendation.sh
#
# To override the GPU count, pass it (it takes precedence over the default 4):
#   bash scripts/mpopd_recommendation.sh trainer.n_gpus_per_node=8
#
# Quick smoke run (small data, 2 steps, no checkpoint) to verify the full
# MP-OPD data flow end-to-end before committing to a full run:
#   MPOPD_QUICK=1 bash scripts/mpopd_recommendation.sh
# Once it prints mpopd_kl_loss and global_step=2, drop MPOPD_QUICK for the full run.
#
# Extra arguments are forwarded unchanged as Hydra overrides; later overrides
# take precedence. The student and expert tokenizers must share an identical
# token-to-ID vocabulary (MP-OPD scores the student's exact top-k token IDs).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${REPO_ROOT}/verl"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/home/hechongchong/.jupyter/zhongzhengjia/www/basemodel/student}"
EXPERT_MODEL_PATH="${EXPERT_MODEL_PATH:-/home/hechongchong/.jupyter/zhongzhengjia/www/basemodel/teacher}"
TRAIN_FILES="${TRAIN_FILES:-${REPO_ROOT}/data/recommendation/train.parquet}"
VAL_FILES="${VAL_FILES:-${REPO_ROOT}/data/recommendation/validation.parquet}"

# Where checkpoints land. Default is MP-OPD/models/mpopd_recommendation so
# saved models live inside the repo (not under verl/checkpoints/), each run
# writing last/ plus global_step_{N}/ subdirs there. Override via MODEL_DIR.
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/mpopd_recommendation}"

# Log every step's metrics to a JSONL file so results survive after the
# terminal scrolls. verl's "file" backend writes {step, data} per line.
# To also log to wandb, set LOGGERS="console,wandb" and export WANDB_API_KEY.
export VERL_FILE_LOGGER_ROOT="${VERL_FILE_LOGGER_ROOT:-${REPO_ROOT}/logs}"
LOGGERS="${LOGGERS:-console,file}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mpopd_recommendation}"

# MPOPD_QUICK=1: cap to 8 samples for a fast end-to-end check. Lengths stay
# at the full production budget (32768 prompt / 4096 response) — recommendation
# prompts are 7k-10k tokens and must not be truncated. The slow reference
# attention kernels (causal_conv1d / flash-linear-attention, used by Qwen3.5
# linear-attention layers when the optimized CUDA kernels are unavailable)
# make each step slow. The NCCL timeout is 300s: when ranks are in sync every
# all-gather completes in milliseconds, so a short timeout never trips on
# genuine (slow-but-in-sync) compute — it only fails fast on a real desync
# (one rank never entering a collective). Full-speed training still requires
# the optimized kernels.
if [[ "${MPOPD_QUICK:-0}" == "1" ]]; then
  QUICK_ARGS=(
    data.train_max_samples=8
    data.train_batch_size=8
    trainer.save_freq=-1
  )
  # total_training_steps is intentionally NOT set here: it auto-derives as
  # len(dataloader) * total_epochs (ray_trainer.py, when the yaml default
  # null stays). With 8 samples / batch 8 = 1 step per epoch, so
  # total_epochs=N -> N steps. Set total_epochs only.
  echo "MPOPD_QUICK=1: 8 samples, auto steps (1 step/epoch x total_epochs), full 32768 prompt / 4096 response — verifying the data flow only."
else
  QUICK_ARGS=()
fi

cd "${VERL_ROOT}"

# Free shared memory before launching. The container caps /dev/shm at 512M and
# stale Ray/NCCL/PSM shared-memory files from prior crashed runs otherwise leave
# too little for the DataLoader workers and Ray object store, causing
# "unable to allocate shared memory" failures. Safe to remove: these are
# throwaway IPC files, recreated on demand by a fresh run.
rm -f /dev/shm/psm_* /dev/shm/nccl-* /dev/shm/torch_* 2>/dev/null || true

# dataloader_num_workers=0 (set below in the python args): load batches in
# the main process. This is REQUIRED, not optional, here. The container caps
# /dev/shm at 512M, and any num_workers>0 has PyTorch pass batch tensors from
# the worker subprocess to the main process through shared memory
# (_share_fd_cpu_ -> multiprocessing.queues._feed). At batch_size=128 with
# 32768-token prompts a single batch's tensors are ~100s of MB; with 4
# workers x 4 DP ranks = 16 workers that immediately oversubscribes the
# 512M cap -> "unable to allocate shared memory(shm): Resource temporarily
# unavailable (11)" on the very first batch (progress stuck at 0/54).
# num_workers=2 hit the same wall (just later, at checkpoint time, as an OOM
# SIGKILL of an idle worker -> fatal "DataLoader worker killed"). 0 avoids
# the shm path entirely. prefetch_factor is kept for config symmetry but is
# a no-op at num_workers=0 (ray_trainer.py only passes it when >0).
# Throughput is barely affected: rollout (vLLM) + FSDP forward/backward
# dominate step time (gen ~43s of a ~56s step).
# NOTE: keep this comment OUTSIDE the backslash-continued python command --
# a '#' inside a continued line comments out every arg after it.

exec python -m verl.trainer.main_ppo \
  algorithm.train_mode=multi_prompt_distill \
  'algorithm.mp_opd.expert_names=[chasing,long_term,repurchase,generalized]' \
  algorithm.mp_opd.top_k=10 \
  algorithm.mp_opd.expert_temperature=1.0 \
  algorithm.mp_opd.token_temperature=1.0 \
  algorithm.mp_opd.lambda_value=1.0 \
  algorithm.mp_opd.expert_forward_micro_batch_size=1 \
  algorithm.mp_opd.max_expert_prompt_length=32768 \
  critic.enable=false \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  +actor_rollout_ref.ref.model.override_config.attn_implementation=sdpa \
  data.return_raw_chat=true \
  data.truncation=left \
  data.max_prompt_length=32768 \
  data.max_response_length=4096 \
  data.dataloader_num_workers=0 \
  +data.dataloader_prefetch_factor=1 \
  "data.train_files=${TRAIN_FILES}" \
  "data.val_files=${VAL_FILES}" \
  "actor_rollout_ref.model.path=${STUDENT_MODEL_PATH}" \
  "+actor_rollout_ref.model.base_model_path=${STUDENT_MODEL_PATH}" \
  "+actor_rollout_ref.ref.model.path=${EXPERT_MODEL_PATH}" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.nccl_timeout=300 \
  teacher_rollout.nccl_timeout=300 \
  data.train_batch_size=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.mpopd_lr_scale=1.0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=36864 \
  actor_rollout_ref.rollout.max_num_batched_tokens=36864 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  "trainer.logger=[${LOGGERS}]" \
  trainer.project_name=mpopd \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  "trainer.default_local_dir=${MODEL_DIR}" \
  +ray_kwargs.ray_init.object_store_memory=83886080 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=4 \
  "${QUICK_ARGS[@]}" \
  "$@"
