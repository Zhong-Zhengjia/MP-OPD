#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${REPO_ROOT}/verl"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${REPO_ROOT}/models/Qwen3-8B}"
EXPERT_MODEL_PATH="${EXPERT_MODEL_PATH:-${REPO_ROOT}/models/Qwen3-4B}"
TRAIN_FILES="${TRAIN_FILES:-${REPO_ROOT}/data/recommendation/train.parquet}"
VAL_FILES="${VAL_FILES:-${REPO_ROOT}/data/recommendation/validation.parquet}"

cd "${VERL_ROOT}"

exec python -m verl.trainer.main_ppo \
  algorithm.train_mode=multi_prompt_distill \
  'algorithm.mp_opd.expert_names=[chasing,long_term,repurchase,generalized]' \
  algorithm.mp_opd.top_k=10 \
  algorithm.mp_opd.expert_temperature=1.0 \
  algorithm.mp_opd.token_temperature=1.0 \
  algorithm.mp_opd.lambda_value=1.0 \
  algorithm.mp_opd.expert_forward_micro_batch_size=1 \
  data.return_raw_chat=true \
  "data.train_files=${TRAIN_FILES}" \
  "data.val_files=${VAL_FILES}" \
  "actor_rollout_ref.model.path=${STUDENT_MODEL_PATH}" \
  "+actor_rollout_ref.model.base_model_path=${STUDENT_MODEL_PATH}" \
  "+actor_rollout_ref.ref.model.path=${EXPERT_MODEL_PATH}" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.mpopd_lr_scale=1.0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  trainer.logger=console \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  "$@"
