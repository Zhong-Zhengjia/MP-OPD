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
# Extra arguments are forwarded unchanged as Hydra overrides; later overrides
# take precedence. The student and expert tokenizers must share an identical
# token-to-ID vocabulary (MP-OPD scores the student's exact top-k token IDs).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${REPO_ROOT}/verl"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/share/yangshiyu05/models/Qwen/Qwen3.5-0.8B}"
EXPERT_MODEL_PATH="${EXPERT_MODEL_PATH:-/share/yangshiyu05/models/Qwen/Qwen3.5-4B}"
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
  algorithm.mp_opd.max_expert_prompt_length=8192 \
  critic.enable=false \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  +actor_rollout_ref.ref.model.override_config.attn_implementation=eager \
  data.return_raw_chat=true \
  data.truncation=left \
  data.max_prompt_length=8192 \
  data.max_response_length=2048 \
  data.dataloader_num_workers=2 \
  "data.train_files=${TRAIN_FILES}" \
  "data.val_files=${VAL_FILES}" \
  "actor_rollout_ref.model.path=${STUDENT_MODEL_PATH}" \
  "+actor_rollout_ref.model.base_model_path=${STUDENT_MODEL_PATH}" \
  "+actor_rollout_ref.ref.model.path=${EXPERT_MODEL_PATH}" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.mpopd_lr_scale=1.0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=10240 \
  actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  trainer.logger=console \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=4 \
  "$@"
