#!/usr/bin/env bash
set -Eeuo pipefail

# One-update MP-OPD validation for a real server with GPUs and local checkpoints.
# This script intentionally uses the synchronous Hugging Face rollout so that
# vLLM/FlashAttention compatibility is not part of the first training check.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VERL_ROOT="${REPO_ROOT}/verl"

: "${MPOPD_SMOKE_MODEL:?Set MPOPD_SMOKE_MODEL to the student checkpoint}"
: "${MPOPD_SMOKE_REF_MODEL:?Set MPOPD_SMOKE_REF_MODEL to the frozen expert/reference checkpoint}"
: "${MPOPD_SMOKE_TRAIN_PARQUET:?Set MPOPD_SMOKE_TRAIN_PARQUET to a small MP-OPD parquet file}"

GPU_COUNT="${MPOPD_SMOKE_GPU_COUNT:-8}"
TRAIN_SAMPLES="${MPOPD_SMOKE_TRAIN_SAMPLES:-2}"
TOP_K="${MPOPD_SMOKE_TOP_K:-3}"
RESPONSE_LENGTH="${MPOPD_SMOKE_RESPONSE_LENGTH:-32}"
PPO_MINI_BATCH_SIZE="${MPOPD_SMOKE_PPO_MINI_BATCH_SIZE:-8}"
LOG_FILE="${MPOPD_SMOKE_LOG:-/tmp/mpopd-gpu-smoke-real.log}"

cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false

echo "MP-OPD smoke configuration:"
echo "  repository: ${REPO_ROOT}"
echo "  student:    ${MPOPD_SMOKE_MODEL}"
echo "  reference:  ${MPOPD_SMOKE_REF_MODEL}"
echo "  parquet:    ${MPOPD_SMOKE_TRAIN_PARQUET}"
echo "  GPUs:       ${GPU_COUNT}"
echo "  log:        ${LOG_FILE}"

if [[ ! -f "${MPOPD_SMOKE_TRAIN_PARQUET}" ]]; then
  echo "ERROR: parquet file does not exist: ${MPOPD_SMOKE_TRAIN_PARQUET}" >&2
  exit 2
fi

set +e
python -m verl.trainer.main_ppo \
  algorithm.train_mode=multi_prompt_distill \
  algorithm.adv_estimator=grpo \
  algorithm.mp_opd.top_k="${TOP_K}" \
  data.train_files="${MPOPD_SMOKE_TRAIN_PARQUET}" \
  data.val_files="${MPOPD_SMOKE_TRAIN_PARQUET}" \
  data.train_batch_size="${TRAIN_SAMPLES}" \
  data.train_max_samples="${TRAIN_SAMPLES}" \
  data.max_response_length="${RESPONSE_LENGTH}" \
  data.return_raw_chat=true \
  trainer.n_gpus_per_node="${GPU_COUNT}" \
  actor_rollout_ref.model.path="${MPOPD_SMOKE_MODEL}" \
  actor_rollout_ref.model.base_model_path="${MPOPD_SMOKE_MODEL}" \
  actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.ref.model.path="${MPOPD_SMOKE_REF_MODEL}" \
  actor_rollout_ref.actor.mpopd_lr_scale=1.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.logger=console \
  2>&1 | tee "${LOG_FILE}"
TRAINING_EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "training_exit_code=${TRAINING_EXIT_CODE}" | tee -a "${LOG_FILE}"

if [[ "${TRAINING_EXIT_CODE}" -ne 0 ]]; then
  echo "FAIL: the Python training process exited with ${TRAINING_EXIT_CODE}" >&2
  exit "${TRAINING_EXIT_CODE}"
fi

if ! grep -Eq 'global_step[^0-9]*1|global_step[^0-9]*: *1' "${LOG_FILE}"; then
  echo "FAIL: no completed global_step=1 was found; this may have only been config expansion." >&2
  exit 3
fi

if ! grep -Eq 'actor/mpopd_kl_loss|mpopd_kl_loss' "${LOG_FILE}"; then
  echo "FAIL: no MP-OPD KL metric was found in the training log." >&2
  exit 4
fi

if grep -Eiq 'Traceback|CUDA out of memory|ActorDiedError|Missing mandatory value|ImportError|nan|inf' "${LOG_FILE}"; then
  echo "FAIL: the log contains a fatal runtime marker." >&2
  exit 5
fi

echo "PASS: MP-OPD completed one real training update."
