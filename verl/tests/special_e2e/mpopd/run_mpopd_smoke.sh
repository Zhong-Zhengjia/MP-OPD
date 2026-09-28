#!/usr/bin/env bash
set -euo pipefail

: "${MPOPD_SMOKE_MODEL:?Set MPOPD_SMOKE_MODEL to a local small causal-LM checkpoint}"
: "${MPOPD_SMOKE_TRAIN_PARQUET:?Set MPOPD_SMOKE_TRAIN_PARQUET to a two-row MP-OPD parquet file}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SMOKE_LOG="${MPOPD_SMOKE_LOG:-$(mktemp /tmp/mpopd-smoke.XXXXXX)}"

cd "${VERL_ROOT}"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  algorithm.train_mode=multi_prompt_distill \
  'algorithm.mp_opd.expert_names=[chasing,long_term]' \
  'algorithm.mp_opd.expert_instructions={chasing:"focus recent interest",long_term:"focus stable preference"}' \
  algorithm.mp_opd.top_k=3 \
  algorithm.mp_opd.expert_forward_micro_batch_size=1 \
  "data.train_files=${MPOPD_SMOKE_TRAIN_PARQUET}" \
  "data.val_files=${MPOPD_SMOKE_TRAIN_PARQUET}" \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  data.max_response_length=32 \
  data.return_raw_chat=true \
  "actor_rollout_ref.model.path=${MPOPD_SMOKE_MODEL}" \
  "+actor_rollout_ref.model.base_model_path=${MPOPD_SMOKE_MODEL}" \
  "+actor_rollout_ref.ref.model.path=${MPOPD_SMOKE_MODEL}" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.mpopd_lr_scale=1.0 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  trainer.logger=console \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  2>&1 | tee "${SMOKE_LOG}"

for metric in actor/mpopd_kl_loss actor/mpopd_top_k training/global_step; do
  rg -q "${metric}:" "${SMOKE_LOG}"
done

if rg -i -q 'actor/mpopd_kl_loss:(nan|[-+]?inf)' "${SMOKE_LOG}"; then
  echo "MP-OPD smoke produced a non-finite loss; inspect ${SMOKE_LOG}" >&2
  exit 1
fi

echo "MP-OPD smoke passed; log: ${SMOKE_LOG}"
