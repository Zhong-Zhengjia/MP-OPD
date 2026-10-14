#!/usr/bin/env bash
# Produce the Parquet files the MP-OPD training launcher expects.
#
# Converts the normalized JSONL (data/train_1000.jsonl, produced by
# scripts/prepare_train_1000.sh) into Parquet at the paths the launcher reads by
# default: data/recommendation/train.parquet (all rows) and
# data/recommendation/validation.parquet (a small subset — validation is never
# evaluated by default, so it only needs to be a valid same-schema file).
#
# Requires pyarrow (part of the verl training env). Override paths with env:
#   TRAIN_JSONL, OUT_DIR, TRAIN_PARQUET, VAL_PARQUET, VAL_ROWS

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

JSONL="${TRAIN_JSONL:-${REPO_ROOT}/data/train_1000.jsonl}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/recommendation}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${OUT_DIR}/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${OUT_DIR}/validation.parquet}"
VAL_ROWS="${VAL_ROWS:-2}"

if [[ ! -f "${JSONL}" ]]; then
  echo "Missing ${JSONL}; generating it via scripts/prepare_train_1000.sh first..." >&2
  bash scripts/prepare_train_1000.sh
fi
if [[ ! -f "${JSONL}" ]]; then
  echo "ERROR: ${JSONL} still missing after prepare_train_1000.sh" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${REPO_ROOT}/data_pipeline/src${PYTHONPATH:+:${PYTHONPATH}}"

PY=python
if ! command -v python >/dev/null 2>&1; then PY=python3; fi

echo "Writing ${TRAIN_PARQUET} (all rows)..."
"${PY}" -m mpopd_data.converters.jsonl_to_parquet --input "${JSONL}" --output "${TRAIN_PARQUET}"

echo "Writing ${VAL_PARQUET} (first ${VAL_ROWS} rows)..."
"${PY}" -m mpopd_data.converters.jsonl_to_parquet --input "${JSONL}" --output "${VAL_PARQUET}" --limit "${VAL_ROWS}"

echo "Done. Launch training with:"
echo "  STUDENT_MODEL_PATH=<ckpt> EXPERT_MODEL_PATH=<ckpt> bash scripts/mpopd_recommendation.sh trainer.n_gpus_per_node=<N>"
