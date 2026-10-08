#!/usr/bin/env bash
set -euo pipefail

# Prepare and validate the current recommendation train_1000 split.
#
# Override paths when the server layout differs:
#   SOURCE_JSONL=/data/datasets/train_1000.jsonl \
#   CLEAN_TEMPLATE=/data/baseline/student_prompt.txt \
#   bash scripts/prepare_train_1000.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_JSONL="${SOURCE_JSONL:-${REPO_ROOT}/../datasets/ours/splits/train_1000.jsonl}"
CLEAN_TEMPLATE="${CLEAN_TEMPLATE:-${REPO_ROOT}/../baseline/student_prompt.txt}"
EXPERT_PROMPT_DIR="${EXPERT_PROMPT_DIR:-${REPO_ROOT}/prompts}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${REPO_ROOT}/data/train_1000.jsonl}"
VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-${REPO_ROOT}/data/train_1000.validation.jsonl}"

cd "${REPO_ROOT}"
if [[ ! -f "${SOURCE_JSONL}" ]]; then
  echo "Missing source JSONL: ${SOURCE_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${CLEAN_TEMPLATE}" ]]; then
  echo "Missing clean prompt template: ${CLEAN_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -d "${EXPERT_PROMPT_DIR}" ]]; then
  echo "Missing expert prompt directory: ${EXPERT_PROMPT_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/data_pipeline/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m mpopd_data.cli recommendation \
  --input "${SOURCE_JSONL}" \
  --output "${OUTPUT_JSONL}" \
  --clean-template "${CLEAN_TEMPLATE}" \
  --expert-prompt-dir "${EXPERT_PROMPT_DIR}"

python3 -m mpopd_data.cli recommendation \
  --input "${SOURCE_JSONL}" \
  --output "${VALIDATION_OUTPUT}" \
  --clean-template "${CLEAN_TEMPLATE}" \
  --expert-prompt-dir "${EXPERT_PROMPT_DIR}" \
  --validate-only

echo "Prepared: ${OUTPUT_JSONL}"
echo "Validated: ${VALIDATION_OUTPUT}"
