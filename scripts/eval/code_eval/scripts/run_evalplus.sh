#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export HUMANEVAL_OVERRIDE_PATH="${REPO_ROOT}/code_eval/data/HumanEvalPlus.jsonl"
export MBPP_OVERRIDE_PATH="${REPO_ROOT}/code_eval/data/MbppPlus.jsonl"

DATASET=${1:-humaneval}
MODEL=${2:-"Qwen/Qwen3-4B"}
GREEDY=${3:-1}
TEMP=${4:-0.8}
TOP_P=${5:-0.9}
N_SAMPLES=${6:-1}

if [ "$GREEDY" -eq 1 ]; then
    N_SAMPLES=1
fi

echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Greedy: $GREEDY (1=yes, 0=no)"
echo "Temperature: $TEMP"
echo "Top-P: $TOP_P"
echo "Number of samples: $N_SAMPLES"

MODEL_BASE=$(basename "$MODEL")
echo "Model base: $MODEL_BASE"

CODE_CODING_ROOT="${REPO_ROOT}/code_eval/coding"
export PYTHONPATH="${CODE_CODING_ROOT}/evalplus:${PYTHONPATH:-}"

if [ "$GREEDY" -eq 1 ]; then
    python3 "${CODE_CODING_ROOT}/evalplus/evalplus/codegen.py" --model "$MODEL" \
                    --dataset "$DATASET" \
                    --backend vllm \
                    --trust_remote_code \
                    --greedy
    TEMP_VAL="0.0"
else
    python3 "${CODE_CODING_ROOT}/evalplus/evalplus/codegen.py" --model "$MODEL" \
                    --dataset "$DATASET" \
                    --backend vllm \
                    --temperature "$TEMP" \
                    --top-p "$TOP_P" \
                    --trust_remote_code \
                    --n-samples "$N_SAMPLES"
    TEMP_VAL="$TEMP"
fi

echo "Waiting for output file to be generated..."
sleep 2

OUTPUT_FILE=$(find "${CODE_CODING_ROOT}/evalplus_results/${DATASET}" -name "*${MODEL_BASE}_vllm_temp_${TEMP_VAL}.jsonl" ! -name "*.raw.jsonl" -type f | head -n 1)

python3 -m evalplus.evaluate \
    --dataset "$DATASET" \
    --samples "$OUTPUT_FILE" \
    --output_file "${CODE_CODING_ROOT}/evalplus_results/${DATASET}/${MODEL_BASE}_eval_results.json" \
    --min-time-limit 10.0 \
    --gt-time-limit-factor 8.0

echo "Evaluation complete. Results saved to ${CODE_CODING_ROOT}/evalplus_results/${DATASET}/${MODEL_BASE}_eval_results.json"
