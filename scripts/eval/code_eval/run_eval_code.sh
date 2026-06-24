#!/bin/bash
#SBATCH --job-name=eval-code-rmopd
#SBATCH --output=logs/code_eval/slurm_%j.out
#SBATCH --error=logs/code_eval/slurm_%j.err
#SBATCH --chdir=/mnt/phwfile/datafrontier/wurong/code/RM-OPD
#SBATCH --account=research
#SBATCH --partition=DataFrontier_Explore
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail
set -x

REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
CODE_CODING_ROOT=${CODE_CODING_ROOT:-$REPO_ROOT/code_eval/coding}
cd "$CODE_CODING_ROOT"

EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$REPO_ROOT/outputs/code_eval_outputs}

export PYTHONPATH="${CODE_CODING_ROOT}/evalplus:${PYTHONPATH:-}"

CODE_BENCHMARK_ROOT=${CODE_BENCHMARK_ROOT:-$REPO_ROOT/code_eval/data}
EVALPLUS_DATA_ROOT=${EVALPLUS_DATA_ROOT:-$CODE_BENCHMARK_ROOT}
DEFAULT_HUMANEVAL_PATH="${EVALPLUS_DATA_ROOT}/HumanEvalPlus.jsonl"
DEFAULT_MBPP_PATH="${EVALPLUS_DATA_ROOT}/MbppPlus.jsonl"
LCB_DATA_ROOT=${LCB_DATA_ROOT:-$CODE_BENCHMARK_ROOT/LivecodeBench}

resolve_evalplus_dataset_path() {
    local name="$1"
    local override="${2:-}"
    local default_path="$3"
    if [[ -n "$override" && -f "$override" ]]; then
        echo "$override"
        return
    fi
    if [[ -n "$override" ]]; then
        echo "Ignoring invalid ${name}=${override}; using ${default_path}" >&2
    fi
    echo "$default_path"
}

_he_override="${HUMANEVAL_OVERRIDE_PATH:-}"
_mbpp_override="${MBPP_OVERRIDE_PATH:-}"
HUMANEVAL_OVERRIDE_PATH="$(resolve_evalplus_dataset_path HUMANEVAL_OVERRIDE_PATH "$_he_override" "$DEFAULT_HUMANEVAL_PATH")"
MBPP_OVERRIDE_PATH="$(resolve_evalplus_dataset_path MBPP_OVERRIDE_PATH "$_mbpp_override" "$DEFAULT_MBPP_PATH")"
export HUMANEVAL_OVERRIDE_PATH MBPP_OVERRIDE_PATH

is_true() {
    [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "True" ]]
}

# Usage (pick one):
#   1) Direct HF model path:
#        MODEL_PATH=/path/to/merged_hf sbatch scripts/eval/code_eval/run_eval_code.sh
#   2) Resolve from verl checkpoint:
#        CKPT_STEP=20 sbatch scripts/eval/code_eval/run_eval_code.sh
#   3) Benchmark switches (default: humaneval + mbpp on, lcb off):
#        RUN_HUMANEVAL=true  RUN_MBPP=true  RUN_LCB=false
#        RUN_HUMANEVAL=false RUN_MBPP=false RUN_LCB=true   # LCB only
#        RUN_HUMANEVAL=true  RUN_MBPP=false RUN_LCB=false  # HumanEval+ only
#   4) Optional inference overrides, shared by EvalPlus and LCB:
#        MODEL_NAME=my_model N_SAMPLES=4 TEMPERATURE=1.0 TOP_P=1.0 MAX_TOKENS=16384
#   5) Optional benchmark data root override:
#        CODE_BENCHMARK_ROOT=/path/to/benchmarks/code sbatch ...


# MODEL_PATH=/mnt/phwfile/datafrontier/public_models/Qwen3-4B-Non-Thinking-RL-Code-Step300
# CKPT_ROOT=/mnt/phwfile/datafrontier/fudaocheng/checkpoints/trained/MT_weak2strong@4/Qwen3-8B-T4B_CodeRL_Lam1.0_Gfalse_Otrue_0624_1200
# CKPT_STEP=20

has_model_weights() {
    local path="$1"
    compgen -G "${path}/*.safetensors" >/dev/null || compgen -G "${path}/*.bin" >/dev/null
}

resolve_model_name() {
    basename "$1"
}

model_identifier() {
    echo "$1" | sed 's#^\./##' | tr '/' '-'
}

if [[ -n "${MODEL_PATH:-}" ]]; then
    MODEL_PATH=$(cd "$MODEL_PATH" && pwd)
    MODEL_NAME=${MODEL_NAME:-$(resolve_model_name "$MODEL_PATH")}
else
    PROJECT_NAME=${PROJECT_NAME:-MT_weak2strong@4}
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-}
    CKPT_STEP=${CKPT_STEP:-20}
    if [[ -z "$EXPERIMENT_NAME" ]]; then
        echo "Set MODEL_PATH, or set CKPT_ROOT, or set PROJECT_NAME + EXPERIMENT_NAME + CKPT_STEP." >&2
        exit 1
    fi
    CKPT_ROOT=${CKPT_ROOT:-/mnt/phwfile/datafrontier/fudaocheng/checkpoints/trained/${PROJECT_NAME}/${EXPERIMENT_NAME}}
    for candidate in \
        "${CKPT_ROOT}/global_step_${CKPT_STEP}/actor/merged_hf" \
        "${CKPT_ROOT}/global_step_${CKPT_STEP}/actor/huggingface" \
        "${CKPT_ROOT}/global_step_${CKPT_STEP}/hf" \
        "${CKPT_ROOT}/global_step_${CKPT_STEP}"; do
        if [[ -d "$candidate" ]] && has_model_weights "$candidate"; then
            MODEL_PATH="$candidate"
            break
        fi
    done
    MODEL_NAME=${MODEL_NAME:-${EXPERIMENT_NAME}_step${CKPT_STEP}}
fi

if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]] || ! has_model_weights "$MODEL_PATH"; then
    echo "Cannot resolve MODEL_PATH." >&2
    echo "Option A: set MODEL_PATH to a HuggingFace directory containing *.safetensors or *.bin." >&2
    echo "Option B: leave MODEL_PATH unset and provide CKPT_ROOT + CKPT_STEP." >&2
    if [[ -z "${MODEL_PATH:-}" ]]; then
        echo "Tried CKPT_ROOT=${CKPT_ROOT:-unset}, CKPT_STEP=${CKPT_STEP:-unset}" >&2
        echo "If you only have a verl/FSDP checkpoint, merge it first, for example:" >&2
        echo "python3 -m verl.model_merger merge --backend fsdp \\" >&2
        echo "  --local_dir ${CKPT_ROOT}/global_step_${CKPT_STEP}/actor \\" >&2
        echo "  --target_dir ${CKPT_ROOT}/global_step_${CKPT_STEP}/actor/merged_hf" >&2
    else
        echo "MODEL_PATH=${MODEL_PATH} is missing weight files." >&2
    fi
    exit 1
fi

# G-OPD paper code eval defaults and scripts/run_lcb_gen.sh paper settings:
# n=4, temperature=1.0, top_p=1.0, max_tokens=16384.
N_SAMPLES=${N_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
MAX_TOKENS=${MAX_TOKENS:-16384}
TP=${TP:-2}
GREEDY=${GREEDY:-0}
RUN_HUMANEVAL=${RUN_HUMANEVAL:-false}
RUN_MBPP=${RUN_MBPP:-true}
RUN_LCB=${RUN_LCB:-false}
LCB_ROOT=${LCB_ROOT:-$CODE_CODING_ROOT/LiveCodeBench}
LCB_MODEL=${LCB_MODEL:-Qwen3-4B-NonThinking}
LCB_RELEASE_VERSION=${LCB_RELEASE_VERSION:-v6}

MODEL_ID=$(model_identifier "$MODEL_PATH")
if [[ "$GREEDY" == "1" || "$GREEDY" == "true" || "$GREEDY" == "True" ]]; then
    SAMPLES_ID="samples_n1_temp0.0"
else
    SAMPLES_ID="samples_n${N_SAMPLES}_temp${TEMPERATURE}"
fi

if ! is_true "$RUN_HUMANEVAL" && ! is_true "$RUN_MBPP" && ! is_true "$RUN_LCB"; then
    echo "At least one of RUN_HUMANEVAL, RUN_MBPP, RUN_LCB must be enabled." >&2
    exit 1
fi

if is_true "$RUN_HUMANEVAL" && [[ ! -f "$HUMANEVAL_OVERRIDE_PATH" ]]; then
    echo "Missing EvalPlus dataset: ${HUMANEVAL_OVERRIDE_PATH}" >&2
    exit 1
fi
if is_true "$RUN_MBPP" && [[ ! -f "$MBPP_OVERRIDE_PATH" ]]; then
    echo "Missing EvalPlus dataset: ${MBPP_OVERRIDE_PATH}" >&2
    exit 1
fi
if is_true "$RUN_LCB" && [[ ! -f "${LCB_DATA_ROOT}/test.jsonl" ]]; then
    echo "Missing LiveCodeBench dataset: ${LCB_DATA_ROOT}/test.jsonl" >&2
    exit 1
fi

MODEL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}/${MODEL_NAME}"
output_subdirs=()
is_true "$RUN_HUMANEVAL" && output_subdirs+=(humaneval)
is_true "$RUN_MBPP" && output_subdirs+=(mbpp)
is_true "$RUN_LCB" && output_subdirs+=(lcb)
mkdir -p "$REPO_ROOT/logs/code_eval" "$MODEL_OUTPUT_ROOT"
for subdir in "${output_subdirs[@]}"; do
    mkdir -p "${MODEL_OUTPUT_ROOT}/${subdir}"
done

echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "SAMPLES_ID=${SAMPLES_ID}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "N_SAMPLES=${N_SAMPLES} TEMPERATURE=${TEMPERATURE} TOP_P=${TOP_P} TP=${TP}"
echo "CODE_BENCHMARK_ROOT=${CODE_BENCHMARK_ROOT}"
echo "HUMANEVAL_OVERRIDE_PATH=${HUMANEVAL_OVERRIDE_PATH}"
echo "MBPP_OVERRIDE_PATH=${MBPP_OVERRIDE_PATH}"
echo "LCB_DATA_ROOT=${LCB_DATA_ROOT}"
echo "RUN_HUMANEVAL=${RUN_HUMANEVAL} RUN_MBPP=${RUN_MBPP} RUN_LCB=${RUN_LCB}"

run_evalplus_one() {
    local dataset="$1"
    local devices="$2"
    local tp="$3"

    local samples_file="${MODEL_OUTPUT_ROOT}/${dataset}/${SAMPLES_ID}.jsonl"
    local result_file="${MODEL_OUTPUT_ROOT}/${dataset}/eval_results.json"

    echo "[evalplus] dataset=${dataset} devices=${devices} tp=${tp}"

    if [[ "$GREEDY" == "1" || "$GREEDY" == "true" || "$GREEDY" == "True" ]]; then
        CUDA_VISIBLE_DEVICES="$devices" python3 coding/evalplus/evalplus/codegen.py \
            --model "$MODEL_PATH" \
            --dataset "$dataset" \
            --backend vllm \
            --trust_remote_code \
            --tp "$tp" \
            --root "$MODEL_OUTPUT_ROOT" \
            --identifier "$SAMPLES_ID" \
            --max-new-tokens "$MAX_TOKENS" \
            --greedy
    else
        CUDA_VISIBLE_DEVICES="$devices" python3 coding/evalplus/evalplus/codegen.py \
            --model "$MODEL_PATH" \
            --dataset "$dataset" \
            --backend vllm \
            --trust_remote_code \
            --tp "$tp" \
            --root "$MODEL_OUTPUT_ROOT" \
            --identifier "$SAMPLES_ID" \
            --temperature "$TEMPERATURE" \
            --top-p "$TOP_P" \
            --max-new-tokens "$MAX_TOKENS" \
            --n-samples "$N_SAMPLES"
    fi

    if [[ ! -f "$samples_file" ]]; then
        # Legacy layout: evalplus_results/{dataset}/{full_model_path}_vllm_temp_*.jsonl
        samples_file=$(find "${EVAL_OUTPUT_ROOT}/evalplus_results/${dataset}" -name "*${MODEL_ID}_vllm_temp_*.jsonl" ! -name "*.raw.jsonl" -type f 2>/dev/null | head -n 1)
    fi
    if [[ -z "${samples_file:-}" || ! -f "$samples_file" ]]; then
        echo "[evalplus] samples file not found for dataset=${dataset}" >&2
        return 1
    fi

    python3 -m evalplus.evaluate \
        --dataset "$dataset" \
        --samples "$samples_file" \
        --output_file "$result_file" \
        --min-time-limit 10.0 \
        --gt-time-limit-factor 8.0

    echo "[evalplus] dataset=${dataset} samples=${samples_file}"
    echo "[evalplus] dataset=${dataset} results=${result_file}"
}

link_lcb_benchmark_data() {
    local lcb_lite_dir="${LCB_ROOT}/code_generation_lite"
    if [[ ! -d "$lcb_lite_dir" ]]; then
        echo "LiveCodeBench code_generation_lite dir not found: ${lcb_lite_dir}" >&2
        return 1
    fi
    for filename in test.jsonl test2.jsonl test3.jsonl test4.jsonl test5.jsonl test6.jsonl; do
        if [[ -f "${LCB_DATA_ROOT}/${filename}" ]]; then
            ln -sfn "${LCB_DATA_ROOT}/${filename}" "${lcb_lite_dir}/${filename}"
        fi
    done
}

run_lcb() {
    if [[ ! -f "${LCB_ROOT}/lcb_runner/runner/main.py" ]]; then
        echo "LiveCodeBench runner not found at ${LCB_ROOT}" >&2
        echo "Clone G-OPD code_eval/LiveCodeBench or set LCB_ROOT to a valid checkout." >&2
        return 1
    fi
    if ! link_lcb_benchmark_data; then
        return 1
    fi

    LCB_SOURCE_DIR="${LCB_ROOT}/lcb_outputs/${MODEL_NAME}"
    echo "[lcb] model=${LCB_MODEL} local_model_path=${MODEL_PATH} output_dir=${MODEL_NAME}"
    local lcb_status=0
    local old_pwd="$PWD"
    (
        cd "$LCB_ROOT"
        export LCB_LOCAL_MODEL_PATH="$MODEL_PATH"
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m lcb_runner.runner.main \
            --model "$LCB_MODEL" \
            --local_model_path "$MODEL_PATH" \
            --trust_remote_code \
            --scenario codegeneration \
            --release_version "$LCB_RELEASE_VERSION" \
            --tensor_parallel_size 8 \
            --use_cache \
            --n "$N_SAMPLES" \
            --temperature "$TEMPERATURE" \
            --max_tokens "$MAX_TOKENS" \
            --custom_output_save_name "$MODEL_NAME" \
            --top_p "$TOP_P" \
            --timeout 60 \
            --evaluate --continue_existing --continue_existing_with_eval
    ) || lcb_status=$?
    cd "$old_pwd"

    if [[ "$lcb_status" -ne 0 ]]; then
        echo "[lcb] failed with exit code ${lcb_status}" >&2
        return "$lcb_status"
    fi

    local lcb_source="${LCB_ROOT}/lcb_outputs/${MODEL_NAME}"
    if [[ -d "$lcb_source" ]]; then
        rsync -a "$lcb_source"/ "${MODEL_OUTPUT_ROOT}/lcb"/
        echo "[lcb] synced outputs to ${MODEL_OUTPUT_ROOT}/lcb"
    else
        echo "[lcb] no outputs found under ${lcb_source}" >&2
        return 1
    fi
}

pids=()
failed=0
LCB_SOURCE_DIR=""

# Run LCB first while all 8 GPUs are free (needs tp=8).
if is_true "$RUN_LCB"; then
    if ! run_lcb; then
        failed=1
    fi
fi

if is_true "$RUN_HUMANEVAL" && is_true "$RUN_MBPP"; then
    run_evalplus_one humaneval 0,1,2,3 "$TP" &
    pids+=($!)
    run_evalplus_one mbpp 4,5,6,7 "$TP" &
    pids+=($!)
elif is_true "$RUN_HUMANEVAL"; then
    run_evalplus_one humaneval 0,1 "$TP" &
    pids+=($!)
elif is_true "$RUN_MBPP"; then
    run_evalplus_one mbpp 0,1 "$TP" &
    pids+=($!)
fi

if ((${#pids[@]} > 0)); then
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
fi

python3 - <<'PY' "$MODEL_OUTPUT_ROOT" "$MODEL_NAME" "$N_SAMPLES" "$TEMPERATURE" "$TOP_P" "$MAX_TOKENS" "$RUN_HUMANEVAL" "$RUN_MBPP" "$RUN_LCB" "$LCB_SOURCE_DIR"
import json
import sys
import time
from datetime import datetime
from pathlib import Path

output_root = Path(sys.argv[1])
model_name = sys.argv[2]
n_samples = int(sys.argv[3])
temperature = float(sys.argv[4])
top_p = float(sys.argv[5])
max_tokens = int(sys.argv[6])
run_humaneval = sys.argv[7].lower() in ("1", "true")
run_mbpp = sys.argv[8].lower() in ("1", "true")
run_lcb = sys.argv[9].lower() in ("1", "true")
lcb_source_dir = Path(sys.argv[10]) if len(sys.argv) > 10 and sys.argv[10] else None

def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod

def lcb_pass_at_1(eval_all_path):
    for attempt in range(10):
        try:
            if not eval_all_path.is_file():
                raise FileNotFoundError(eval_all_path)
            results = json.loads(eval_all_path.read_text())
            break
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            if attempt == 9:
                raise
            time.sleep(1)
    if not results:
        return None
    scores = []
    for item in results:
        graded = item.get("graded_list") or []
        if not graded:
            continue
        scores.append(estimate_pass_at_k(len(graded), sum(bool(g) for g in graded), 1))
    if not scores:
        return None
    return sum(scores) / len(scores)

def find_lcb_eval_all(*dirs):
    patterns = [
        f"Scenario.codegeneration_{n_samples}_{temperature}_eval_all.json",
        f"codegeneration_{n_samples}_{temperature}_eval_all.json",
        "*codegeneration*_eval_all.json",
    ]
    for lcb_dir in dirs:
        if lcb_dir is None or not Path(lcb_dir).is_dir():
            continue
        lcb_dir = Path(lcb_dir)
        for pattern in patterns:
            if "*" in pattern:
                matches = sorted(lcb_dir.glob(pattern))
                if matches:
                    return matches[0]
            else:
                candidate = lcb_dir / pattern
                if candidate.is_file():
                    return candidate
    return None

print("\n=== Code Eval Summary ===")
print(f"model={model_name}  (n={n_samples}, temp={temperature}, top_p={top_p}, max_tokens={max_tokens})")
print(f"output_root={output_root}")
print(f"ran: humaneval={run_humaneval} mbpp={run_mbpp} lcb={run_lcb}")

summary = {
    "model": model_name,
    "n_samples": n_samples,
    "temperature": temperature,
    "top_p": top_p,
    "max_tokens": max_tokens,
    "output_root": str(output_root),
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "benches_run": {
        "humaneval": run_humaneval,
        "mbpp": run_mbpp,
        "lcb": run_lcb,
    },
    "metrics": {},
}

evalplus_rows = []
for dataset, label in (("humaneval", "HumanEval+"), ("mbpp", "MBPP+")):
    if dataset == "humaneval" and not run_humaneval:
        continue
    if dataset == "mbpp" and not run_mbpp:
        continue
    result_file = output_root / dataset / "eval_results.json"
    if not result_file.is_file():
        if (dataset == "humaneval" and run_humaneval) or (dataset == "mbpp" and run_mbpp):
            print(f"{label}: missing {result_file}")
        continue
    data = json.loads(result_file.read_text())
    plus = data.get("pass_at_k", {}).get("plus", {})
    base = data.get("pass_at_k", {}).get("base", {})
    p1 = plus.get("pass@1")
    if p1 is None:
        p1 = base.get("pass@1")
    if p1 is not None:
        pct = p1 * 100.0
        evalplus_rows.append(pct)
        summary["metrics"][dataset] = {"pass@1": p1, "pass@1_pct": pct}
        print(f"{label}: pass@1 = {pct:.2f}%")
    else:
        print(f"{label}: pass@1 not found in {result_file}")

if evalplus_rows:
    avg = sum(evalplus_rows) / len(evalplus_rows)
    print(f"EvalPlus Avg (pass@1, {len(evalplus_rows)} bench): {avg:.2f}%")

lcb_eval = find_lcb_eval_all(output_root / "lcb", lcb_source_dir)
if lcb_eval is not None:
    try:
        lcb_p1 = lcb_pass_at_1(lcb_eval)
        if lcb_p1 is not None:
            lcb_pct = lcb_p1 * 100.0
            summary["metrics"]["lcb"] = {
                "pass@1": lcb_p1,
                "pass@1_pct": lcb_pct,
                "eval_all": str(lcb_eval),
            }
            print(f"LCB v6: pass@1 = {lcb_pct:.2f}%  ({lcb_eval.name})")
        else:
            print(f"LCB: could not parse pass@1 from {lcb_eval}")
    except Exception as exc:
        print(f"LCB: failed to read {lcb_eval}: {exc}")
elif run_lcb:
    print(f"LCB: missing eval_all under {output_root / 'lcb'}")

summary_path = output_root / "summary.json"
if summary_path.is_file():
    try:
        prev = json.loads(summary_path.read_text())
        for key, value in prev.get("metrics", {}).items():
            summary["metrics"].setdefault(key, value)
    except json.JSONDecodeError:
        pass

evalplus_rows = [
    summary["metrics"][key]["pass@1_pct"]
    for key in ("humaneval", "mbpp")
    if key in summary["metrics"] and summary["metrics"][key].get("pass@1_pct") is not None
]
if evalplus_rows:
    summary["metrics"]["evalplus_avg_pass@1_pct"] = sum(evalplus_rows) / len(evalplus_rows)

code_avg_parts = []
for key in ("humaneval", "mbpp", "lcb"):
    metric = summary["metrics"].get(key)
    if metric and metric.get("pass@1_pct") is not None:
        code_avg_parts.append(metric["pass@1_pct"])
if len(code_avg_parts) == 3:
    code_avg = sum(code_avg_parts) / 3
    summary["metrics"]["code_avg_pass@1_pct"] = code_avg
    print(f"Code Avg (HE+ / MBPP+ / LCB): {code_avg:.2f}%")

summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"summary saved to {summary_path}")
PY

if [[ "$failed" -ne 0 ]]; then
    echo "One or more code eval subprocesses failed." >&2
    exit 1
fi
echo "Model ${MODEL_NAME} code eval done!"
