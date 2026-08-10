#!/bin/bash


set -x
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

export RAY_memory_usage_threshold=0.99


student_model_name="Qwen3-8B"
teacher_model_name="Qwen3-4B-Non-Thinking-RL-Math-Step1200" # proxy expert
teacher_base_model_name="Qwen3-4B"           # proxy base
student_tag="Qwen3-8B"
teacher_tag="4B_Math_RL_Step500"
ability=Math

# sbatch copies the script to /var/spool/slurmd/...; BASH_SOURCE is unreliable there.
# SLURM_SUBMIT_DIR is the directory where sbatch was invoked (repo root).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
VERL_ROOT="${REPO_ROOT}/verl"

# Auto-start math verify HTTP server (Ray workers call it remotely; do not run math_verify in-process).
source "${REPO_ROOT}/math_eval_server/start_math_verify_server.sh"
start_math_verify_server "${REPO_ROOT}" || exit 1
trap stop_math_verify_server EXIT

cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export MATH_VERIFY_SERVER_URL

test_files=./data/math/test.parquet
train_files=./data/math/train.parquet

student_model_path="./models/${student_model_name}"
teacher_model_path="./models/${teacher_model_name}"
teacher_base_model_path="./models/${teacher_base_model_name}"

today=$(date +%m%d_%H%M)

opd_top_k=0
lambda_vals=1.0    # lambda value for the lambda-based reward function
student_rollout_n=1

# actor param offload: set false when GPU memory allows (skips actor CPU<->GPU each step)
actor_param_offload=true

# include student primary-base in parallel OPD prep (teacher proxies always parallel)
opd_parallel_student_base=false


# validation config
val_n=16
val_metric_group=math_avg
val_metric_sources=AIME2024,AIME2025,AIME2026,SMT2025,CMIMC2025,HMMT2025FEB,HMMT2025NOV,HMMT2026FEB
val_metric_sources_hydra="[${val_metric_sources}]"

# base learning rate
lr=1e-6

n_node=1
n_gpu=8


project_name=PUST_Math@${val_n}

resume_path=""

extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume_path)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --resume_path requires a path argument" >&2
                exit 1
            fi
            resume_path="$2"
            shift 2
            ;;
        --resume_path=*)
            resume_path="${1#--resume_path=}"
            shift
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done
set -- "${extra_args[@]}"

batch_size_to_bool() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${name}_batch_size must be a non-negative integer, got: ${value}" >&2
        exit 1
    fi
    if (( 10#$value > 0 )); then
        echo true
    else
        echo false
    fi
}

output_model_name="${student_tag}-T${teacher_tag}_Lam${lambda_vals}"

resume_args=()
if [[ -n "$resume_path" ]]; then
    output_path="$resume_path"
    experiment_name="$(basename "$output_path")"
    resume_args=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path="$resume_path"
    )
else
    experiment_name="${output_model_name}_${today}"
    output_path="./models/saved_models/${project_name}/${experiment_name}"
fi

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

unset RAY_ADDRESS
unset RAY_NAMESPACE
unset RAY_DASHBOARD_ADDRESS
unset RAY_RUNTIME_ENV
unset RAY_JOB_ID
unset RAY_HEAD_IP
unset RAY_PORT


python3 -m verl.trainer.main_ppo \
    +algorithm.train_mode=heterogeneous_distill \
    +algorithm.hetero_distill.student_rollout_n=$student_rollout_n \
    +algorithm.hetero_distill.opd_top_k=$opd_top_k \
    +algorithm.hetero_distill.opd_parallel_student_base=$opd_parallel_student_base \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=3412 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path=$student_model_path \
    +actor_rollout_ref.model.base_model_path=$student_model_path \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    +actor_rollout_ref.ref.model.base_model_path=$teacher_base_model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=$actor_param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$val_n \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    +actor_rollout_ref.actor.grpo_lr_scale=$grpo_lr_scale \
    +actor_rollout_ref.actor.opd_lr_scale=$opd_lr_scale \
    actor_rollout_ref.actor.policy_loss.lambda_vals=$lambda_vals \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=false \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=0 \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=$n_node \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.save_freq=10 \
    +trainer.val_aggregate_group=$val_metric_group \
    +trainer.val_aggregate_sources="${val_metric_sources_hydra}" \
    +trainer.best_metric_name="val-core/${val_metric_group}/reward/mean@${val_n}" \
    +trainer.best_metric_mode="max" \
    trainer.default_local_dir="$output_path" \
    "${resume_args[@]}" \
    trainer.test_freq=10 \
    trainer.total_epochs=3 "$@"
