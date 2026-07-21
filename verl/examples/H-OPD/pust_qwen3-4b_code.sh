#!/bin/bash
#SBATCH --job-name=e04-w2s_code
#SBATCH --output=logs/w2s_code/slurm_code_%j.out
#SBATCH --error=logs/w2s_code/slurm_code_%j.err
#SBATCH --chdir=/mnt/petrelfs/wurong/workspace/RM-OPD
#SBATCH --account=research
#SBATCH --partition=DataFrontier_Explore
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1


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


export WANDB_API_KEY="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"
export WANDB_MODE=online
export USED_MODEL="no_api"

student_model_name="Qwen3-8B"
teacher_model_name="Qwen3-4B-Non-Thinking-RL-Code-Step300" # code proxy expert
teacher_base_model_name="Qwen3-4B"                         # proxy base
student_tag="Qwen3-8B"
teacher_tag="4B_CodeRL"
ability=Code

student_rollout_n=1

# sbatch copies the script to /var/spool/slurmd/...; BASH_SOURCE is unreliable there.
# SLURM_SUBMIT_DIR is the directory where sbatch was invoked (repo root).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
VERL_ROOT="${REPO_ROOT}/verl"
cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

dataset_root="${REPO_ROOT}/data/opd/training/g_opd"
train_files=${dataset_root}/Eurus/code_train.parquet
test_files=${dataset_root}/Eurus/code_validation.parquet
code_reward_path=${VERL_ROOT}/verl/utils/reward_score/code_eval_reward/__init__.py

student_model_path="/mnt/phwfile/datafrontier/public_models/${student_model_name}"
teacher_model_path="/mnt/phwfile/datafrontier/public_models/${teacher_model_name}"
teacher_base_model_path="/mnt/phwfile/datafrontier/public_models/${teacher_base_model_name}"

today=$(date +%m%d_%H%M)

# grpo configs
use_grpo=false
grpo_lr_scale=1.0

# opd configs
use_opd=true
opd_lr_scale=1.0
opd_top_k=0
lambda_vals=1.0

# actor param offload: set false when GPU memory allows (skips actor CPU<->GPU each step)
actor_param_offload=true

# include student primary-base in parallel OPD prep (teacher proxies always parallel)
opd_parallel_student_base=false

# update mode config
update_mode=both   # in [alt, both, warmup]
    # alt update configs
opd_steps=10
grpo_steps=50
    # warmup update configs
warmup_steps=10

# validation config
val_n=4
val_metric_group=code_avg
val_metric_sources=taco,apps,codecontests,codeforces
val_metric_sources_hydra="[${val_metric_sources}]"
code_eval_workers=32

# base learning rate
lr=1e-6

n_node=1
n_gpu=8


project_name=MT_weak2strong@${val_n}

# resume_path="/mnt/phwfile/datafrontier/fudaocheng/checkpoints/trained/TRA_strong2weak@16/Qwen3-1.7B-T4B-_GB1024_OB0_0606_0711/"
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

output_model_name="${student_tag}-T${teacher_tag}_Lam${lambda_vals}-n${student_rollout_n}"

resume_args=()
if [[ -n "$resume_path" ]]; then
    output_path="$resume_path"
    experiment_name="$(basename "$output_path")"
    resume_args=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path="$resume_path"
    )
else
    experiment_name="Code_${output_model_name}_G${use_grpo}_O${use_opd}_${today}"
    output_path="/mnt/phwfile/datafrontier/fudaocheng/checkpoints/trained/${project_name}/${experiment_name}"
fi

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES


# Avoid DataLoader workers colliding with ProcessPoolExecutor code reward.
# export TMPDIR="${SLURM_TMPDIR:-/tmp/ray_code_reward_${USER}_${SLURM_JOB_ID:-local}}"
# mkdir -p "$TMPDIR"

unset RAY_ADDRESS
unset RAY_NAMESPACE
unset RAY_DASHBOARD_ADDRESS
unset RAY_RUNTIME_ENV
unset RAY_JOB_ID
unset RAY_HEAD_IP
unset RAY_PORT

export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}
mkdir -p "${RAY_TMPDIR}"

python3 -m verl.trainer.main_ppo \
    +algorithm.train_mode=heterogeneous_distill \
    +algorithm.hetero_distill.student_rollout_n=$student_rollout_n \
    +algorithm.hetero_distill.use_grpo=$use_grpo \
    +algorithm.hetero_distill.use_opd=$use_opd \
    +algorithm.hetero_distill.update_mode=$update_mode \
    +algorithm.hetero_distill.opd_steps=$opd_steps \
    +algorithm.hetero_distill.grpo_steps=$grpo_steps \
    +algorithm.hetero_distill.warmup_steps=$warmup_steps \
    +algorithm.hetero_distill.opd_top_k=$opd_top_k \
    +algorithm.hetero_distill.opd_parallel_student_base=$opd_parallel_student_base \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=3412 \
    data.return_raw_chat=True \
    data.dataloader_num_workers=0 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    +actor_rollout_ref.actor.grpo_lr_scale=$grpo_lr_scale \
    +actor_rollout_ref.actor.opd_lr_scale=$opd_lr_scale \
    actor_rollout_ref.actor.policy_loss.lambda_vals=$lambda_vals \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward_model.reward_manager=batch \
    custom_reward_function.path=$code_reward_path \
    custom_reward_function.name=reward_func_batched \
    +custom_reward_function.reward_kwargs.code_eval_workers=$code_eval_workers \
    val_custom_reward_function.path=$code_reward_path \
    val_custom_reward_function.name=reward_func_batched \
    val_custom_reward_function.reward_kwargs.code_eval_workers=$code_eval_workers \
    trainer.critic_warmup=0 \
    trainer.val_before_train=false \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=0 \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=$n_node \
    trainer.max_actor_ckpt_to_keep=1 \
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
