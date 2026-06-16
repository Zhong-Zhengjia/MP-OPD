set -x
export PYTHONUNBUFFERED=1


export WANDB_API_KEY="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"
export WANDB_MODE=online
export USED_MODEL="no_api"

student_model_name=DeepSeek-R1-Distill-Qwen-1.5B
teacher_model_name=Skywork-OR1-Math-7B
student_model_abb=DS15
teacher_model_abb=SK7
ability=Math

test_files=/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/MathTestTotal/test.parquet
train_files=/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/train_80_percent.parquet

student_model_path="/mnt/phwfile/datafrontier/public_models/${student_model_name}"
teacher_model_path="/mnt/phwfile/datafrontier/public_models/${teacher_model_name}"

today=$(date +%m%d_%H%M)

# grpo configs
use_grpo=false
grpo_lr_scale=1.0

# opd configs
use_opd=true
opd_lr_scale=1.0 
opd_top_k=100

# update mode config
update_mode=both   # in [alt, both, warmup]
    # alt update configs
opd_steps=10
grpo_steps=50
    # warmup update configs
warmup_steps=20

# validation config
val_n=16

# base learning rate
lr=1e-6

n_node=1
n_gpu=8


project_name=MathRL_strong2weak@${val_n}

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

output_model_name="${student_model_abb}-T${teacher_model_abb}-${update_mode}${warmup_steps}"

resume_args=()
if [[ -n "$resume_path" ]]; then
    output_path="$resume_path"
    experiment_name="$(basename "$output_path")"
    resume_args=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path="$resume_path"
    )
else
    experiment_name="${output_model_name}_G${use_grpo}_O${use_opd}_${today}"
    output_path="/mnt/phwfile/datafrontier/fudaocheng/checkpoints/trained/${project_name}/${experiment_name}"
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

export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}
mkdir -p "${RAY_TMPDIR}"

python3 -m verl.trainer.main_ppo \
    +algorithm.train_mode=heterogeneous_distill \
    +algorithm.hetero_distill.student_rollout_n=8 \
    +algorithm.hetero_distill.use_grpo=$use_grpo \
    +algorithm.hetero_distill.use_opd=$use_opd \
    +algorithm.hetero_distill.update_mode=$update_mode \
    +algorithm.hetero_distill.opd_steps=$opd_steps \
    +algorithm.hetero_distill.grpo_steps=$grpo_steps \
    +algorithm.hetero_distill.warmup_steps=$warmup_steps \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=10240 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=3412 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.model.path=$student_model_path \
    +actor_rollout_ref.model.base_model_path=$student_model_path \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=11264 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.data_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=11264 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$val_n \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    +actor_rollout_ref.actor.grpo_lr_scale=$grpo_lr_scale \
    +actor_rollout_ref.actor.opd_lr_scale=$opd_lr_scale \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=true \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=0 \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=$n_node \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.save_freq=10 \
    +trainer.best_metric_name="val-core/DeepMath-103K/reward/mean@${val_n}" \
    +trainer.best_metric_mode="max" \
    trainer.default_local_dir="$output_path" \
    "${resume_args[@]}" \
    trainer.test_freq=10 \
    trainer.total_epochs=10 "$@"
