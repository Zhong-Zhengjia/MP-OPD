set -x
export PYTHONUNBUFFERED=1


export WANDB_API_KEY="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"
export WANDB_MODE=online
export USED_MODEL="no_api"


test_files=/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/val_union_mini_1000.parquet

train_files=/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/train_union_passed_70.parquet

student_model_subfix=4B
teacher_model_subfix=1.7B
student_model_name="Qwen3-${student_model_subfix}"
teacher_model_name="Qwen3-${teacher_model_subfix}"
ability=Math

student_model_path="/mnt/petrelfs/fudaocheng/checkpoints/huggingface/Qwen3-${student_model_subfix}"
teacher_model_path="/mnt/petrelfs/fudaocheng/checkpoints/huggingface/Qwen3-${teacher_model_subfix}"

today=$(date +%Y%m%d)
output_model_name="Qwen3-${student_model_subfix}-T${teacher_model_subfix}-${ability}-default-params-actor-loss-actor-clip"
method=HOPD
output_path="/mnt/petrelfs/fudaocheng/checkpoints/trained/${output_model_name}_${method}_${today}"

n_gpu=8
lr=1e-6

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
    +algorithm.hetero_distill.student_rollout_n=4 \
    +algorithm.hetero_distill.teacher_rollout_n=4 \
    +algorithm.hetero_distill.use_sdft=true \
    +algorithm.hetero_distill.use_icl_opd=true \
    +algorithm.hetero_distill.sdft_weight=1.0 \
    +algorithm.hetero_distill.icl_opd_weight=1.0 \
    +algorithm.hetero_distill.sample_demo_strategy=random \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=144 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path=$student_model_path \
    +actor_rollout_ref.model.base_model_path=$student_model_path \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=false \
    actor_rollout_ref.actor.policy_loss.multi_teacher_distill=false \
    teacher_rollout.model.path=$teacher_model_path \
    teacher_rollout.model.use_remove_padding=true \
    teacher_rollout.actor.optim.lr_warmup_steps_ratio=0.0 \
    teacher_rollout.actor.ppo_mini_batch_size=8 \
    teacher_rollout.actor.ppo_micro_batch_size_per_gpu=1 \
    teacher_rollout.actor.ppo_max_token_len_per_gpu=9216 \
    teacher_rollout.actor.fsdp_config.param_offload=false \
    teacher_rollout.actor.fsdp_config.optimizer_offload=false \
    teacher_rollout.model.enable_gradient_checkpointing=true \
    teacher_rollout.rollout.log_prob_micro_batch_size_per_gpu=4 \
    teacher_rollout.rollout.tensor_model_parallel_size=2 \
    teacher_rollout.rollout.name=vllm \
    teacher_rollout.rollout.mode=sync \
    teacher_rollout.rollout.free_cache_engine=false \
    teacher_rollout.rollout.gpu_memory_utilization=0.8 \
    teacher_rollout.rollout.n=1 \
    teacher_rollout.rollout.max_num_batched_tokens=9216 \
    teacher_rollout.rollout.temperature=1.0 \
    teacher_rollout.rollout.top_p=1.0 \
    teacher_rollout.rollout.val_kwargs.do_sample=True \
    teacher_rollout.rollout.val_kwargs.temperature=1.0 \
    teacher_rollout.rollout.val_kwargs.top_p=1.0 \
    teacher_rollout.rollout.val_kwargs.n=1 \
    teacher_rollout.rollout.calculate_log_probs=false \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_is_threshold=null \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=false \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
    trainer.project_name='heterogeneous-distillation' \
    trainer.experiment_name="${output_model_name}_${method}_${today}" \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.save_freq=10 \
    +trainer.best_metric_name="val-core/DeepMath-103K/reward/mean@1" \
    +trainer.best_metric_mode="max" \
    trainer.default_local_dir=$output_path \
    trainer.test_freq=10 \
    trainer.total_epochs=2 $@