set -x
export PYTHONUNBUFFERED=1


export WANDB_API_KEY="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"
export WANDB_MODE=online
export USED_MODEL="no_api"


aime24_test_path=/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/AIME2024/test.parquet
aime25_test_path=/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/AIME2025/test.parquet

test_files="['$aime24_test_path', '$aime25_test_path']"

train_files=/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
model_path=/mnt/petrelfs/fudaocheng/checkpoints/huggingface/Qwen3-4B


output_model_name=Qwen3-4B-math
method=GRPO
output_path="/mnt/petrelfs/fudaocheng/checkpoints/trained/${output_model_name}_${method}"

batch_size=128
n_gpu=8
lr=1e-6

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# GRPO
python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=$batch_size \
        data.max_prompt_length=2048 \
        data.max_response_length=10240 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="$model_path" \
        actor_rollout_ref.actor.optim.lr=$lr \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=32 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.critic_warmup=0 \
        trainer.val_before_train=False \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name="${output_model_name}_grpo_bsz${batch_size}_n${n_gpu}_lr${lr}" \
        trainer.n_gpus_per_node=$n_gpu \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.default_local_dir="$output_path" \
        trainer.test_freq=10 \
        trainer.total_epochs=3 $@

