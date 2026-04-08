set -x

dataset_name=MathTestTotal
model_name=Qwen3-8B
n_gpu=1

data_path=/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/$dataset_name/test.parquet 
save_path="/mnt/petrelfs/fudaocheng/experiments_output/G-OPD/${model_name}_${dataset_name}.parquet"  
model_path=/mnt/petrelfs/fudaocheng/checkpoints/huggingface/$model_name


unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$n_gpu \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=16384 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8
