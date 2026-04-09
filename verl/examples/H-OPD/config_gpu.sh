#!/usr/bin/env bash

calc_rollout_tp() {
    local n_gpu=$1
    local tp=$(( n_gpu / 2 ))
    (( tp < 1 )) && tp=1
    echo "$tp"
}

calc_train_batch_size() {
    local n_gpu=$1
    local per_gpu_batch=4
    echo $((n_gpu * per_gpu_batch))
}

calc_ppo_mini_batch_size() {
    local n_gpu=$1
    local base=2
    if [ "$n_gpu" -le 2 ]; then
        echo 2
    else
        echo $((base * n_gpu / 2))
    fi
}


calc_ppo_micro_batch_size_per_gpu() {
    local n_gpu=$1
    echo 1
}

calc_logprob_micro_batch_size_per_gpu() {
    local n_gpu=$1
    echo 2
}

calc_ppo_max_token_len_per_gpu() {
    local n_gpu=$1
    echo 2048
}

setup_verl_train_params() {
    local n_gpu=$1

    if [ -z "$n_gpu" ]; then
        echo "[config_gpu.sh] ERROR: n_gpu is empty" >&2
        return 1
    fi

    if [ "$n_gpu" -lt 1 ]; then
        echo "[config_gpu.sh] ERROR: n_gpu must be >= 1" >&2
        return 1
    fi

    export ROLLOUT_TP
    export TRAIN_BATCH_SIZE
    export PPO_MINI_BATCH_SIZE
    export PPO_MICRO_BATCH_SIZE_PER_GPU
    export LOGPROB_MICRO_BATCH_SIZE_PER_GPU
    export PPO_MAX_TOKEN_LEN_PER_GPU
    export MAX_NUM_BATCHED_TOKENS

    ROLLOUT_TP=$(calc_rollout_tp "$n_gpu")
    TRAIN_BATCH_SIZE=$(calc_train_batch_size "$n_gpu")
    PPO_MINI_BATCH_SIZE=$(calc_ppo_mini_batch_size "$n_gpu")
    PPO_MICRO_BATCH_SIZE_PER_GPU=$(calc_ppo_micro_batch_size_per_gpu "$n_gpu")
    LOGPROB_MICRO_BATCH_SIZE_PER_GPU=$(calc_logprob_micro_batch_size_per_gpu "$n_gpu")
    PPO_MAX_TOKEN_LEN_PER_GPU=$(calc_ppo_max_token_len_per_gpu "$n_gpu")

    if [ "$ROLLOUT_TP" -gt "$n_gpu" ]; then
        echo "[config_gpu.sh] ERROR: ROLLOUT_TP=$ROLLOUT_TP > n_gpu=$n_gpu" >&2
        return 1
    fi

    if [ $((n_gpu % ROLLOUT_TP)) -ne 0 ]; then
        echo "[config_gpu.sh] ERROR: n_gpu=$n_gpu is not divisible by ROLLOUT_TP=$ROLLOUT_TP" >&2
        return 1
    fi

    echo "[config_gpu.sh] n_gpu=$n_gpu"
    echo "[config_gpu.sh] ROLLOUT_TP=$ROLLOUT_TP"
    echo "[config_gpu.sh] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
    echo "[config_gpu.sh] PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE"
    echo "[config_gpu.sh] PPO_MICRO_BATCH_SIZE_PER_GPU=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    echo "[config_gpu.sh] LOGPROB_MICRO_BATCH_SIZE_PER_GPU=$LOGPROB_MICRO_BATCH_SIZE_PER_GPU"
    echo "[config_gpu.sh] PPO_MAX_TOKEN_LEN_PER_GPU=$PPO_MAX_TOKEN_LEN_PER_GPU"
}