#!/bin/bash

set -eux




code_dir="your_code_path"
cd "${code_dir}" || exit 1     # enter code directory

OUTPUT_DIR=""
EXPERIMENT_NAME=""
EXP_DIR="${OUTPUT_DIR}/${EXPERIMENT_NAME}"


while true; do
    timestamp=$(date +%Y%m%d_%H%M%S)
    cluster_log_path="${EXP_DIR}/log_${timestamp}.txt"
    if [ -f "${cluster_log_path}" ]; then
        echo "Log file already exists: ${cluster_log_path}"
        sleep 1
    else
        echo "Creating log file: ${cluster_log_path}"
        break
    fi
done

mkdir -p $EXP_DIR


VISUAL_DATASET_TRAIN='Video-R1-video_sample_ratio_0p2.parquet'
VISUAL_DATASET_TRAIN2='Time_r1_coc.parquet'
VISUAL_DATASET_TRAIN3='CG-Bench.parquet'
VISUAL_DATASET_TRAIN4='rextime.parquet'
REF_MODEL_PATH="Your_Qwen2.5-VL-7B_checkpoint_path"

train_cmd="python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    data.train_files=[${VISUAL_DATASET_TRAIN},${VISUAL_DATASET_TRAIN2},${VISUAL_DATASET_TRAIN3},${VISUAL_DATASET_TRAIN4}] \
    data.val_files=[${VISUAL_DATASET_TRAIN}] \
    data.train_batch_size=32 \
    data.max_prompt_length=8892 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=16384 \
    actor_rollout_ref.rollout.agent.max_turns=2 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    trainer.logger=['tensorboard','rl_logging_board'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=10000 \
    trainer.project_name=rl \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${EXP_DIR} \
    +trainer.tensorboard_dir=${EXP_DIR}/tensorboard \
    +trainer.rl_logging_board_dir=${EXP_DIR}/rl_logging_board \
    trainer.total_epochs=1"

echo '' > "${cluster_log_path}" 2>&1            # clear log file
# Step 5: display code version and other information
{
    echo "Code directory: ${code_dir}"
    echo "Code branch: $(git symbolic-ref --short HEAD)"
    echo "Code version: $(git rev-parse HEAD)"
} >> "${cluster_log_path}" 2>&1


# Step 7: run training
echo "Training command: ${train_cmd}"
tic=$(date +%s.%N)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=${WORLD_SIZE} NODE_RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} OMP_NUM_THREADS=16 ${train_cmd} >> "${cluster_log_path}" 2>&1   # run training

toc=$(date +%s.%N)
n_hour=$(awk -v tic="$tic" -v toc="$toc" 'BEGIN { printf "%.2f\n", (toc - tic) / 3600 }')
echo "Training finished. Time elapsed: ${n_hour} hours."