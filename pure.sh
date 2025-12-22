#!/bin/bash

# 参考 submit_training.sh 的单机训练脚本示例
# 用法：根据需要修改下面的变量，然后执行：
#   bash train_run.sh

set -euo pipefail

######################## 基础配置（请按需修改） ########################
MODEL_NAME="Qwen2.5-Math-1.5B"
MODEL_PATH="/mnt/shared-storage-user/mineru4s/dingruiyi/Qwen2.5-Math-1.5B"
DATASET_NAME="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/math500split_train.parquet"

TRAIN_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/math500split_train.parquet"
VAL_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/math500split_val.parquet"

ALGORITHM="pure"       
ROLLOUT_N=8               # rollout 数
GPU_BATCH_SIZE=16         # 每张卡 batch
NUM_GPUS=8                # 单机 GPU 数
NUM_NODES=1               # 节点数
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
SAVE_FREQ=100             # checkpoint 保存间隔

# 训练元信息
PROJECT_NAME="verl_srpo_${DATASET_NAME}"
MODEL_SHORT=$(echo "${MODEL_NAME}" | sed 's/Qwen2.5-Math-//' | sed 's/Qwen2.5-//' | sed 's/-Instruct//' | tr '[:upper:]' '[:lower:]')
EXPERIMENT_NAME="${DATASET_NAME}-${MODEL_SHORT}-${ALGORITHM}-r${ROLLOUT_N}"
CHECKPOINT_DIR="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# 派生配置
TOTAL_BATCH_SIZE=$((GPU_BATCH_SIZE * NUM_GPUS * NUM_NODES))

######################## 环境设置 ########################
source /mnt/shared-storage-user/mineru4s/dingruiyi/anaconda/bin/activate
conda activate verl

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
unset RAY_ADDRESS
unset ROCR_VISIBLE_DEVICES

######################## 运行信息提示 ########################
echo "====== 训练配置 ======"
echo "模型:          ${MODEL_NAME}"
echo "模型路径:      ${MODEL_PATH}"
echo "数据集:        ${DATASET_NAME}"
echo "训练数据:      ${TRAIN_DATA}"
echo "验证数据:      ${VAL_DATA}"
echo "算法:          ${ALGORITHM}"
echo "rollout 数:    ${ROLLOUT_N}"
echo "GPU/节点:      ${NUM_GPUS} / ${NUM_NODES}"
echo "每卡 batch:    ${GPU_BATCH_SIZE}"
echo "总 batch:      ${TOTAL_BATCH_SIZE}"
echo "最大响应长:    ${MAX_RESPONSE_LENGTH}"
echo "实验名:        ${EXPERIMENT_NAME}"
echo "ckpt 目录:     ${CHECKPOINT_DIR}"
echo "======================"

######################## 启动训练 ########################
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${TOTAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TOTAL_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='${PROJECT_NAME}' \
    trainer.experiment_name='${EXPERIMENT_NAME}' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NUM_NODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=2000 \
    trainer.total_epochs=10 \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.default_local_dir='${CHECKPOINT_DIR}' \
    reward_model.reward_manager=${ALGORITHM}

