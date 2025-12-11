#!/bin/bash

# 统一的训练任务提交脚本
# 用法: ./submit_training.sh <model_name> <dataset_name> <num_gpus> [algorithm] [gpu_batch_size] [rollout_n] [model_path]

set -e

# 检查参数
if [ $# -lt 3 ]; then
    echo "用法: $0 <model_name> <dataset_name> <num_gpus> [algorithm] [gpu_batch_size] [rollout_n] [max_response_length] [num_nodes] [model_path]"
    echo "示例: $0 Qwen2.5-Math-1.5B math17 8 grpo 16 8 2048 1"
    echo ""
    echo "参数说明:"
    echo "  model_name: 模型名称或简称 (如: qwen2-1.5b, qwen3-8b, Qwen2.5-Math-1.5B)"
    echo "               支持使用简称，详见 model_mapping.txt"
    echo "  dataset_name: 数据集名称 (如: math500, gsm8k, deepmath等)"
    echo "  num_gpus: GPU数量 (如: 6, 8, 16等)"
    echo "  algorithm: 算法类型 (可选, 默认: grpo, 可选: grpo, srpo2)"
    echo "  gpu_batch_size: 每个GPU的batch size (可选, 默认: 16)"
    echo "  rollout_n: rollout数量 (可选, 默认: 8, 常见值: 8, 16, 32)"
    echo "  max_response_length: 最大响应长度 (可选, 默认: 2048, 常见值: 2048, 32768)"
    echo "  num_nodes: 节点数量 (可选, 默认: 1)"
    echo "  model_path: 模型路径 (可选, 如果使用简称则自动从映射表获取)"
    exit 1
fi

MODEL_NAME_INPUT="$1"
DATASET_NAME="$2"
NUM_GPUS="$3"
ALGORITHM="${4:-grpo}"  # 默认使用grpo
GPU_BATCH_SIZE="${5:-16}"  # 默认每个GPU batch size为16
ROLLOUT_N="${6:-8}"  # 默认rollout数量为8
MAX_RESPONSE_LENGTH="${7:-2048}"  # 默认最大响应长度为2048
NUM_NODES="${8:-1}"  # 默认节点数量为1
MODEL_PATH_INPUT="${9:-}"


# 模型映射表路径
MAPPING_FILE="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/model_mapping.txt"

# 从映射表中查找模型
MODEL_NAME="$MODEL_NAME_INPUT"
MODEL_SHORT="$MODEL_NAME_INPUT"
MODEL_PATH=""

if [ -f "$MAPPING_FILE" ]; then
    # 在映射表中查找匹配的简称或完整名称
    while IFS='|' read -r short_name full_name model_path || [ -n "$short_name" ]; do
        # 跳过注释行和空行
        case "$short_name" in
            \#*) continue ;;
            "") continue ;;
        esac
        
        # 如果输入的模型名称匹配简称或完整名称
        if [ "$MODEL_NAME_INPUT" = "$short_name" ] || [ "$MODEL_NAME_INPUT" = "$full_name" ]; then
            MODEL_NAME="$full_name"
            MODEL_SHORT="$short_name"
            MODEL_PATH="$model_path"
            break
        fi
    done < "$MAPPING_FILE"
fi

# 如果映射表中没找到，使用用户指定的路径或默认路径
if [ -z "$MODEL_PATH" ]; then
    if [ -n "$MODEL_PATH_INPUT" ]; then
        MODEL_PATH="$MODEL_PATH_INPUT"
    else
        MODEL_PATH="/mnt/shared-storage-user/mineru4s/dingruiyi/${MODEL_NAME_INPUT}"
    fi
    # 生成简称用于实验名称
    MODEL_SHORT=$(echo "$MODEL_NAME_INPUT" | sed 's/Qwen2.5-Math-//' | sed 's/Qwen2.5-//' | sed 's/-Instruct//' | tr '[:upper:]' '[:lower:]')
    MODEL_NAME="$MODEL_NAME_INPUT"
fi

# 验证算法类型
if [ "$ALGORITHM" != "grpo" ] && [ "$ALGORITHM" != "srpo3" ] && [ "$ALGORITHM" != "srpo2" ] && [ "$ALGORITHM" != "rflux" ] && [ "$ALGORITHM" != "srpo4" ]; then
    echo "错误: 算法类型必须是 'grpo' 或 'srpo3' 或 'srpo2' 或 'srpo4'，当前为: $ALGORITHM"
    exit 1
fi

# 验证GPU数量
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -le 0 ]; then
    echo "错误: GPU数量必须是正整数，当前为: $NUM_GPUS"
    exit 1
fi

# 验证GPU batch size
if ! [[ "$GPU_BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$GPU_BATCH_SIZE" -le 0 ]; then
    echo "错误: GPU batch size必须是正整数，当前为: $GPU_BATCH_SIZE"
    exit 1
fi

# 验证rollout数量
if ! [[ "$ROLLOUT_N" =~ ^[0-9]+$ ]] || [ "$ROLLOUT_N" -le 0 ]; then
    echo "错误: rollout数量必须是正整数，当前为: $ROLLOUT_N"
    exit 1
fi

# 验证最大响应长度
if ! [[ "$MAX_RESPONSE_LENGTH" =~ ^[0-9]+$ ]] || [ "$MAX_RESPONSE_LENGTH" -le 0 ]; then
    echo "错误: 最大响应长度必须是正整数，当前为: $MAX_RESPONSE_LENGTH"
    exit 1
fi

# 计算总batch size，确保是GPU数量 * GPU batch size的倍数
# 默认使用GPU数量 * GPU batch size作为总batch size
TOTAL_BATCH_SIZE=$((NUM_GPUS * GPU_BATCH_SIZE * NUM_NODES))

# 规范化数据集名称（首字母大写）
DATASET_CAPITALIZED=$(echo "${DATASET_NAME}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')

save_freq=100
# 数据文件路径 - 特殊处理 gsm8k 数据集
if [ "$DATASET_NAME" = "gsm8k" ]; then
    TRAIN_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/gsm8k/gsm8ktrain.parquet"
    VAL_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/gsm8k/gsm8ktest.parquet"
    save_freq=58
else
    # 数据文件路径 - 尝试多种可能的命名格式
    # 优先尝试: train{Name}_data.parquet / val{Name}_data.parquet
    TRAIN_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/train${DATASET_CAPITALIZED}_data.parquet"
    VAL_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/val${DATASET_CAPITALIZED}_data.parquet"
    save_freq=93
    # 如果文件不存在，尝试其他格式
    if [ ! -f "$TRAIN_DATA" ]; then
    # 尝试: train{dataset}_data.parquet (全小写格式，如trainall_data.parquet)
    ALT_TRAIN="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/train${DATASET_NAME}_data.parquet"
    ALT_VAL="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/val${DATASET_NAME}_data.parquet"
    if [ -f "$ALT_TRAIN" ]; then
        TRAIN_DATA="$ALT_TRAIN"
        VAL_DATA="$ALT_VAL"
    else
        # 尝试: {dataset}split_train.parquet / {dataset}split_val.parquet
        ALT_TRAIN="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}split_train.parquet"
        ALT_VAL="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}split_val.parquet"
        if [ -f "$ALT_TRAIN" ]; then
            TRAIN_DATA="$ALT_TRAIN"
            VAL_DATA="$ALT_VAL"
        else
            # 尝试: {dataset}_train.parquet / {dataset}_val.parquet (如math3to5_train.parquet)
            ALT_TRAIN="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}_train.parquet"
            ALT_VAL="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}_val.parquet"
            if [ -f "$ALT_TRAIN" ]; then
                TRAIN_DATA="$ALT_TRAIN"
                VAL_DATA="$ALT_VAL"
            else
                # 尝试: {dataset}.parquet / {dataset}val.parquet
                ALT_TRAIN="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}.parquet"
                ALT_VAL="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}val.parquet"
                if [ -f "$ALT_TRAIN" ]; then
                    TRAIN_DATA="$ALT_TRAIN"
                    VAL_DATA="$ALT_VAL"
                else
                    echo "警告: 未找到训练数据文件，使用默认路径: $TRAIN_DATA"
                    echo "如果路径不正确，请手动修改生成的训练脚本"
                fi
            fi
        fi
    fi
    fi
fi

# 清理任务名，确保符合rjob命名规则: ^[a-zA-Z0-9][-a-zA-Z0-9]{1,61}[a-zA-Z0-9]$
# 规则：必须以字母或数字开头和结尾，只能包含字母、数字和连字符，长度3-63字符
clean_rjob_name() {
    local name="$1"
    # 将所有非字母数字字符替换为连字符
    name=$(echo "$name" | sed 's/[^a-zA-Z0-9]/-/g')
    # 移除开头和结尾的连字符
    name=$(echo "$name" | sed 's/^-\+//; s/-\+$//')
    # 将连续的连字符替换为单个连字符
    name=$(echo "$name" | sed 's/-\+/-/g')
    # 如果为空或太短，添加默认前缀
    if [ -z "$name" ] || [ ${#name} -lt 3 ]; then
        name="task-${name}"
    fi
    # 限制长度在63个字符以内（使用cut命令兼容性更好）
    if [ ${#name} -gt 63 ]; then
        name=$(echo "$name" | cut -c1-63)
        # 确保截断后不以连字符结尾
        name=$(echo "$name" | sed 's/-\+$//')
        # 如果截断后为空或太短，重新生成
        if [ -z "$name" ] || [ ${#name} -lt 3 ]; then
            name="task-$(echo "$1" | sed 's/[^a-zA-Z0-9]//g' | cut -c1-60)"
        fi
    fi
    # 确保以字母或数字开头
    if ! [[ "$name" =~ ^[a-zA-Z0-9] ]]; then
        name="t${name}"
    fi
    # 确保以字母或数字结尾
    if ! [[ "$name" =~ [a-zA-Z0-9]$ ]]; then
        name="${name}0"
    fi
    echo "$name"
}
MAX_PROMPT_LENGTH=2048

# 生成实验名称和路径
# MODEL_SHORT 已经在上面从映射表中获取或生成
# 实验名包含rollout数量，以便训练框架区分不同rollout配置的存储路径
EXPERIMENT_NAME="${DATASET_NAME}-${MODEL_SHORT}-${ALGORITHM}-r${ROLLOUT_N}"
PROJECT_NAME="verl_srpo_${DATASET_NAME}"
# checkpoint目录名包含rollout数量，以便区分不同rollout配置
CHECKPOINT_DIR="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# 生成训练脚本文件名（使用实验名）
TRAIN_SCRIPT="${EXPERIMENT_NAME}.sh"
# rjob任务名包含rollout数量，格式：{dataset}-{model}-{algorithm}-r{rollout_n}
# 清理任务名以确保符合rjob命名规则
RJOB_NAME_RAW="${DATASET_NAME}-${MODEL_SHORT}-${ALGORITHM}-r${ROLLOUT_N}"
RJOB_NAME=$(clean_rjob_name "$RJOB_NAME_RAW")


# 生成训练脚本
cat > "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/$TRAIN_SCRIPT" << EOF
source /mnt/shared-storage-user/mineru4s/dingruiyi/anaconda/bin/activate
conda activate verl

### 先不使用wandb，如有需要可以开启 ###
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
### 单机训练，也先不用ray，多机多卡之后再探索 ###
unset RAY_ADDRESS # 暂时不用多机，另外在训练代码main_ppo.py的开头需要加一行
### ray.init(address="local", ignore_reinit_error=True) ### 表示我们先不用ray

unset ROCR_VISIBLE_DEVICES # Ray 启动的 worker 进程中同时加载了两个 GPU 可见性控制变量，unset其中一个，隐式保留CUDA_VISIBLE_DEVICES即可

### 下面的脚本是官方example当中提供的,我只更改了模型路径 ###
python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    data.train_files=${TRAIN_DATA} \\
    data.val_files=${VAL_DATA} \\
    data.train_batch_size=${TOTAL_BATCH_SIZE} \\
    data.max_prompt_length=${MAX_PROMPT_LENGTH}\\
    data.max_response_length=$MAX_RESPONSE_LENGTH \\
    data.filter_overlong_prompts=True \\
    data.truncation='error' \\
    actor_rollout_ref.model.path=${MODEL_PATH} \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size=${TOTAL_BATCH_SIZE} \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4\\
    actor_rollout_ref.actor.use_kl_loss=True \\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.entropy_coeff=0 \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \\
    actor_rollout_ref.actor.strategy="fsdp2" \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \\
    actor_rollout_ref.rollout.max_num_batched_tokens=65536\\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \\    
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    algorithm.use_kl_in_reward=False \\
    trainer.critic_warmup=0 \\
    trainer.logger=['console','wandb'] \\
    trainer.project_name='${PROJECT_NAME}' \\
    trainer.experiment_name='${EXPERIMENT_NAME}' \\
    trainer.n_gpus_per_node=${NUM_GPUS} \\
    trainer.nnodes=${NUM_NODES} \\
    trainer.save_freq=${save_freq} \\
    trainer.test_freq=2000 \\
    trainer.total_epochs=10 \\
    trainer.max_actor_ckpt_to_keep=10 \\
    trainer.default_local_dir='${CHECKPOINT_DIR}' \\
    reward_model.reward_manager=${ALGORITHM}
EOF

chmod +x "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/$TRAIN_SCRIPT"

echo "=========================================="
echo "训练任务配置"
echo "=========================================="
echo "模型输入:        $MODEL_NAME_INPUT"
echo "模型名称:        $MODEL_NAME"
echo "模型简称:        $MODEL_SHORT"
echo "数据集名称:      $DATASET_NAME"
echo "GPU数量:         $NUM_GPUS"
echo "算法类型:        $ALGORITHM"
echo "每个GPU batch:   $GPU_BATCH_SIZE"
echo "最大响应长度:    $MAX_RESPONSE_LENGTH"
echo "总batch size:    $TOTAL_BATCH_SIZE (${NUM_GPUS} GPUs × ${NUM_NODES} Nodes × ${GPU_BATCH_SIZE} batch size per GPU)"
echo "Rollout数量:     $ROLLOUT_N"
echo "模型路径:        $MODEL_PATH"
echo "训练数据:        $TRAIN_DATA"
echo "验证数据:        $VAL_DATA"
echo "实验名称:        $EXPERIMENT_NAME"
echo "检查点目录:      $CHECKPOINT_DIR"
echo "最大ckpt数量:    2"
echo "保存频率:        $save_freq"
echo "=========================================="
echo ""
echo "已生成训练脚本: $TRAIN_SCRIPT"
echo ""

echo "节点数量:        $NUM_NODES"
echo "=========================================="
echo ""
# 提交rjob任务
echo "正在提交rjob任务: $RJOB_NAME"
rjob delete "$RJOB_NAME" 2>/dev/null || true
rjob submit \
    --name="$RJOB_NAME" \
    --gpu="$NUM_GPUS" \
    --memory=1000000 \
    --cpu=128 \
    --charged-group=mineru4sh_gpu \
    --private-machine=group \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    --image=registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/yehc:torch-2.6.0-57d787c2-0627 \
    --host-network=true \
    -P $NUM_NODES \
    -- bash -c "bash /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/${TRAIN_SCRIPT}"

echo ""
echo "任务已提交！"
echo "训练脚本保存在: $TRAIN_SCRIPT"
echo "可以使用以下命令查看任务状态: rjob status $RJOB_NAME"

