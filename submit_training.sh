#!/bin/bash

# 统一的训练配置生成脚本（不再直接提交/运行训练），适配 verl1 main_ppo
# 用法: ./submit_training.sh <model_name> <dataset_name> <num_gpus> [algorithm] [gpu_batch_size] [rollout_n] [max_response_length] [num_nodes] [model_path]

set -e

# 参数校验
if [ $# -lt 3 ]; then
    echo "用法: $0 <model_name> <dataset_name> <num_gpus> [algorithm] [gpu_batch_size] [rollout_n] [max_response_length] [num_nodes] [model_path]"
    echo "示例: $0 Qwen2.5-Math-1.5B math17 8 grpo 16 8 2048 1"
    exit 1
fi

PRM_ENDPOINT="http://localhost:4997/v1/step_rewards"
MODEL_NAME_INPUT="$1"
DATASET_NAME="$2"
NUM_GPUS="$3"                  # 默认视为每节点 GPU 数
ALGORITHM="${4:-grpo}"         # grpo/srpo2/srpo3/srpo4/rflux
GPU_BATCH_SIZE="${5:-16}"
ROLLOUT_N="${6:-8}"
MAX_RESPONSE_LENGTH="${7:-2048}"
NUM_NODES="${8:-1}"
MODEL_PATH_INPUT="${9:-}"

# 模型映射表
MAPPING_FILE="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/model_mapping.txt"
MODEL_NAME="$MODEL_NAME_INPUT"
MODEL_SHORT="$MODEL_NAME_INPUT"
MODEL_PATH=""

if [ -f "$MAPPING_FILE" ]; then
    while IFS='|' read -r short_name full_name model_path || [ -n "$short_name" ]; do
        case "$short_name" in
            \#*) continue ;;
            "") continue ;;
        esac
        if [ "$MODEL_NAME_INPUT" = "$short_name" ] || [ "$MODEL_NAME_INPUT" = "$full_name" ]; then
            MODEL_NAME="$full_name"
            MODEL_SHORT="$short_name"
            MODEL_PATH="$model_path"
            break
        fi
    done < "$MAPPING_FILE"
fi

if [ -z "$MODEL_PATH" ]; then
    if [ -n "$MODEL_PATH_INPUT" ]; then
        MODEL_PATH="$MODEL_PATH_INPUT"
    else
        MODEL_PATH="/mnt/shared-storage-user/mineru4s/dingruiyi/${MODEL_NAME_INPUT}"
    fi
    MODEL_SHORT=$(echo "$MODEL_NAME_INPUT" | sed 's/Qwen2.5-Math-//' | sed 's/Qwen2.5-//' | sed 's/-Instruct//' | tr '[:upper:]' '[:lower:]')
    MODEL_NAME="$MODEL_NAME_INPUT"
fi

# 算法检查
if [ "$ALGORITHM" != "grpo" ] && [ "$ALGORITHM" != "srpo3" ] && [ "$ALGORITHM" != "srpo2" ] && [ "$ALGORITHM" != "rflux" ] && [ "$ALGORITHM" != "srpo4" ] && [ "$ALGORITHM" != "srpo" ]; then
    echo "错误: 算法类型必须是 grpo/srpo/srpo2/srpo3/srpo4/rflux，当前为: $ALGORITHM"
    exit 1
fi

# 数值参数校验
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -le 0 ]; then
    echo "错误: GPU数量必须是正整数，当前为: $NUM_GPUS"
    exit 1
fi
if ! [[ "$GPU_BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$GPU_BATCH_SIZE" -le 0 ]; then
    echo "错误: GPU batch size必须是正整数，当前为: $GPU_BATCH_SIZE"
    exit 1
fi
if ! [[ "$ROLLOUT_N" =~ ^[0-9]+$ ]] || [ "$ROLLOUT_N" -le 0 ]; then
    echo "错误: rollout数量必须是正整数，当前为: $ROLLOUT_N"
    exit 1
fi
if ! [[ "$MAX_RESPONSE_LENGTH" =~ ^[0-9]+$ ]] || [ "$MAX_RESPONSE_LENGTH" -le 0 ]; then
    echo "错误: 最大响应长度必须是正整数，当前为: $MAX_RESPONSE_LENGTH"
    exit 1
fi

# 计算总 batch（节点数 * 每节点GPU * 每GPU batch）
TOTAL_BATCH_SIZE=$((NUM_GPUS * GPU_BATCH_SIZE * NUM_NODES))
MAX_PROMPT_LENGTH=2048

# 数据路径推断
DATASET_CAPITALIZED=$(echo "${DATASET_NAME}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
save_freq=100
if [ "$DATASET_NAME" = "gsm8k" ]; then
    TRAIN_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/gsm8k/gsm8ktrain.parquet"
    VAL_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/gsm8k/gsm8ktest.parquet"
    save_freq=58
else
    TRAIN_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/train${DATASET_CAPITALIZED}_data.parquet"
    VAL_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/val${DATASET_CAPITALIZED}_data.parquet"
    save_freq=93
    if [ ! -f "$TRAIN_DATA" ]; then
        ALT_TRAIN="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/train${DATASET_NAME}_data.parquet"
        ALT_VAL="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/val${DATASET_NAME}_data.parquet"
        if [ -f "$ALT_TRAIN" ]; then
            TRAIN_DATA="$ALT_TRAIN"; VAL_DATA="$ALT_VAL"
        else
            ALT_TRAIN="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}split_train.parquet"
            ALT_VAL="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}split_val.parquet"
            if [ -f "$ALT_TRAIN" ]; then
                TRAIN_DATA="$ALT_TRAIN"; VAL_DATA="$ALT_VAL"
            else
                ALT_TRAIN="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}_train.parquet"
                ALT_VAL="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}_val.parquet"
                if [ -f "$ALT_TRAIN" ]; then
                    TRAIN_DATA="$ALT_TRAIN"; VAL_DATA="$ALT_VAL"
                else
                    ALT_TRAIN="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}.parquet"
                    ALT_VAL="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/${DATASET_NAME}val.parquet"
                    if [ -f "$ALT_TRAIN" ]; then
                        TRAIN_DATA="$ALT_TRAIN"; VAL_DATA="$ALT_VAL"
                    else
                        echo "警告: 未找到训练数据文件，使用默认路径: $TRAIN_DATA"
                    fi
                fi
            fi
        fi
    fi
fi

# 清理任务名（保留以备兼容）
clean_rjob_name() {
    local name="$1"
    name=$(echo "$name" | sed 's/[^a-zA-Z0-9]/-/g')
    name=$(echo "$name" | sed 's/^\-\+//; s/\-\+$//')
    name=$(echo "$name" | sed 's/\-\+/-/g')
    if [ -z "$name" ] || [ ${#name} -lt 3 ]; then
        name="task-${name}"
    fi
    if [ ${#name} -gt 63 ]; then
        name=$(echo "$name" | cut -c1-63)
        name=$(echo "$name" | sed 's/\-\+$//')
        if [ -z "$name" ] || [ ${#name} -lt 3 ]; then
            name="task-$(echo "$1" | sed 's/[^a-zA-Z0-9]//g' | cut -c1-60)"
        fi
    fi
    if ! [[ "$name" =~ ^[a-zA-Z0-9] ]]; then
        name="t${name}"
    fi
    if ! [[ "$name" =~ [a-zA-Z0-9]$ ]]; then
        name="${name}0"
    fi
    echo "$name"
}

EXPERIMENT_NAME="${DATASET_NAME}-${MODEL_SHORT}-${ALGORITHM}-r${ROLLOUT_N}"
PROJECT_NAME="verl_srpo_${DATASET_NAME}"
CHECKPOINT_DIR="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# conversation_parser 选择
CONV_PARSER="qwen"
if [ "$ALGORITHM" = "srpo2" ] || [ "$ALGORITHM" = "srpo3" ]; then
    CONV_PARSER="deepseek"
elif [ "$ALGORITHM" = "rflux" ]; then
    CONV_PARSER="deepseek"
elif [ "$ALGORITHM" = "srpo4" ]; then
    CONV_PARSER="qwen"
fi

# 生成 Hydra 配置
CONFIG_DIR="$(pwd)/configs"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="${CONFIG_DIR}/${EXPERIMENT_NAME}.yaml"

cat > "$CONFIG_FILE" << EOF
# Auto-generated by submit_training.sh
# 用法示例：
#   python -m verl.trainer.main_ppo \\
#     --config-path=./configs --config-name=${EXPERIMENT_NAME}

defaults:
  - _self_

algorithm:
  adv_estimator: NONE
  use_kl_in_reward: false
  gamma: 1.0
  lam: 1.0

reward_manager:
  source: "register"
  name: "${ALGORITHM}"

reward_model:
  reward_kwargs:
    endpoint: "${PRM_ENDPOINT}"
    proxies:
      http: null
      https: null
    max_branches: 5
    min_gap: 10
    conversation_parser: "${CONV_PARSER}"

actor_rollout_ref:
  hybrid_engine: true
  model:
    path: "${MODEL_PATH}"
    tokenizer_path: null
    use_remove_padding: true
    enable_gradient_checkpointing: true
  actor:
    strategy: fsdp2
    ppo_micro_batch_size_per_gpu: 4
    ppo_mini_batch_size: ${TOTAL_BATCH_SIZE}
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    entropy_coeff: 0.0
    fsdp_config:
      param_offload: false
      optimizer_offload: false
    optim:
      lr: 1.0e-6
  ref:
    fsdp_config:
      param_offload: true
    log_prob_micro_batch_size_per_gpu: 4
  rollout:
    name: vllm
    n: ${ROLLOUT_N}
    max_num_batched_tokens: 65536
    log_prob_micro_batch_size_per_gpu: 4
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.6
    val_kwargs:
      do_sample: true
      temperature: 1.0
      top_p: 0.9
    agent:
      num_workers: 1

data:
  train_files: "${TRAIN_DATA}"
  val_files: "${VAL_DATA}"
  train_batch_size: ${TOTAL_BATCH_SIZE}
  max_prompt_length: ${MAX_PROMPT_LENGTH}
  max_response_length: ${MAX_RESPONSE_LENGTH}
  filter_overlong_prompts: true
  truncation: error

trainer:
  project_name: "${PROJECT_NAME}"
  experiment_name: "${EXPERIMENT_NAME}"
  default_local_dir: "${CHECKPOINT_DIR}"
  logger: ["console"]
  critic_warmup: 0
  max_actor_ckpt_to_keep: 10
  save_freq: ${save_freq}
  test_freq: 2000
  total_epochs: 10
  n_gpus_per_node: ${NUM_GPUS}
  nnodes: ${NUM_NODES}
  n_cpus_per_node: 32
  rollout_correction:
    bypass_mode: false

ray_kwargs:
  ray_init:
    address: "auto"
    num_gpus: ${NUM_GPUS}
    num_cpus: 32
    _temp_dir: "/tmp/ray"
    runtime_env:
      env_vars:
        TOKENIZERS_PARALLELISM: "false"
        NCCL_DEBUG: "WARN"
        VLLM_LOGGING_LEVEL: "WARN"
        VLLM_ALLOW_RUNTIME_LORA_UPDATING: "true"

logging:
  wandb:
    enable: false
    project: "${PROJECT_NAME}"
    run_name: "${EXPERIMENT_NAME}"
EOF

echo "=========================================="
echo "已生成 Hydra 配置: $CONFIG_FILE"
echo "模型输入:        $MODEL_NAME_INPUT"
echo "模型名称:        $MODEL_NAME"
echo "模型简称:        $MODEL_SHORT"
echo "数据集名称:      $DATASET_NAME"
echo "GPU数量(每节点): $NUM_GPUS"
echo "节点数量:        $NUM_NODES"
echo "算法类型:        $ALGORITHM"
echo "每个GPU batch:   $GPU_BATCH_SIZE"
echo "总batch size:    $TOTAL_BATCH_SIZE"
echo "最大响应长度:    $MAX_RESPONSE_LENGTH"
echo "Rollout数量:     $ROLLOUT_N"
echo "模型路径:        $MODEL_PATH"
echo "训练数据:        $TRAIN_DATA"
echo "验证数据:        $VAL_DATA"
echo "实验名称:        $EXPERIMENT_NAME"
echo "检查点目录:      $CHECKPOINT_DIR"
echo "保存频率:        $save_freq"
echo "=========================================="
echo ""
echo "运行示例："
echo "python -m verl.trainer.main_ppo --config-path=./configs --config-name=${EXPERIMENT_NAME}"
exit 0
