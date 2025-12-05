#!/bin/bash
# Rjob提交脚本：批量评估checkpoint目录下的所有checkpoint
# 使用方法:
#   bash eval_checkpoints_rjob.sh <checkpoint_dir> <dataset> [batch_size] [job_name]
#
# 示例:
#   bash eval_checkpoints_rjob.sh checkpoints/verl_srpo_math500/math500-qwen2.5-math-1.5b-rflux-r8 math500
#   bash eval_checkpoints_rjob.sh checkpoints/verl_srpo_gsm8k/gsm8k-qwen2.5-math-1.5b-rflux-r8 gsm8k 4 eval_gsm8k

set -e

# 参数检查
if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint_dir> <dataset> [batch_size] [job_name]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_dir: 包含多个checkpoint的目录路径"
    echo "  dataset:        测试数据集名称 (math500, gsm8k, aime2024, aime2025)"
    echo "  batch_size:     Batch大小 (默认: 4)"
    echo "  job_name:       任务名称 (默认: eval_<dataset>_<checkpoint_basename>)"
    echo ""
    echo "Examples:"
    echo "  $0 checkpoints/verl_srpo_math500/math500-qwen2.5-math-1.5b-rflux-r8 math500"
    echo "  $0 checkpoints/verl_srpo_gsm8k/gsm8k-qwen2.5-math-1.5b-rflux-r8 gsm8k 4"
    exit 1
fi

CHECKPOINT_DIR=$1
DATASET=$2
BATCH_SIZE=${3:-4}

# 验证数据集名称
if [[ ! "$DATASET" =~ ^(math500|gsm8k|aime2024|aime2025)$ ]]; then
    echo "Error: Invalid dataset. Must be one of: math500, gsm8k, aime2024, aime2025"
    exit 1
fi

# 清理字符串以符合rjob命名规则的函数
# 规则: ^[a-zA-Z0-9][-a-zA-Z0-9]{1,61}[a-zA-Z0-9]$
clean_job_name() {
    local name=$1
    # 只保留字母、数字和连字符
    name=$(echo "$name" | sed 's/[^a-zA-Z0-9-]//g')
    # 合并多个连字符
    name=$(echo "$name" | sed 's/--*/-/g')
    # 移除开头和结尾的连字符
    name=$(echo "$name" | sed 's/^-\+//' | sed 's/-\+$//')
    # 确保以字母或数字开头
    if [[ ! "$name" =~ ^[a-zA-Z0-9] ]]; then
        name="a${name}"
    fi
    # 确保以字母或数字结尾（移除末尾的连字符）
    while [[ "$name" =~ -$ ]]; do
        name="${name%-}"
    done
    if [[ ! "$name" =~ [a-zA-Z0-9]$ ]]; then
        name="${name}0"
    fi
    # 限制长度：为任务名称留出空间（任务ID = job_name-task_name，总共最多63字符）
    # 保守估计任务名称最多15字符，所以job_name限制为45字符（留3字符余量）
    if [ ${#name} -gt 45 ]; then
        name="${name:0:45}"
        # 移除末尾的连字符（如果有）
        while [[ "$name" =~ -$ ]]; do
            name="${name%-}"
        done
        # 确保截断后仍以字母或数字结尾
        if [[ ! "$name" =~ [a-zA-Z0-9]$ ]]; then
            name="${name%?}0"
        fi
    fi
    # 强制确保以字母或数字结尾（移除最后一个字符如果是连字符）
    if [[ "${name: -1}" == "-" ]]; then
        name="${name%-}"
    fi
    if [[ ! "$name" =~ [a-zA-Z0-9]$ ]]; then
        name="${name}0"
    fi
    # 确保至少3个字符
    if [ ${#name} -lt 3 ]; then
        name="${name}00"
        name="${name:0:3}"
    fi
    # 最终验证：确保名称匹配rjob命名规则
    # 规则: ^[a-zA-Z0-9][-a-zA-Z0-9]{1,61}[a-zA-Z0-9]$
    # 简化验证：确保以字母数字开头和结尾，长度在3-63之间，中间只包含字母数字和连字符
    # 如果名称过长（超过63），逐步缩短
    while [ ${#name} -gt 63 ]; do
        name="${name%?}"
        while [[ "$name" =~ -$ ]]; do
            name="${name%-}"
        done
        if [[ ! "$name" =~ [a-zA-Z0-9]$ ]]; then
            name="${name}0"
        fi
    done
    echo "$name"
}

# 生成任务名称
# rjob命名规则: ^[a-zA-Z0-9][-a-zA-Z0-9]{1,61}[a-zA-Z0-9]$
if [ -n "$4" ]; then
    JOB_NAME=$(clean_job_name "$4")
else
    # 从checkpoint_dir提取basename
    CHECKPOINT_BASENAME=$(basename "$CHECKPOINT_DIR")
    # 构建任务名
    RAW_JOB_NAME="eval-${DATASET}-${CHECKPOINT_BASENAME}"
    JOB_NAME=$(clean_job_name "$RAW_JOB_NAME")
fi

echo "=========================================="
echo "Rjob Task Configuration"
echo "=========================================="
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Dataset:        $DATASET"
echo "Batch Size:     $BATCH_SIZE"
echo "Job Name:       $JOB_NAME"
echo "=========================================="

# 处理checkpoint_dir路径（如果是相对路径，转换为绝对路径）
if [[ "$CHECKPOINT_DIR" != /* ]]; then
    # 相对路径，需要基于项目根目录
    CHECKPOINT_DIR="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/$CHECKPOINT_DIR"
fi

# 构建Python命令（在rjob任务内部激活conda环境）
PYTHON_CMD="source /mnt/shared-storage-user/mineru4s/dingruiyi/anaconda/bin/activate && conda activate verl && cd /mnt/shared-storage-user/mineru4s/dingruiyi/srpo && python eval_all_checkpoints.py --checkpoint_dir '$CHECKPOINT_DIR' --dataset $DATASET --batch_size $BATCH_SIZE"

# 删除旧任务（如果存在）
echo "Deleting old job if exists: $JOB_NAME"
rjob delete "$JOB_NAME" 2>/dev/null || true

# 提交rjob任务
echo "Submitting rjob task: $JOB_NAME"
rjob submit \
    --name="$JOB_NAME" \
    --gpu=1 \
    --memory=96000 \
    --cpu=16 \
    --charged-group=mineru4sh_gpu \
    --private-machine=group \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    --image=registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/yehc:torch-2.6.0-57d787c2-0627 \
    --host-network=true \
    -- bash -c "$PYTHON_CMD"

echo ""
echo "=========================================="
echo "Task submitted successfully!"
echo "=========================================="
echo "Job Name: $JOB_NAME"
echo "You can check the status with: rjob status $JOB_NAME"
echo "You can view logs with: rjob logs $JOB_NAME"
echo "=========================================="

