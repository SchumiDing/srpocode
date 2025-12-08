#!/bin/bash
# 持续监测checkpoint路径，当检测到完整的10个checkpoint时自动运行评估脚本
# 使用方法:
#   bash monitor_checkpoints.sh <checkpoint_path> [check_interval]
#
# 示例:
#   bash monitor_checkpoints.sh checkpoints/verl_srpo_math500/math500-qwen2.5-math-7b-grpo-r8
#   bash monitor_checkpoints.sh checkpoints/verl_srpo_math500/math500-qwen2.5-math-7b-grpo-r8 60

set -e

# 参数检查
if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [check_interval]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path: 要监测的checkpoint目录路径"
    echo "  check_interval:  检查间隔（秒），默认60秒"
    echo ""
    echo "Examples:"
    echo "  $0 checkpoints/verl_srpo_math500/math500-qwen2.5-math-7b-grpo-r8"
    echo "  $0 checkpoints/verl_srpo_math500/math500-qwen2.5-math-7b-grpo-r8 120"
    exit 1
fi

CHECKPOINT_PATH=$1
CHECK_INTERVAL=${2:-60}
REQUIRED_CHECKPOINTS=10
DATASET="math500"
BATCH_SIZE=128
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_checkpoints_rjob.sh"

# 处理checkpoint_path（如果是相对路径，转换为绝对路径）
if [[ "$CHECKPOINT_PATH" != /* ]]; then
    # 相对路径，需要基于项目根目录
    CHECKPOINT_PATH="$SCRIPT_DIR/$CHECKPOINT_PATH"
fi

# 记录已处理的路径，避免重复运行
PROCESSED_MARKER_FILE="${CHECKPOINT_PATH}.eval_processed"

echo "=========================================="
echo "Checkpoint Monitor Configuration"
echo "=========================================="
echo "Checkpoint Path: $CHECKPOINT_PATH"
echo "Check Interval:  ${CHECK_INTERVAL} seconds"
echo "Required Checkpoints: $REQUIRED_CHECKPOINTS"
echo "Dataset:         $DATASET"
echo "Batch Size:      $BATCH_SIZE"
echo "Eval Script:     $EVAL_SCRIPT"
echo "=========================================="
echo ""

# 检查评估脚本是否存在
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Eval script not found: $EVAL_SCRIPT"
    exit 1
fi

# 检查checkpoint路径是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Warning: Checkpoint path does not exist: $CHECKPOINT_PATH"
    echo "Will start monitoring once the path is created..."
fi

# 函数：检查checkpoint数量
check_checkpoint_count() {
    local path=$1
    if [ ! -d "$path" ]; then
        echo 0
        return
    fi
    
    # 统计global_step_*目录的数量
    local count=$(find "$path" -maxdepth 1 -type d -name "global_step_*" | wc -l)
    echo $count
}

# 函数：检查是否已经处理过
is_already_processed() {
    local path=$1
    if [ -f "$PROCESSED_MARKER_FILE" ]; then
        return 0
    fi
    return 1
}

# 函数：标记为已处理
mark_as_processed() {
    local path=$1
    touch "$PROCESSED_MARKER_FILE"
    echo "Marked as processed: $PROCESSED_MARKER_FILE"
}

# 主循环
echo "Starting monitoring loop (press Ctrl+C to stop)..."
echo ""

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 检查路径是否存在
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "[$TIMESTAMP] Iteration $ITERATION: Path does not exist yet, waiting..."
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # 检查是否已经处理过
    if is_already_processed "$CHECKPOINT_PATH"; then
        echo "[$TIMESTAMP] Iteration $ITERATION: Already processed, skipping..."
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # 检查checkpoint数量
    COUNT=$(check_checkpoint_count "$CHECKPOINT_PATH")
    echo "[$TIMESTAMP] Iteration $ITERATION: Found $COUNT checkpoint(s) (required: $REQUIRED_CHECKPOINTS)"
    
    if [ "$COUNT" -ge "$REQUIRED_CHECKPOINTS" ]; then
        echo ""
        echo "=========================================="
        echo "✓ Found $COUNT checkpoints (>= $REQUIRED_CHECKPOINTS)"
        echo "Starting evaluation..."
        echo "=========================================="
        echo ""
        
        # 运行评估脚本
        # 使用相对路径传递给eval脚本（脚本内部会处理路径转换）
        RELATIVE_PATH=$(realpath --relative-to="$SCRIPT_DIR" "$CHECKPOINT_PATH" 2>/dev/null || echo "$CHECKPOINT_PATH")
        
        bash "$EVAL_SCRIPT" "$RELATIVE_PATH" "$DATASET" "$BATCH_SIZE"
        
        # 标记为已处理
        mark_as_processed "$CHECKPOINT_PATH"
        break
        echo ""
        echo "=========================================="
        echo "Evaluation task submitted successfully!"
        echo "Monitoring will continue but will skip this path in future checks."
        echo "=========================================="
        echo ""
    fi
    
    sleep $CHECK_INTERVAL
done

