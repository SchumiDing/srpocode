#!/bin/bash

# 列出所有可用的模型简称映射

MAPPING_FILE="/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/model_mapping.txt"

echo "=========================================="
echo "模型简称映射表"
echo "=========================================="
echo ""
echo "格式: 简称 | 完整模型名称 | 模型路径"
echo ""

if [ ! -f "$MAPPING_FILE" ]; then
    echo "错误: 映射文件不存在: $MAPPING_FILE"
    exit 1
fi

printf "%-20s | %-35s | %s\n" "简称" "完整模型名称" "模型路径"
echo "------------------------------------------------------------------------------------------------------------------------"

while IFS='|' read -r short_name full_name model_path || [ -n "$short_name" ]; do
    # 跳过注释行和空行
    case "$short_name" in
        \#*) continue ;;
        "") continue ;;
    esac
    
    printf "%-20s | %-35s | %s\n" "$short_name" "$full_name" "$model_path"
done < "$MAPPING_FILE"

echo ""
echo "=========================================="
echo "使用示例:"
echo "=========================================="
echo "./submit_training.sh qwen1.5 math17 6"
echo "./submit_training.sh qwen3-8b math17 8 grpo"
echo "./submit_training.sh Qwen2.5-Math-1.5B math17 6 srpo 16"
echo ""

