#!/bin/bash
# 批量测试脚本：运行所有模型和数据集组合的测试

source /mnt/shared-storage-user/mineru4s/dingruiyi/anaconda/bin/activate
conda activate verl

# 限制可见 GPU（与 rjob --gpu 对齐）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

python /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/batch_test.py

