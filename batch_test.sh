#!/bin/bash
# 批量测试脚本：运行所有模型和数据集组合的测试

source /mnt/shared-storage-user/mineru4s/dingruiyi/anaconda/bin/activate
conda activate verl

python /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/batch_test.py

