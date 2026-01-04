#!/bin/bash
# 通过 rjob 提交批量测试任务

name="batch-evaluation-test-pure7"
rjob delete $name 2>/dev/null || true

rjob submit \
    --name=$name \
    --gpu=4 \
    --memory=1280000 \
    --cpu=170 \
    --charged-group=mineru4sh_gpu \
    --private-machine=group \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    --image=registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/yehc:torch-2.6.0-57d787c2-0627 \
    --host-network=true \
    -- bash -c "bash /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/batch_test.sh > /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/batch_evaluation_results.log 2> /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/batch_evaluation_results.err"

