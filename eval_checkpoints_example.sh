#!/bin/bash
# 使用示例脚本

# ============================================
# 方式1: 直接运行Python脚本（本地执行）
# ============================================

# 示例1: 评估math500数据集的所有checkpoint
python eval_all_checkpoints.py \
    --checkpoint_dir checkpoints/verl_srpo_math500/math500-qwen2.5-math-1.5b-rflux-r8 \
    --dataset math500 \
    --batch_size 4

# 示例2: 评估gsm8k数据集的所有checkpoint
python eval_all_checkpoints.py \
    --checkpoint_dir checkpoints/verl_srpo_gsm8k/gsm8k-qwen2.5-math-1.5b-rflux-r8 \
    --dataset gsm8k \
    --batch_size 4

# 示例3: 评估aime2024数据集的所有checkpoint
python eval_all_checkpoints.py \
    --checkpoint_dir checkpoints/verl_srpo_math500/math500-qwen2.5-math-1.5b-rflux-r8 \
    --dataset aime2024 \
    --batch_size 4

# 示例4: 评估aime2025数据集的所有checkpoint
python eval_all_checkpoints.py \
    --checkpoint_dir checkpoints/verl_srpo_math500/math500-qwen2.5-math-1.5b-rflux-r8 \
    --dataset aime2025 \
    --batch_size 4

# 示例5: 指定输出目录和临时目录
python eval_all_checkpoints.py \
    --checkpoint_dir checkpoints/verl_srpo_math500/math500-qwen2.5-math-1.5b-rflux-r8 \
    --dataset math500 \
    --output_dir ./my_results \
    --temp_dir ./my_temp \
    --batch_size 4

# ============================================
# 方式2: 使用rjob提交任务（推荐，在集群上运行）
# ============================================

# 示例6: 通过rjob提交math500评估任务
bash eval_checkpoints_rjob.sh \
    checkpoints/verl_srpo_math500/math500-qwen2.5-math-1.5b-rflux-r8 \
    math500 \
    4

# 示例7: 通过rjob提交gsm8k评估任务
bash eval_checkpoints_rjob.sh \
    checkpoints/verl_srpo_gsm8k/gsm8k-qwen2.5-math-1.5b-rflux-r8 \
    gsm8k \
    4

# 示例8: 通过rjob提交aime2024评估任务
bash eval_checkpoints_rjob.sh \
    checkpoints/verl_srpo_math500/math500-qwen2.5-math-1.5b-rflux-r8 \
    aime2024 \
    4

# 示例9: 通过rjob提交aime2025评估任务，并指定任务名称
bash eval_checkpoints_rjob.sh \
    checkpoints/verl_srpo_math500/math500-qwen2.5-math-1.5b-rflux-r8 \
    aime2025 \
    4 \
    my_eval_task

# ============================================
# 查看rjob任务状态
# ============================================
# rjob status <job_name>
# rjob logs <job_name>

