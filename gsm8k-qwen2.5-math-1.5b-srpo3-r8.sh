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
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/gsm8k/gsm8ktrain.parquet \
    data.val_files=/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/gsm8k/gsm8ktest.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048\
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/mineru4s/dingruiyi/Qwen2.5-Math-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4\
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_srpo_gsm8k' \
    trainer.experiment_name='gsm8k-qwen2.5-math-1.5b-srpo3-r8' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=58 \
    trainer.test_freq=2000 \
    trainer.total_epochs=10 \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.default_local_dir='/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/checkpoints/verl_srpo_gsm8k/gsm8k-qwen2.5-math-1.5b-srpo3-r8' \
    reward_model.reward_manager=srpo3
