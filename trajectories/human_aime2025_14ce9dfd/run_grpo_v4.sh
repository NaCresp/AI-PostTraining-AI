#!/bin/bash
# GRPO v4: Continue from best v3 checkpoint
# Key changes from v3:
# - Lower LR (2e-6) since we're building on v3's progress
# - Higher entropy_coeff (0.01) to maintain exploration diversity
# - Updated reward function v3 (integer bonus, degeneration penalty)
# - 50 total steps

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

WORK_DIR="/workspace/AI4AI/experiments/claude-code-human/workspace"
MODEL_PATH="${WORK_DIR}/checkpoints/grpo_v3_step50_hf"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_model_len=8704 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    \
    data.train_files=${WORK_DIR}/data/train.parquet \
    data.val_files=${WORK_DIR}/data/val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    \
    reward.custom_reward_function.path=${WORK_DIR}/reward_function_v3.py \
    reward.custom_reward_function.name=compute_score \
    \
    trainer.total_epochs=3 \
    trainer.total_training_steps=50 \
    trainer.project_name=grpo_math \
    trainer.experiment_name=grpo_v4 \
    trainer.logger='["console"]' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.default_local_dir=${WORK_DIR}/checkpoints/grpo_v4 \
    2>&1 | tee ${WORK_DIR}/logs/grpo_v4.log
