#!/usr/bin/env bash
set -x

TASK_NAME="sk2decompile-rl-grpo"
LOG_FILE="${TASK_NAME}.log"
ERR_FILE="${TASK_NAME}.err"

touch "$LOG_FILE"
touch "$ERR_FILE"

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_trainer-sk2decompile.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=SK2DECOMPILE/data/sk2decompile-rl-examples.parquet \
    data.val_files=SK2DECOMPILE/data/sk2decompile-rl-examples.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=LLM4Binary/llm4decompile-6.7b-v2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_lm4dc' \
    trainer.experiment_name=$TASK_NAME \
    trainer.default_local_dir=SK2DECOMPILE/saves/${TASK_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=2 "$@" \
    > >(tee -a "$LOG_FILE") \
    2> >(tee -a "$ERR_FILE" >&2)

echo "STDOUT saved to: $LOG_FILE"
echo "STDERR saved to: $ERR_FILE"
