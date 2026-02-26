#!/usr/bin/env bash
set -euo pipefail

VERL_DIR="$HOME/projects/GRPO/verl"
VENV="$VERL_DIR/.venv"

source "$VENV/bin/activate"

EXP="grpo_qwen3_1p7b_gsm8k_think_4gpu_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$HOME/projects/GRPO/training/$EXP"

export HF_HOME="$HOME/projects/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$HOME/projects/.cache"
export RAY_TMPDIR="$HOME/projects/.cache/ray"
export TMPDIR="$HOME/projects/.cache/tmp"

mkdir -p "$RUN_DIR"/{logs,checkpoints,config,tensorboard,val_generations} \
         "$HOME/projects/data" \
         "$HF_HOME" "$HF_DATASETS_CACHE" "$RAY_TMPDIR" "$TMPDIR"

python3 "$VERL_DIR/examples/data_preprocess/gsm8k.py" --local_save_dir "$HOME/projects/data/gsm8k"

export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
reward_path="/home/seankopylov/projects/GRPO/training/scripts/reward_gsm8k_flexible_wrapper.py"

cat > "$RUN_DIR/config/launch_cmd.sh" <<CMD
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V1=1 PYTHONUNBUFFERED=1 \\
HF_HOME=$HF_HOME \\
HF_DATASETS_CACHE=$HF_DATASETS_CACHE \\
TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \\
XDG_CACHE_HOME=$XDG_CACHE_HOME \\
RAY_TMPDIR=$RAY_TMPDIR \\
TMPDIR=$TMPDIR \\
TENSORBOARD_DIR=$TENSORBOARD_DIR \\
python3 -m verl.trainer.main_ppo \\
  algorithm.adv_estimator=grpo \\
  data.train_files=$HOME/projects/data/gsm8k/train.parquet \\
  data.val_files=$HOME/projects/data/gsm8k/test.parquet \\
  data.return_raw_chat=True \\
  data.train_batch_size=32 \\
  data.max_prompt_length=512 \\
  data.max_response_length=2048 \\
  data.filter_overlong_prompts=True \\
  data.truncation=error \\
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \\
  +actor_rollout_ref.model.override_config.attn_implementation=eager \\
  actor_rollout_ref.model.use_remove_padding=False \\
  actor_rollout_ref.actor.optim.lr=2e-6 \\
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \\
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \\
  actor_rollout_ref.actor.use_kl_loss=True \\
  actor_rollout_ref.actor.kl_loss_coef=0.003 \\
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
  actor_rollout_ref.rollout.name=vllm \\
  actor_rollout_ref.rollout.mode=async \\
  actor_rollout_ref.rollout.agent.num_workers=4 \\
  actor_rollout_ref.rollout.n=8 \\
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \\
  actor_rollout_ref.rollout.max_num_seqs=192 \\
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \\
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \\
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \\
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \\
  algorithm.use_kl_in_reward=False \\
  reward.custom_reward_function.path=$reward_path \\
  reward.custom_reward_function.name=compute_score \\
  trainer.critic_warmup=0 \\
  trainer.val_before_train=True \\
  trainer.log_val_generations=8 \\
  trainer.validation_data_dir=$RUN_DIR/val_generations \\
  trainer.logger='["console","tensorboard"]' \\
  trainer.default_local_dir=$RUN_DIR/checkpoints \\
  trainer.project_name=codex_grpo \\
  trainer.experiment_name=$EXP \\
  trainer.n_gpus_per_node=4 \\
  trainer.nnodes=1 \\
  trainer.save_freq=100 \\
  trainer.test_freq=5 \\
  trainer.total_epochs=1
CMD
chmod +x "$RUN_DIR/config/launch_cmd.sh"

cd "$VERL_DIR"
LOG_FILE="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "EXP=$EXP"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"

(time bash "$RUN_DIR/config/launch_cmd.sh" 2>&1 | tee "$LOG_FILE")


# source /home/seankopylov/projects/GRPO/verl/.venv/bin/activate
# tensorboard --logdir /home/seankopylov/projects/GRPO/training/grpo_qwen3_1p7b_gsm8k_think_4gpu_20260226_132716/tensorboard --host 0.0.0.0 --port 10201
# ssh -L 10201:localhost:10201 ya100
