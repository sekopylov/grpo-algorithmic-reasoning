#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

VERL_DIR="verl"
VENV="$VERL_DIR/.venv"
source "$VENV/bin/activate"

EXP="grpo_qwen3_1p7b_taskenv_think_4gpu_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="training/$EXP"

CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
ROLLOUT_GPU_UTILIZATION="${ROLLOUT_GPU_UTILIZATION:-0.6}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"

export HF_HOME="$ROOT_DIR/.cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export RAY_TMPDIR="$ROOT_DIR/.ray"
export TMPDIR="$ROOT_DIR/.tmp"

mkdir -p "$RUN_DIR"/{logs,checkpoints,config,tensorboard,val_generations,train_generations} \
         data/taskenv \
         "$HF_HOME" "$HF_DATASETS_CACHE" "$RAY_TMPDIR" "$TMPDIR"

python3 training/scripts/taskenv_preprocess.py \
  --local_save_dir data/taskenv \
  --train_size 8192 \
  --val_per_difficulty 128 \
  --train_difficulties 1,2,3,4,5,6,7,8,9,10 \
  --val_difficulties 1,2,3,4,5,6,7,8,9,10 \
  --train_seed 42 \
  --val_seed 31415 \
  --modulus 1000 \
  --max_steps 220 \
  --max_attempts 50

export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
reward_path="training/scripts/reward_taskenv_wrapper.py"

cat > "$RUN_DIR/config/launch_cmd.sh" <<CMD
cd $ROOT_DIR
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES VLLM_USE_V1=1 PYTHONUNBUFFERED=1 \\
HF_HOME=$HF_HOME \\
HF_DATASETS_CACHE=$HF_DATASETS_CACHE \\
TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \\
XDG_CACHE_HOME=$XDG_CACHE_HOME \\
RAY_TMPDIR=$RAY_TMPDIR \\
TMPDIR=$TMPDIR \\
TENSORBOARD_DIR=$TENSORBOARD_DIR \\
python3 -m verl.trainer.main_ppo \\
  algorithm.adv_estimator=grpo \\
  data.train_files=data/taskenv/train.parquet \\
  data.val_files=data/taskenv/test.parquet \\
  data.return_raw_chat=True \\
  data.train_batch_size=32 \\
  data.max_prompt_length=1536 \\
  data.max_response_length=768 \\
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
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_UTILIZATION \\
  actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \\
  actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \\
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
  trainer.rollout_data_dir=$RUN_DIR/train_generations \\
  trainer.validation_data_dir=$RUN_DIR/val_generations \\
  trainer.logger="[\"console\",\"tensorboard\"]" \\
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

LOG_FILE="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "EXP=$EXP"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "CUDA_DEVICES=$CUDA_DEVICES"
echo "ROLLOUT_GPU_UTILIZATION=$ROLLOUT_GPU_UTILIZATION"
echo "ROLLOUT_MAX_NUM_SEQS=$ROLLOUT_MAX_NUM_SEQS"
echo "ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS"

time bash "$RUN_DIR/config/launch_cmd.sh" 2>&1 | tee "$LOG_FILE"


# tensorboard --logdir /home/seankopylov/projects/GRPO/training/grpo_qwen3_1p7b_taskenv_think_4gpu_2048_20260227_005449/tensorboard --host 0.0.0.0 --port 10901
