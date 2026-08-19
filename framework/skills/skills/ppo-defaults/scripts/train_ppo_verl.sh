#!/usr/bin/env bash
set -euo pipefail

# Reference script: verl PPO launch with conservative defaults.
# Edit args and EXTRA_OVERRIDES for your run.

usage() {
  cat <<'USAGE'
Usage:
  train_ppo_verl.sh \
    --train_file <train.parquet> \
    --val_file <val.parquet> \
    --model_path <base_or_sft_model> \
    --save_dir <checkpoint_dir> \
    [--gpus 4] [--lr 1e-6] [--kl_coef 0.005]

Optional env:
  EXTRA_OVERRIDES='actor_rollout_ref.rollout.temperature=0.8 data.train_batch_size=512'
USAGE
}

TRAIN_FILE=""
VAL_FILE=""
MODEL_PATH=""
SAVE_DIR=""
GPUS=4
LR="1e-6"
KL_COEF="0.005"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_file) TRAIN_FILE="$2"; shift 2 ;;
    --val_file) VAL_FILE="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --save_dir) SAVE_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$TRAIN_FILE" && -n "$VAL_FILE" && -n "$MODEL_PATH" && -n "$SAVE_DIR" ]] || {
  echo "Missing required args." >&2
  usage
  exit 1
}

mkdir -p "$SAVE_DIR"

: "${EXTRA_OVERRIDES:=}"

python3 -m verl.trainer.main_ppo \
  --config-name ppo_trainer \
  algorithm.adv_estimator=gae \
  algorithm.use_kl_in_reward=True \
  algorithm.kl_penalty=kl \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef="$KL_COEF" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  trainer.n_gpus_per_node="$GPUS" \
  trainer.default_local_dir="$SAVE_DIR" \
  ${EXTRA_OVERRIDES}
