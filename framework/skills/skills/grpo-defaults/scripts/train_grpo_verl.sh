#!/usr/bin/env bash
set -euo pipefail

# Reference script: verl GRPO launch with sane defaults.
# Edit args and EXTRA_OVERRIDES; avoid changing many algorithm knobs at once.

usage() {
  cat <<'USAGE'
Usage:
  train_grpo_verl.sh \
    --train_file <train.parquet> \
    --val_file <val.parquet> \
    --model_path <base_or_sft_model> \
    --save_dir <checkpoint_dir> \
    [--gpus 4] [--lr 1e-6] [--rollout_n 8] [--clip_ratio 0.2]

Optional env:
  EXTRA_OVERRIDES='actor_rollout_ref.rollout.temperature=0.9 data.train_batch_size=512'
USAGE
}

TRAIN_FILE=""
VAL_FILE=""
MODEL_PATH=""
SAVE_DIR=""
GPUS=4
LR="1e-6"
ROLLOUT_N=8
CLIP_RATIO="0.2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_file) TRAIN_FILE="$2"; shift 2 ;;
    --val_file) VAL_FILE="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --save_dir) SAVE_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
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
  algorithm.adv_estimator=grpo \
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
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio="$CLIP_RATIO" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  trainer.n_gpus_per_node="$GPUS" \
  trainer.default_local_dir="$SAVE_DIR" \
  ${EXTRA_OVERRIDES}
