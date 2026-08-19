#!/bin/bash
set -e
cd /home/ben/task

echo "Starting training v3 at $(date)"
accelerate launch --num_processes 4 --multi_gpu train_v3.py \
  --model-path /home/ben/task/model_v2_bf16 \
  --data-file /home/ben/task/training_data/train.jsonl \
  --output-dir /home/ben/task/checkpoints_v3 \
  --epochs 1.0 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-6 \
  --max-length 768 \
  --save-steps 5000 \
  --logging-steps 100

echo "Training v3 finished at $(date)"
