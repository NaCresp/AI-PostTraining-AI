#!/bin/bash
set -e
cd /home/ben/task

echo "Starting training v2 at $(date)"
accelerate launch --num_processes 4 --multi_gpu train_v3.py \
  --model-path /home/ben/task/model_v1_bf16 \
  --data-file /home/ben/task/training_data/train.jsonl \
  --output-dir /home/ben/task/checkpoints_v2 \
  --epochs 1.0 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 5e-6 \
  --max-length 768 \
  --save-steps 2000 \
  --logging-steps 50

echo "Training v2 finished at $(date)"
