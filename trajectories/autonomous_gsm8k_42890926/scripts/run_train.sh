#!/bin/bash
set -e
cd /home/ben/task

echo "Starting training at $(date)"
accelerate launch --num_processes 4 --multi_gpu train_v2.py \
  --data-file /home/ben/task/training_data/train.jsonl \
  --output-dir /home/ben/task/checkpoints_v1 \
  --epochs 1.0 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-5 \
  --max-length 768 \
  --save-steps 2000 \
  --logging-steps 50

echo "Training finished at $(date)"
