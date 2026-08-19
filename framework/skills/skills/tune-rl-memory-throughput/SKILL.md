---
name: tune-rl-memory-throughput
description: Increase RL training throughput via micro-batch, dynamic batching, packing, and GPU split without changing algorithm hyperparameters. Use when GPU util is low or steps are slow but metrics are healthy.
---

# Tune RL memory and throughput

Tune **performance** knobs only — see `rl-experiment-discipline` for algorithm knobs (LR, clip, n, KL).

## Checklist

### 1. Actor memory

- Enable **gradient checkpointing**.
- Increase per-GPU micro-batch until OOM, then back off one step.

### 2. Forward-only stages

Reference / rollout log-prob and critic forward need no backward graph — often **~2×** actor micro-batch.

### 3. Dynamic batching (variable length)

- Cap **max tokens per GPU**; raise until OOM.
- Rule of thumb: cap ≥ **2 × (max_prompt + max_response)**.
- Distributed: all ranks must agree on splits (missing `dp_group` → NCCL deadlock).

### 4. Sequence packing

- Remove padding waste on skewed length distributions — often **50–70%** compute recovered.
- Some stacks allow packing **or** dynamic batching, not both.

### 5. Activation offload

- Trade PCIe for memory when FSDP-style training is memory-bound; usually needs checkpointing.

### 6. GPU split (colocated / hybrid)

| Observation | Action |
|-------------|--------|
| Trainer idle | More rollout GPUs |
| Rollout idle | More training GPUs |

Target: rollout wall time ≈ training wall time.

## Verification

- Step time drops; **entropy, prob diff, reward trends unchanged** in direction.
- No new NCCL hangs after dynamic batching changes.

## Pitfalls

- Raising global `train_batch_size` or `n` thinking it is “throughput only” — changes algorithm.
- Async before sync baseline is healthy.

## Related skills

- `rl-experiment-discipline`, `async-rl-escalation`, `posttraining-known-pitfalls`
