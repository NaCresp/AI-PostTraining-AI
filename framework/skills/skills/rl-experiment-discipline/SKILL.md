---
name: rl-experiment-discipline
description: Change one RL hyperparameter at a time; separate throughput tuning from algorithm knobs. Use when planning ablations or debugging regressions.
---

# RL experiment discipline

## Golden rule

Change **one** hyperparameter or feature per experiment. Otherwise root cause is unknowable.

## Safe for throughput (usually no algorithm change)

- Per-GPU micro-batch until OOM, then back off one step
- Max tokens per GPU / dynamic batching caps
- Sequence packing (where supported)
- Activation offload, rollout GPU count
- Gradient checkpointing on actor

## Affects convergence — one at a time

- Global train batch, mini-batch, PPO/GRPO epochs
- Learning rate, group size `n`, clip range, KL coef
- Loss aggregation mode (token-mean vs seq-mean-token-mean)

## Async escalation

Only after healthy **synchronous** baseline:

1. Sync on-policy (or one-step-off)
2. One-step off-policy if rollout > ~50% of step time
3. Fully async with staleness caps and correction

## Verification

- Experiment log states single hypothesis and single changed knob.
- Throughput changes did not coincide with LR/clip/KL changes in the same run.

## Pitfalls

- Tuning micro-batch and learning rate together → ambiguous outcome.
- Skipping sync baseline before async → harder debugging surface.