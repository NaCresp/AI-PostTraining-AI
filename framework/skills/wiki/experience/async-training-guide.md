# Async Training Guide

Overlap rollout and training to cut idle GPUs (~1.5–2.7× in large DAPO runs). Prerequisites: [RL Training Principles](rl-training-principles.md) (sync baseline healthy first).

## When to Use

- Rollout **> ~50%** of step time (`trainer/idle_ratio` high)
- Long CoT or multi-turn tool traces
- Large clusters where pipeline bubbles hurt

## Option 1: One-Step Off-Policy (~23–40%)

Disaggregate GPUs: more for training, fewer for rollout. Train on previous step’s trajectories while generating the next. Low risk; quality near sync.

## Option 2: Fully Async (~1.5–2.7×)

- **Staleness cap** ~0.5 as starting point (0 = sync; 1.0 ≈ one-step off).
- **`require_batches` > 1** for stability before chasing max speed.
- **Partial rollout** — interrupt in-flight generations cleanly at weight sync; required for multi-turn tools.

## Resource Allocation

Rollout duration ≈ training duration. See [Performance Tuning § GPU Split](performance-tuning.md#7-gpu-split).

## Off-Policy Drift

Enable importance-sampling / rejection correction when staleness > 0. Health signals: IS mean ≈ 1, effective sample size > 0.3, correction KL < 0.1.

→ [Rollout Correction](../concepts/rollout-correction.md), [Async Training](../concepts/async-training.md)

## Framework Caveats

| Issue | Stacks affected |
|-------|-----------------|
| GenRM / LLM judge in fully-async | verl (hardcoded off) |
| MoE checkpoint save + async + no bypass | verl Megatron |
| flash_attn varlen + remove-padding | upgrade `transformers` 5.4+ |
| Partial rollout + tools | use async-safe agent loop |

## Key Pitfalls

- `require_batches=1` can destabilize ordering-sensitive training.
- Weight sync latency matters most for small batches.
- Do not enable async before sync path passes [collapse triage](training-collapse-triage.md) metrics.

## Derived From

- [Async Training](../concepts/async-training.md), [Colocated vs Disaggregated](../concepts/colocated-vs-disaggregated.md)
