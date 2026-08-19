---
name: escalate-rl-algorithm
description: When to move beyond baseline GRPO (DAPO, DrGRPO, DPPO, FAPO, distillation). Use after stable GRPO stalls or shows specific failure modes.
---

# Escalate RL algorithm

Escalate **one** innovation at a time.

## Selection table

| Situation | Consider |
|-----------|----------|
| First math RL run | GRPO + KL |
| Entropy collapse / plateau | DAPO (clip-higher, dynamic sampling, overlong buffer) |
| Runaway response length | DrGRPO loss normalization |
| MoE / rare-token instability | DPPO (divergence mask vs ratio clip) |
| Correct answer, bad reasoning | FAPO / GenRM |
| Strong frozen teacher | On-policy distillation |
| Tools / sandbox agents | GRPO + multi-turn pipeline |

## Brief notes

- **DAPO** — Asymmetric clip, no KL loss; dynamic filter of all-correct/all-wrong groups.
- **DrGRPO** — Fixes length bias from GRPO aggregation.
- **FAPO** — Deploy GenRM as separate inference service at scale.
- **OPD** — Batched weight sync critical for throughput.

## Verification

- Baseline GRPO config and metrics captured before escalation.
- Only one algorithmic change vs previous run.

## Pitfalls

- Stacking DAPO + DrGRPO + new LR in one run.
- GenRM inside training loop at scale without external service → instability.