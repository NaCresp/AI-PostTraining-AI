# On-Policy Distillation Guide

Teacher → student soft targets during on-policy training. Stability baseline: [RL Training Principles](rl-training-principles.md).

## Steps

1. **Estimator:** distillation / GKD-style advantage (no critic).
2. **Engines:** teacher on inference stack (throughput); student on training stack (backward-matched log probs). Small numeric gap vs teacher targets is expected.
3. **Async:** one-/two-step-off schedulers; **batched** NCCL weight sync (~12× vs streaming) — tune bucket size if sync dominates.
4. **Padding symmetry:** teacher and student must apply the same top-k / padding path for distillation logits.
5. **Async limits:** fully-async may lack GenRM; large MoE + async + certain bypass modes → checkpoint failures (verl).

## Key Pitfalls

- Asymmetric `use_remove_padding` between teacher and student → KeyError or wrong loss.
- Plan sync path if combining distillation with model-based rewards.

## Derived From

- [On-Policy Distillation](../concepts/on-policy-distillation.md), [Async Training](../concepts/async-training.md)
