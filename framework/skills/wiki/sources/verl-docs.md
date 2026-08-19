# verl Documentation

- **Type**: documentation
- **Raw location**: `raw/verl-docs/`
- **Ingested**: 2026-05-19

## Key Takeaways

1. verl supports multiple RL algorithms (PPO, GRPO, DAPO, DPPO, GPG, OPO, RLOO, SPPO, SPIN) with a unified config system, but GRPO/DAPO are the primary recipes for math-reasoning RL.
2. Entropy collapse is a fundamental bottleneck — the Entropy Mechanism paper (Cui et al., 2025) shows R = −a·exp(H) + b, establishing that performance is capped by entropy exhaustion. Two mitigation strategies: Clip-Cov and KL-Cov.
3. Loss aggregation mode is critical: `token-mean` is recommended over `seq-mean-token-mean` (original GRPO), which can be unstable for long-CoT. DrGRPO further removes length bias by normalizing with a global constant.
4. Training-inference mismatch (precision, backend differences between rollout and training engines) is a common and under-recognized cause of RL collapse. Diagnose with `rollout_probs_diff_mean` — should be < 0.005.
5. Rollout Correction provides a unified framework for off-policy problems: importance sampling weights + rejection sampling, with multiple aggregation levels (token, sequence, geometric). The key insight: naive PPO implementations incorrectly assume π_old = π_rollout.
6. DAPO introduces four key innovations: separated clip ratios (clip-higher), dynamic sampling with group filtering, token-level loss aggregation, and overlong reward shaping.
7. DPPO shows PPO's ratio-based clipping over-penalizes low-probability tokens (numbers, math symbols, reasoning words) and under-penalizes high-probability tokens — TV divergence provides a more principled trust region.
8. Async training (one-step off-policy, fully async) can achieve 1.5-2.7x speedup by decoupling rollout and training, with staleness_threshold < 1 recommended to balance speed vs accuracy.
9. Default `entropy_coeff` changed to 0.0 since verl 0.3.x — explicit entropy bonus is no longer default.
10. Known bug: vLLM cascade attention on non-Hopper GPUs causes precision mismatch; workaround: `disable_cascade_attn=True`.
11. CUDA graph (`enforce_eager=False`) can cause model performance degradation — cause still under investigation.

## Pages Created or Updated

- [GRPO](../entities/grpo.md) — algorithm details, configuration, DrGRPO extension
- [DAPO](../entities/dapo.md) — four key innovations, configuration, reproduction results
- [verl](../entities/verl.md) — library overview, supported algorithms, architecture
- [Entropy Collapse](../concepts/entropy-collapse.md) — mechanism, R-H relationship, Clip-Cov/KL-Cov mitigation
- [KL Divergence Control](../concepts/kl-divergence-control.md) — KL loss vs KL in reward, estimator types, coefficient tuning
- [Loss Aggregation](../concepts/loss-aggregation.md) — token-mean vs seq-mean, DrGRPO, length bias
- [Clip Ratio and Trust Region](../concepts/clip-ratio-and-trust-region.md) — PPO clip, DAPO clip-higher, DPPO divergence-based
- [Rollout Correction](../concepts/rollout-correction.md) — IS weights, rejection sampling, off-policy correction framework
- [Training-Inference Mismatch](../concepts/training-inference-mismatch.md) — causes, diagnosis, mitigations
- [Batch Size Tuning](../concepts/batch-size-tuning.md) — train/mini/micro relationships, dynamic batching
- [Async Training](../concepts/async-training.md) — one-step off-policy, fully async, staleness control

## Notes

- Ignored all hardware-specific content (Ascend NPU, AMD MI300, CUDA driver details).
- Ignored environment setup (install, Docker, multi-node Ray cluster configuration).
- Focused exclusively on algorithmic knowledge and engineering practices for stable RL training.
- The rollout correction documentation is exceptionally detailed and represents a significant body of engineering knowledge about off-policy problems in LLM-RL.
