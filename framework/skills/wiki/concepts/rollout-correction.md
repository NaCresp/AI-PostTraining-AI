# Rollout Correction

Unified framework for handling off-policy problems in RL training — any scenario where the data collection distribution differs from the training distribution. Developed in response to the insight that naive PPO implementations incorrectly assume π_old = π_rollout.

## Mechanism

### The Core Problem

In LLM-RL, three distinct policies exist:

1. **π_rollout** (behavior policy): the inference engine that generates responses (e.g., vLLM in BF16)
2. **π_old** (proximal policy): the reference for PPO clipping
3. **π_θ** (current policy): being updated

Most implementations incorrectly treat π_old as the behavior policy, ignoring that π_rollout ≠ π_old due to precision differences (FP8/BF16/FP32), different backends (vLLM vs SGLang vs FSDP), or model staleness in async training. This causes RL collapse.

### Two Distribution Shifts

- **Drift 1: π_rollout → π_old** (off-policy gap): corrected via importance sampling
- **Drift 2: π_old → π_θ** (policy update drift): corrected via PPO clipping

### Correction Mechanisms

**1. Importance Sampling (IS) Weights** — continuous reweighting:
- Token-level: ρ_t = π_old(t)/π_rollout(t), each token independently bounded
- Sequence-level: ρ_seq = ∏ρ_t, product bounded

**2. Rejection Sampling (RS)** — binary filtering:
- Rejects tokens/sequences with IS ratios outside [lower, upper]
- Modifies response_mask to exclude rejected samples

**3. Geometric Mean RS** — length-invariant rejection:
- Solves the "Length Trap" where long sequences are systematically rejected because ratio products explode
- ρ_geo = (∏ρ_t)^(1/T), normalized by sequence length
- Typical bounds: "0.999_1.001" (~±0.1%)

### Operating Modes

| Mode | Policies | Speed | Properties |
|------|----------|-------|------------|
| Decoupled | 3 (π_rollout, π_old, π_θ) | Standard | Batch size invariant, separate drift correction |
| Bypass (PPO-clip) | 2 (π_old = π_rollout) | Fast | Skips old_log_prob computation |
| Bypass (REINFORCE) | 2 | Fast | Explicit IS weights, no PPO clipping |

### Recommended Workflow

1. **Start with metrics only** — enable `calculate_log_probs: true`, monitor `rollout_corr/kl` and `rollout_corr/log_ppl_abs_diff`
2. **Enable rejection sampling** if outlier fraction is high
3. **Enable full IS correction** once comfortable with metrics

### Health Check Thresholds

- `rollout_is_mean` should be close to 1.0 (< 0.5 or > 2.0 = warning)
- `rollout_is_eff_sample_size` should be > 0.3
- `rollout_corr/kl` magnitude should be < 0.1
- `chi2_token` > 1.0 indicates severe distribution shift

## Diagnostic Relevance

**When to enable rollout correction:**
- Actor `grad_norm` continuously increases during training
- Training produces good rewards but poor test-time performance
- Using async training with stale rollout data
- Using different precision/backend for rollout vs training
- Training on replay buffer data

**Key config:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: token        # or sequence, null
    rollout_is_threshold: 2.0
    rollout_rs: null         # or token_k1, seq_mean_k1, seq_mean_k3
    rollout_rs_threshold: null
    bypass_mode: true        # skip old_log_prob for speed
actor_rollout_ref:
  rollout:
    calculate_log_probs: true  # required!
```

## Related Pages

- [Training-Inference Mismatch](training-inference-mismatch.md) — the primary cause of off-policy drift
- [Async Training](async-training.md) — where rollout correction is most critical
- [GRPO](../entities/grpo.md) — affected by incorrect π_old assumption
- [Clip Ratio and Trust Region](clip-ratio-and-trust-region.md) — PPO clipping handles Drift 2

## Sources

- [verl-docs](../sources/verl-docs.md) — comprehensive rollout correction documentation, mathematical formulations, preset configurations
