# Clip Ratio and Trust Region

Mechanisms to limit the magnitude of policy updates, preventing catastrophically large steps that destabilize training. The core stability mechanism in PPO-family algorithms.

## Mechanism

### Standard PPO Clip

Clips the importance sampling ratio r(θ) = π_θ/π_old to [1-ε, 1+ε]:

```
L = -min(r·A, clip(r, 1-ε, 1+ε)·A)
```

Default: `clip_ratio: 0.2` (i.e., ε = 0.2, ratio clamped to [0.8, 1.2]).

### DAPO Separated Clip (Clip-Higher)

DAPO decouples the lower and upper clip bounds:

```yaml
actor_rollout_ref.actor:
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
```

This allows more aggressive upward policy changes while maintaining tight downward constraints. Ratio is clamped to [1-0.2, 1+0.28] = [0.8, 1.28].

### Dual-Clip PPO

Applies an additional lower bound when advantage < 0, preventing the clipped loss from exceeding a threshold:

```yaml
actor_rollout_ref.actor:
  clip_ratio_c: 3.0   # lower bound for dual-clip
```

### DPPO (Divergence-Based Trust Region)

DPPO (Qi et al., 2026) replaces ratio-based clipping with divergence-based masking (e.g., Total Variation distance). Key insight: PPO's ratio-based clipping is fundamentally flawed for LLMs:

- **Over-penalizes low-probability tokens**: Tokens like numbers (1, 4), math symbols (+, -, =), and reasoning words (Wait, Thus, Next) are the most frequently clipped, yet they are the most important for reasoning.
- **Under-penalizes high-probability tokens**: Common tokens can have large policy changes without triggering the clip.

DPPO-Binary-TV uses TV divergence as the trust region, which is more stable for MoE models where probability ratios are highly volatile for low-probability tokens.

```bash
LOSS_MODE=dppo_tv bash examples/dppo_trainer/run_qwen3_30b_a3b_megatron.sh
```

## Diagnostic Relevance

**Metrics to monitor:**
- `actor/clip_ratio` — fraction of tokens being clipped. If persistently > 0.1, the policy may be changing too aggressively or the clip range is too tight.
- For long responses with high clip ratios, consider increasing `max_response_length` or cleaning data.
- If training shows no improvement despite high rewards, the clip may be too tight — try clip-higher.

**Practical guidance:**
- Start with `clip_ratio: 0.2` (default)
- For DAPO-style training: `clip_ratio_low: 0.2, clip_ratio_high: 0.28`
- For MoE models or long-context training, consider DPPO's divergence-based approach

## Related Pages

- [GRPO](../entities/grpo.md) — uses standard PPO clip
- [DAPO](../entities/dapo.md) — uses clip-higher
- [Entropy Collapse](entropy-collapse.md) — clip-higher provides indirect mitigation
- [KL Divergence Control](kl-divergence-control.md) — complementary regularization mechanism

## Sources

- [verl-docs](../sources/verl-docs.md) — clip ratio configuration, DPPO trust region analysis, DAPO clip-higher
