# Entropy Collapse

Sharp decline in policy entropy during RL training, leading to overconfidence, loss of exploration ability, and performance saturation. The primary bottleneck for scaling RL on LLMs.

## Mechanism

Cui et al. (2025) establish an empirical relationship: **R = −a·exp(H) + b**, where R is performance and H is entropy. Performance is bottlenecked by entropy exhaustion — once entropy drops below a threshold, no further learning occurs regardless of training compute.

The theoretical explanation: entropy change is driven by the **covariance between action probability and logit updates**, which correlates with advantage in Policy Gradient methods:

- High-probability, high-advantage tokens **reduce** entropy (the policy becomes more confident)
- Rare, high-advantage tokens **increase** entropy (the policy explores more)

Empirically, the covariance term remains positive throughout training, causing entropy to decline monotonically.

### Why It Matters for Long-CoT

In long chain-of-thought scenarios, entropy collapse is especially damaging because:
- The model loses the ability to explore alternative reasoning paths
- Response diversity collapses — all outputs converge to a single pattern
- Performance plateaus even as training continues

## Diagnostic Relevance

**Metrics to monitor:**
- `actor/entropy` — should not decline monotonically to near-zero
- Response length distribution — collapse often manifests as length uniformity
- Reward variance across group — zero variance means all responses are identical

**When to worry:** If entropy at plateau is orders of magnitude lower than initial entropy, the model has likely exhausted its exploration budget.

## Mitigation Strategies

### 1. Clip-Cov

Restricts updates for high-covariance tokens (those most responsible for entropy reduction). Directly targets the mechanism driving collapse.

### 2. KL-Cov

Uses KL divergence to restrict updates for high-covariance tokens. Provides similar entropy preservation with different regularization properties.

### 3. Clip-Higher (DAPO)

Asymmetric clip ratios (`clip_ratio_low: 0.2`, `clip_ratio_high: 0.28`) allow upward exploration while constraining downward updates. Not specifically designed for entropy but indirectly helps by allowing more aggressive policy changes.

### 4. Entropy Coefficient

`actor_rollout_ref.actor.entropy_coeff` adds explicit entropy bonus to the loss. Default changed to 0.0 since verl v0.3.x. Can be set > 0 to directly prevent entropy collapse, but requires careful tuning — too large disrupts learning.

### 5. KL Regularization

[KL divergence control](kl-divergence-control.md) against a reference policy indirectly prevents entropy collapse by constraining the policy to stay near its initialization.

### Empirical Results (Clip-Cov and KL-Cov)

| Method | Model | AIME24 | AIME25 | Avg (7 benchmarks) |
|--------|-------|--------|--------|---------------------|
| GRPO | Qwen2.5-7B | 21.2 | 9.6 | 38.6 |
| + Clip-Higher | Qwen2.5-7B | 18.1 | 11.5 | 38.8 |
| + Clip-Cov | Qwen2.5-7B | 22.1 | 15.8 | 40.4 |
| + KL-Cov | Qwen2.5-7B | 22.6 | 12.9 | 40.6 |
| GRPO | Qwen2.5-32B | 21.8 | 16.2 | 45.8 |
| + KL-Cov | Qwen2.5-32B | 36.8 | 30.8 | 52.2 |

Gains are more pronounced at larger scale (6.4% avg improvement for 32B vs 2.0% for 7B).

## Related Pages

- [GRPO](../entities/grpo.md) — primary algorithm affected by entropy collapse
- [KL Divergence Control](kl-divergence-control.md) — indirect mitigation via policy regularization
- [Clip Ratio and Trust Region](clip-ratio-and-trust-region.md) — clip-higher as partial mitigation
- [Loss Aggregation](loss-aggregation.md) — token-mean avoids compounding entropy reduction

## Sources

- [verl-docs](../sources/verl-docs.md) — entropy mechanism paper, Clip-Cov/KL-Cov implementation and results
