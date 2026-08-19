# KL Divergence Control

Mechanisms to prevent the trained policy from diverging too far from a reference policy. Essential for training stability — without KL control, the policy can drift into degenerate regions.

## Mechanism

Two orthogonal approaches in verl, which should not be used simultaneously:

### KL Loss (Added to Actor Loss)

Directly adds KL divergence between actor and reference policy to the training loss:

```yaml
actor_rollout_ref.actor:
  use_kl_loss: True           # enable KL loss
  kl_loss_coef: 0.001         # weight of KL term
  kl_loss_type: low_var_kl    # estimator type (k3)
```

**Recommended for GRPO.** Not used in DAPO (which relies on clip-higher instead).

### KL Penalty in Reward

Adds a KL penalty term to the reward signal before advantage computation:

```yaml
algorithm:
  use_kl_in_reward: True
  kl_penalty: kl              # estimator type
  kl_ctrl:
    type: fixed                # or adaptive
    kl_coef: 0.005
    horizon: 10000             # for adaptive controller
    target_kl: 0.1             # for adaptive controller
```

**Used in PPO.** The adaptive controller dynamically adjusts the coefficient to maintain KL near `target_kl`.

### KL Estimator Types

| Name | Formula | Properties |
|------|---------|------------|
| `kl` / `k1` | E[log(π_old/π_θ)] | Standard KL, can be negative |
| `abs` | E[|log(π_old/π_θ)|] | Absolute, always positive |
| `mse` / `k2` | E[0.5·(log(π_old/π_θ))²] | Symmetric quadratic |
| `low_var_kl` / `k3` | E[exp(log_ratio) - log_ratio - 1] | Non-negative per token, lower variance |
| `full` | Full distributional KL | Most accurate, most expensive |

**Practical recommendation:** `low_var_kl` (k3) is generally preferred — it's non-negative per token (more stable) and provides lower-variance gradient estimates than k1.

**Straight-through variants** (k1+, k3+): Append "+" to use k2 for unbiased gradient estimation while displaying the chosen KL value. This decouples the displayed metric from the gradient signal.

### Coefficient Tuning

- `kl_loss_coef: 0.001` — standard starting point for GRPO
- Larger values curb reward hacking but reduce exploration
- If KL grows unbounded during training, increase the coefficient
- If training stalls with no reward improvement, decrease the coefficient

## Diagnostic Relevance

**Metrics to monitor:**
- `actor/kl` — KL divergence between actor and reference
- If KL explodes (> 1.0), the policy has diverged significantly from initialization
- If KL stays near zero, the KL coefficient may be too strong (preventing learning)

**Reference model activation:** The reference model is only loaded when `use_kl_loss=True` or `use_kl_in_reward=True`. For models >7B, enable parameter offload for the reference model (`actor_rollout_ref.ref.fsdp_config.param_offload: True`).

## Related Pages

- [GRPO](../entities/grpo.md) — uses KL loss by default
- [DAPO](../entities/dapo.md) — disables KL, uses clip-higher instead
- [Entropy Collapse](entropy-collapse.md) — KL control indirectly prevents entropy collapse
- [Clip Ratio and Trust Region](clip-ratio-and-trust-region.md) — complementary stability mechanism

## Sources

- [verl-docs](../sources/verl-docs.md) — KL configuration reference, estimator implementations, PPO/GRPO usage
