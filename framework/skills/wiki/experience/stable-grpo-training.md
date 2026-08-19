# Stable GRPO/DAPO Launch Checklist

Math-reasoning RL with GRPO or DAPO. Universal defaults live in [RL Training Principles](rl-training-principles.md); below adds algorithm-specific choices and a **verl** config reference.

## Algorithm Checklist

### GRPO baseline

- `adv_estimator`: grpo (or framework equivalent)
- Group size `n`: 8–16
- `loss_agg_mode` / loss type: **token-mean**
- KL loss on actor: enabled, coef **0.001**, type **low_var_kl**
- Clip: **0.2**
- LR: **~1e-6** (full fine-tune)
- Rollout log probs: **on** (for mismatch monitoring)

### DAPO upgrade (entropy collapse / plateau)

- Asymmetric clip: low **0.2**, high **0.28**
- **Disable** KL loss (clip-higher replaces that role)
- Dynamic sampling: filter all-correct / all-wrong groups
- `gen_batch_size` > `train_batch_size` for filter headroom
- Overlong buffer + penalty near context limit

→ [DAPO](../entities/dapo.md), [Advanced Algorithms](advanced-algorithms.md)

## Pre-Launch

- [ ] Rollout + training log-prob logging enabled
- [ ] Token-mean loss unless using DrGRPO intentionally
- [ ] Gradient checkpointing on
- [ ] SFT checkpoint evaluated on target task
- [ ] Dependencies pinned (especially `transformers`)

## Runtime

See monitoring table in [RL Training Principles](rl-training-principles.md#runtime-monitoring). On anomaly → [Training Collapse Triage](training-collapse-triage.md).

## verl Reference Config (Optional)

<details>
<summary>verl YAML snippets</summary>

```yaml
algorithm:
  adv_estimator: grpo
actor_rollout_ref:
  rollout:
    n: 16
    temperature: 1.0
    calculate_log_probs: true
  actor:
    clip_ratio: 0.2
    loss_agg_mode: token-mean
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    optim:
      lr: 1e-6
      clip_grad: 1.0
data:
  train_batch_size: 512
```

DAPO: `clip_ratio_low/high`, `use_kl_loss: false`, `filter_groups`, `overlong_buffer` — see [DAPO](../entities/dapo.md).

</details>

## Derived From

- [RL Training Principles](rl-training-principles.md)
- [GRPO](../entities/grpo.md), [DAPO](../entities/dapo.md)
- [verl](../entities/verl.md) — config naming (`_per_gpu` vs global)
