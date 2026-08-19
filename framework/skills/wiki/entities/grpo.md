# GRPO (Group Relative Policy Optimization)

Critic-free policy optimization algorithm that uses group sampling and relative reward normalization instead of a learned value function. Proposed in DeepSeekMath (2024).

## Details

GRPO generates G responses per prompt, scores them, and normalizes advantages within each group:

```
Â_i = (R_i - mean(R)) / std(R)
```

Key configuration in verl:

- `algorithm.adv_estimator: grpo`
- `actor_rollout_ref.rollout.n`: group size G (commonly 16 or 64)
- `actor_rollout_ref.actor.clip_ratio: 0.2` — PPO-style clip range
- `actor_rollout_ref.actor.use_kl_loss: True` — adds KL loss to actor (standard for GRPO)
- `actor_rollout_ref.actor.kl_loss_coef: 0.001`
- `actor_rollout_ref.actor.loss_agg_mode: token-mean` — recommended over the original paper's `seq-mean-token-mean`

Despite config keys starting with `ppo_`, they work across GRPO/DAPO/PPO in verl.

### DrGRPO Extension

DrGRPO (2025) addresses GRPO's optimization bias that artificially lengthens responses, especially incorrect ones. It normalizes token-level losses with a global constant:

- `loss_agg_mode: seq-mean-token-sum-norm`
- `loss_scale_factor`: optional constant (e.g., max response length)
- `use_kl_loss: False`
- `algorithm.norm_adv_by_std_in_grpo: False`

### Baseline Results (GSM8K)

| Model | Method | Score |
|-------|--------|-------|
| Qwen2.5-0.5B-Instruct | GRPO | 54.3 (LoRA) |
| Qwen2-7B-Instruct | GRPO | 89.0 |
| Qwen2.5-7B-Instruct | GRPO-LoRA | 93.4 |
| Qwen2.5-32B-Instruct | GRPO-LoRA | 95.8 |

## Relevance

GRPO is the dominant algorithm for math-reasoning RL training. Its simplicity (no critic) makes it computationally cheaper than PPO, while group sampling provides natural variance reduction. Understanding its failure modes — particularly [entropy collapse](../concepts/entropy-collapse.md) and [loss aggregation](../concepts/loss-aggregation.md) choices — is essential for stable training.

## Related Pages

- [DAPO](dapo.md) — extends GRPO with separated clip, dynamic sampling, token-level loss
- [Entropy Collapse](../concepts/entropy-collapse.md) — primary failure mode in GRPO training
- [Loss Aggregation](../concepts/loss-aggregation.md) — critical choice affecting training stability
- [KL Divergence Control](../concepts/kl-divergence-control.md) — regularization mechanisms
- [Clip Ratio and Trust Region](../concepts/clip-ratio-and-trust-region.md) — update magnitude control

## Sources

- [verl-docs](../sources/verl-docs.md) — algorithm specification, config reference, baseline results
