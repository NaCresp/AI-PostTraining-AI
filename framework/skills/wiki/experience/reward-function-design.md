# Reward Function Design

Stable rewards for RL. GRPO scale / KL interaction: [RL Training Principles](rl-training-principles.md).

## Steps

### 1. Choose Type

| Scenario | Type |
|----------|------|
| Verifiable math | Rule-based 0/1 |
| Math + reasoning quality | Rule + GenRM (FAPO) |
| Open alignment | Model RM |
| Tools | Outcome + optional process signal |

### 2. Rule-Based Math

- Extract final answer from **last ~300 chars** for `\boxed{}` (ignore intermediate boxes in CoT).
- Normalize strings before compare.
- Return dict with `score` + metrics used for dynamic sampling (`acc`, etc.).

### 3. GenRM (Flawed Positives)

If rule says correct, GenRM can zero reward when reasoning is flawed. Prefer **external** vLLM/SGLang service at scale.

### 4. Overlong Shaping

Linear penalty approaching `max_response_length` — smoother than hard cut.

### 5. Multi-Source Routing

Every `data_source` value must map to a handler — missing routes **fail silently**.

### 6. Scale vs KL

Binary 0/1 pairs with KL coef ~0.001; rescale KL if reward range is much larger.

## Key Pitfalls

- Flawed positives, reward hacking, sparse-only signal — see [Reward Design](../concepts/reward-design.md), [FAPO](../entities/fapo.md).

## Derived From

- [Reward Design](../concepts/reward-design.md), [KL Divergence Control](../concepts/kl-divergence-control.md)
