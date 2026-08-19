# Advanced RL Algorithms Beyond GRPO

When baseline GRPO is insufficient. Defaults and monitoring: [RL Training Principles](rl-training-principles.md). Escalate **one innovation at a time**.

## DAPO

Clip-higher, dynamic group filtering, token-mean loss, overlong penalty. **No KL loss** — exploration via asymmetric clip.

→ [DAPO](../entities/dapo.md)

## DrGRPO

Fixes length bias from GRPO aggregation; use when responses grow unbounded without reward gain.

→ [Loss Aggregation](../concepts/loss-aggregation.md)

## DPPO

Divergence-based masking instead of ratio clip — MoE, long context, rare reasoning tokens.

## FAPO

GenRM flags correct answers with flawed reasoning; deploy judge as **separate** inference service at scale.

→ [FAPO](../entities/fapo.md), [Reward Function Design](reward-function-design.md)

## On-Policy Distillation

Teacher soft targets; batched weight sync critical for throughput.

→ [Distillation Guide](distillation-guide.md), [On-Policy Distillation](../concepts/on-policy-distillation.md)

## Selection Table

| Situation | Try |
|-----------|-----|
| First math RL | GRPO + KL |
| Collapse / plateau | DAPO |
| Runaway length | DrGRPO |
| MoE ratio noise | DPPO |
| Reasoning quality | FAPO |
| Strong teacher | OPD |
| Tools / sandbox | GRPO + multi-turn |
