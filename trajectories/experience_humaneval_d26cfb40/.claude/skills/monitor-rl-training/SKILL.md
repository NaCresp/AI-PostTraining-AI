---
name: monitor-rl-training
description: Runtime metrics and thresholds for healthy LLM RL (entropy, rollout prob diff, grad_norm, group reward std). Use while training is running.
---

# Monitor RL training

## Metrics table

| Metric | Healthy | Investigate when |
|--------|---------|------------------|
| Policy entropy | Stable band, not monotonic crash to ~0 | → `triage-training-collapse` |
| Rollout vs train prob diff (`rollout_probs_diff_mean` or equivalent) | < 0.005 | 0.005–0.01 borderline; **> 0.01** severe mismatch |
| `grad_norm` | Stable / slow drift | Sustained rise → mismatch or LR too high |
| Within-group reward std | > 0 | = 0 → identical completions, weak GRPO signal |
| Response length std | Has variance | Uniform length → late entropy collapse |

## Procedure

1. Log metrics every step (or every N steps) to a file; tail the log — do not flood agent context with full training stdout.
2. On anomaly, classify symptom first (`triage-training-collapse`), then apply **one** fix.
3. After weight sync or checkpoint resume, spot-check generations — silent wrong weights happen.

## Verification

- You can point to which metric triggered an intervention.
- Post-fix run shows movement on the targeted metric within ~10 steps.