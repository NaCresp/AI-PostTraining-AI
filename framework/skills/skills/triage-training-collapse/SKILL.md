---
name: triage-training-collapse
description: Step-by-step diagnosis when RL rewards stall, entropy collapses, KL explodes, or eval degrades. Framework-agnostic. Use when GRPO/PPO training looks unhealthy.
---

# Training collapse triage

Change **one** mitigation per retry.

## Step 1: Classify symptom

| Symptom | Likely cause | Check first |
|---------|-------------|-------------|
| Reward stalls, entropy crashes | Entropy collapse | Policy entropy curve |
| Reward up, eval down | Train–infer mismatch | Rollout vs train log-prob diff |
| `grad_norm` keeps rising | Mismatch or LR too high | Prob diff + LR |
| KL ≫ 0.1 (e.g. > 1.0) | Policy far from reference | KL metric |
| Identical length/content in group | Late entropy collapse | Length std, within-group reward std |
| Reward spike then crash | Reward hacking / grad explosion | Reward curve + grad_norm |

## Step 2: Entropy collapse

1. Entropy falling monotonically to near zero?
2. Within-group reward variance ≈ 0?
3. Response lengths becoming uniform?

**Mitigations (in order):** token-mean loss → larger-model KL-Cov/Clip-Cov if available → DAPO clip-higher (0.2 / 0.28) → small `entropy_coeff` only as last resort.

## Step 3: Train–inference mismatch

1. Log probs enabled on rollout **and** training?
2. Mean prob diff: **< 0.005** healthy; **0.005–0.01** watch; **> 0.01** act.

**Mitigations:** match precision; fix known bad inference kernels; rollout correction; reduce async staleness.

## Step 4: KL explosion

- Normal KL often **< 0.1**; **> 1.0** severe.
- Too high → increase KL penalty coef (e.g. 0.001 → 0.01).
- Stuck at zero → coef may be too large.
- Never KL-in-reward **and** KL-in-loss together.

## Step 5: Gradients

1. Rule out mismatch (step 3).
2. Halve learning rate.
3. Enable grad clip (~1.0).
4. Check rewards for NaN/Inf.

## Step 6: Still stuck

Reproduce with a known-good public recipe; diff configs; **one** change per run.

## Step 7: Silent correctness

If loss runs but metrics smell wrong → skill `diagnose-silent-rl-failures`.

## Pitfalls

- Collapse and mismatch both raise `grad_norm` — use prob diff and entropy to separate them.
- Do not batch-fix with many hyperparameter changes.