# Training Collapse Triage

Step-by-step diagnosis when RL rewards stall, explode, or eval degrades. Framework-agnostic; see [RL Training Principles](rl-training-principles.md) for defaults and [verl Known Bugs](verl-known-bugs.md) for stack-specific traps.

## Step 1: Classify the Symptom

| Symptom | Likely cause | Check first |
|---------|-------------|-------------|
| Reward stalls, entropy crashes | Entropy collapse | Policy entropy curve |
| Reward up, eval down | Train–infer mismatch | Rollout vs train log-prob diff |
| `grad_norm` keeps rising | Mismatch or LR too high | Prob diff + LR |
| KL ≫ 0.1 (e.g. > 1.0) | Policy far from reference | KL metric |
| Identical length / content in group | Late entropy collapse | Length std, within-group reward std |
| Reward spike then crash | Reward hacking / grad explosion | Reward curve + grad_norm |

## Step 2: Entropy Collapse

1. Entropy falling monotonically to near zero?
2. Within-group reward variance ≈ 0?
3. Response lengths becoming uniform?

**Mitigations (in order):**

1. Use **token-mean** loss aggregation (not seq-mean-token-mean for long CoT).
2. Larger models: KL-Cov or Clip-Cov (recipe-dependent).
3. DAPO-style **clip-higher** (e.g. 0.2 / 0.28).
4. Small `entropy_coeff` only as last resort.

→ [Entropy Collapse](../concepts/entropy-collapse.md)

## Step 3: Train–Inference Mismatch

1. Log probs enabled on rollout **and** training paths?
2. Mean prob diff:
   - **< 0.005** — healthy
   - **0.005–0.01** — watch
   - **> 0.01** — act

**Mitigations:** match precision; fix known inference kernel issues; [Rollout Correction](../concepts/rollout-correction.md); reduce async staleness.

→ [Training-Inference Mismatch](../concepts/training-inference-mismatch.md)

## Step 4: KL Explosion

- Normal KL often **< 0.1**; **> 1.0** is severe drift.
- Too high → increase KL penalty coef (e.g. 0.001 → 0.01).
- Stuck at zero → coef may be too large.
- Never enable KL-in-reward **and** KL-in-loss together.

→ [KL Divergence Control](../concepts/kl-divergence-control.md)

## Step 5: Gradient Anomalies

1. Rule out mismatch (Step 3).
2. Halve learning rate.
3. Enable grad clip (~1.0).
4. Check rewards for NaN/Inf.

## Step 6: Still Stuck

1. Reproduce with a known-good public recipe config.
2. Diff configs line by line.
3. Restore **one** change per run.

## Step 7: Silent Correctness

If loss runs but metrics smell wrong → [RL Training Principles — Silent Failures](rl-training-principles.md#silent-failures-no-crash-wrong-training) and framework bug lists.

## Key Pitfalls

- Do not disable rollout log-prob logging — it is the main mismatch diagnostic.
- Collapse and mismatch both raise `grad_norm` — use prob diff and entropy to separate them.
- Do not batch-fix with many hyperparameter changes at once.

## Derived From

- [Entropy Collapse](../concepts/entropy-collapse.md), [Training-Inference Mismatch](../concepts/training-inference-mismatch.md)
- [KL Divergence Control](../concepts/kl-divergence-control.md), [Rollout Correction](../concepts/rollout-correction.md)
- [Clip Ratio and Trust Region](../concepts/clip-ratio-and-trust-region.md)
