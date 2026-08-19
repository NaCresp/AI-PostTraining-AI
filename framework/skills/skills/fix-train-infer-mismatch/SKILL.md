---
name: fix-train-infer-mismatch
description: Diagnose and fix rollout vs training log-probability mismatch (rollout_probs_diff_mean). Use when reward rises but eval drops, grad_norm drifts up, or prob diff exceeds 0.005.
---

# Fix train–inference mismatch

Rollout (vLLM/SGLang/HF generate) and training forward can disagree on token probabilities even with identical weights.

## Diagnose

1. Enable log probs on **both** rollout and training paths.
2. Track mean absolute prob diff (`rollout_probs_diff_mean` or equivalent):

| Diff | Action |
|------|--------|
| < 0.005 | Healthy |
| 0.005–0.01 | Monitor |
| > 0.01 | Fix before more RL steps |

3. Common co-symptoms: `grad_norm` rising; in-group reward up but held-out eval flat or down.

## Mitigations (try one at a time)

1. **Match precision** — same dtype path for rollout and trainer (bf16 compute + fp32 master is a safe default).
2. **vLLM cascade attention** — on non-Hopper GPUs, long CoT/reasoning models can show large diffs; disable cascade attention in vLLM engine kwargs if your stack exposes it.
3. **Known bad kernels** — fused kernels + sequence parallel on VLMs can regress quality silently; disable one axis at a time.
4. **Rollout correction** — importance sampling / rejection sampling when off-policy gap is intentional (async or precision gap).
5. **Reduce async staleness** — if using async RL, lower staleness before full IS correction.

## Verification

- Prob diff returns below 0.01 and stays stable over 10+ steps.
- Eval trend aligns with training reward direction.

## Pitfalls

- Treating mismatch as pure LR problem — halving LR without fixing engines prolongs wrong training.
- Enabling bypass mode (π_old = π_rollout) while a real precision gap exists — hides the bug.

## Related skills

- `triage-training-collapse` (step 3), `async-rl-escalation`, `posttraining-known-pitfalls`
