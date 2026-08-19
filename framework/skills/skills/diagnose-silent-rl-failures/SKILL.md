---
name: diagnose-silent-rl-failures
description: Training runs without crash but wrong behavior (zero reward, eval mismatch, LoRA export, NCCL). Use when metrics look OK but results are wrong.
---

# Diagnose silent RL failures

## Symptom catalog

| Symptom | Likely cause |
|---------|----------------|
| Rewards stuck at zero | Tokenizer/chat-template API change; reward routing bug |
| Metrics OK, eval bad | Train–infer mismatch; wrong attention mask / padding |
| VLM quality drop, no error | Fused kernels + sequence parallel; retokenization on images |
| LoRA trained but eval = base | Exported `lora_alpha=0` or merge dropped adapter scale |
| NCCL hang, no traceback | Dynamic batching shards differ across ranks without shared `dp_group` |
| 0% accuracy right after weight sync | Stale or dummy rollout weights |

## Infrastructure checks

1. After init or checkpoint load, short generation eval.
2. NCCL deadlock: uneven micro-batch counts; sort tensor names before broadcast on new archs.
3. Dtype: bf16 compute + fp32 master weights; NaN `grad_norm` → check dtype before LR.
4. `transformers` major bump → full regression on tokenizer and RoPE.

## Procedure

1. Confirm reward function on a **single** known completion (should not be always 0).
2. Compare rollout vs training log-prob diff on same tokens.
3. If using verl, search known issues for your stack version (tokenizer API, async RM, dynamic sampling).

## Pitfalls

- Tuning LR when the bug is reward routing or tokenizer API.
- High-risk combo: remove-padding + sequence parallel + new VLM arch — test one axis at a time.