---
name: posttraining-known-pitfalls
description: Quick reference for verl silent failures, crashes, and version coupling. Use when training runs without error but metrics or rewards are wrong, or on NCCL/dtype/export errors with verl.
---

# Post-training known pitfalls

**verl-only.** For generic RL symptoms see `diagnose-silent-rl-failures`.

## Silent correctness (training runs, wrong results)

| Symptom | Likely cause | Mitigation |
|---------|--------------|------------|
| Rewards always zero | `transformers≥5` `apply_chat_template` returns `BatchEncoding` not `list[int]` | Normalize token ids or pin transformers<5 |
| clipfrac >40%, inflated KL at step 0 | `use_remove_padding=False` + FSDP2 mask bug | `use_remove_padding=True` |
| LoRA trained, eval = base | Merged `lora_alpha=0` in export | Assert `adapter_config.json` matches training alpha |
| VLM quality drop, no error | Fused kernels + Ulysses SP>1 | Disable fused kernels with SP>1 |

## Crashes / hangs

| Symptom | Mitigation |
|---------|------------|
| NCCL hang, no Python traceback | Dynamic batching: all ranks need same split (`dp_group`); sort buffer names before broadcast on new archs |
| CPU/CUDA `state_dict` on weight sync | Avoid full CPU offload policy during `update_weights` |
| NaN `grad_norm` | Do not force fp16 weights; bf16 compute + fp32 master |
| FP8 weight sync shape errors | FP8 rollout sync often not plug-and-play — match quant paths or avoid |

## Performance / stability

| Issue | Workaround |
|-------|------------|
| `rollout_probs_diff_mean` > 0.01 on non-Hopper | Disable vLLM cascade attention |
| CUDA graph + DAPO/LoRA degradation | `enforce_eager=True` |
| Colocated vLLM wrong weights | Isolate IPC/socket per job ID |
| DAPO hard dynamic masks | All-zero/all-one filters — soften thresholds or batch size |

## Version coupling

Treat **`transformers` major bumps** as full regression: tokenizer API, RoPE helpers, nested tensors.

## Procedure

1. Match symptom to table — fix **one** item.
2. Spot-check generations after checkpoint load or weight sync (0% acc often = stale rollout weights).
3. Log fix in `experiment.jsonl`; if reproducible for your stack, extend this skill via `create-skill`.

## Related skills

- `data-parquet-schema`, `grpo-lora-config`, `fix-train-infer-mismatch`
