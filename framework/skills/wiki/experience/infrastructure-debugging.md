# Infrastructure Debugging Guide

For **algorithmic** collapse (entropy, KL, rewards) use [Training Collapse Triage](training-collapse-triage.md). This page covers distributed training, engines, and NCCL — not loss formulas.

## Checklist

1. **Weight sync validation** — after init or checkpoint load, run a short generation eval; 0% accuracy often means stale or dummy rollout weights.
2. **NCCL deadlock** — hangs with no Python traceback:
   - Uneven micro-batch counts across ranks (dynamic batching without shared data-parallel group).
   - Heterogeneous buffer broadcast order — sort tensor names before broadcast on new architectures.
3. **CPU offload + weight export** — `state_dict()` CPU/CUDA mismatch during sync; try param offload without full offload policy if the stack allows.
4. **OOM** — reduce inference memory fraction or rollout batch; enable gradient checkpointing on trainer.
5. **Multi-job colocation** — isolate IPC/socket paths per job ID when multiple RL jobs share a host.
6. **FSDP / wrap policy** — partial `_no_split_modules` lists can fail layer resolution after `transformers` upgrades.
7. **Dtype** — fp32 master weights + bf16 compute is the safe default; NaN `grad_norm` → check dtype/autocast before LR.
8. **Version coupling** — treat `transformers` major bumps as a full regression test (tokenizer, RoPE, nested tensors).

## Silent Failures

Symptom catalog (framework-agnostic): [RL Training Principles — Silent Failures](rl-training-principles.md#silent-failures-no-crash-wrong-training).

verl-specific table: [verl Known Bugs](verl-known-bugs.md).

## High-Risk Combinations

- Remove-padding + sequence parallel + new VLM arch — test one axis at a time.
- FP8 weight sync between trainer and vLLM — often not plug-and-play without matched quant paths.

## Derived From

- [FSDP Training](../concepts/fsdp-training.md), [Training-Inference Mismatch](../concepts/training-inference-mismatch.md)
- [Data Pipeline](../concepts/data-pipeline.md)
