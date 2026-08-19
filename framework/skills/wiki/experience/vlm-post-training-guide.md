# VLM Post-Training Guide

Vision-language SFT/RL. General alignment rules: [RL Training Principles](rl-training-principles.md).

## Checklist

1. **Versions:** match `transformers` / flash-attn to model family (Qwen3-VL, Gemma, etc.).
2. **Attention:** FlashAttention-2 with packing; avoid SDPA + remove-padding combos that break.
3. **Never** fused kernels + Ulysses SP > 1 — silent quality regression (no crash).
4. **Dynamic batching:** pass shared `dp_group` or disable — variable vision tokens cause rank skew / NCCL hang.
5. **Multimodal batches:** detect real image inputs, not `vision_config` alone.
6. **Data path:** token-in-token-out through vLLM/SGLang; client-side chat template in agent loops — avoid re-tokenizing raw prompts.
7. **Filters:** `pop()` not `get()` for image/video columns in dataset maps.
8. **Post sync:** validate generations + `rollout_probs_diff_mean` after weight sync.

## verl-Specific

Engine choice (VeOmni, verl-omni), device-placement bugs, PR #6127 nested-tensor jagged dim — see [VLM Post-Training](../concepts/vlm-post-training.md), [verl Known Bugs](verl-known-bugs.md).

## Derived From

- [VLM Post-Training](../concepts/vlm-post-training.md), [FSDP Training](../concepts/fsdp-training.md)
