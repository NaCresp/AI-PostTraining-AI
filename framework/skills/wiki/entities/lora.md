# LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning method that trains low-rank adapter matrices on frozen base weights. In verl post-training pipelines it is used in both SFT and RL phases to reduce memory and checkpoint size while retaining most of full fine-tuning quality.

## Details

### Working Configuration

For GRPO+LoRA on Qwen models, a validated configuration uses `lora_rank=64` and `lora_alpha=128` (verl Issues #6380). This rank/alpha pairing appears in successful GRPO training runs where rewards increase during training.

### Checkpoint Export Fragility

LoRA checkpoint export is fragile. When converting checkpoints via `base_model_merger.py`, the saved `adapter_config.json` can record `lora_alpha=0` even though training used `lora_alpha=128` (verl Issues #6380). The merge then produces weights identical to the base model despite visible reward improvement during training. The root cause is in the merge code path: when `actor_rollout_ref.model.lora.merge=True`, PEFT regenerates adapter config without preserving training-time `lora_alpha`. **Always verify `adapter_config.json` after checkpoint conversion** before trusting merged weights.

### Megatron-Bridge vs FSDP2

Megatron-Bridge LoRA can underperform FSDP2 by approximately 10% or collapse to 0 accuracy (verl Issues #5094). The root cause is incorrect LoRA weight handling fixed in Megatron-Bridge PR #1817. Use Megatron-Bridge commit `6259ae83` or later (verl Issues #5094).

### LoRA Sync to vLLM-Omni

Current LoRA weight sync to vLLM-omni uses a monkey-patch that loads tensors inline (verl Issues #6078). An RFC proposes replacing this with tmpfs staging plus `cudaHostRegister` for more reliable and performant weight transfer (verl Issues #6078).

## Relevance

LoRA makes GRPO and SFT feasible on constrained GPU budgets, but export and backend-specific bugs can silently discard all adapter learning. Verifying adapter config after merge and choosing FSDP2 over Megatron-Bridge LoRA (unless on a fixed Bridge commit) are essential operational checks before evaluating RL checkpoints.

## Related Pages

- [verl](verl.md) — library that implements LoRA training and checkpoint merging
- [GRPO](grpo.md) — primary RL algorithm used with LoRA in verl recipes
- [FSDP Training](../concepts/fsdp-training.md) — recommended backend when Megatron-Bridge LoRA underperforms

## Sources

- [verl Issues](../sources/verl-issues.md) — GRPO+LoRA config (#6380), checkpoint export bug with `lora_alpha=0` (#6380), Megatron-Bridge underperformance and PR #1817 fix (#5094), vLLM-omni LoRA sync RFC (#6078)
