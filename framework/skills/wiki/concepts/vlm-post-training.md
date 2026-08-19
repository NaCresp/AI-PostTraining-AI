# VLM Post-Training

Vision-Language Model post-training requires special handling for multi-modal inputs (images, video) beyond standard text-only RL or SFT. Position encoding, tensor placement, and sequence packing all introduce failure modes absent in text-only pipelines.

## Mechanism

### RoPE and Position ID Fragility

VLM training is sensitive to RoPE and `position_ids` handling: qwen2_vl utilities, nested tensor jagged dimension flips, and mRoPE 3D tensors all cause shape mismatches when misconfigured (verl Issues #4483, #6073).

### Vision Tensor Device Placement

With Qwen3-VL + FSDP, vision tensors can be placed on CPU while the model runs on CUDA for the policy (but not reference) path (verl Issues #4906). Fix is upstream in transformers PR #41536 (verl Issues #4906).

### Fused Kernels and Ulysses SP

Never combine `use_fused_kernels=True` with Ulysses sequence parallelism SP>1 for VLMs — this causes silent quality regression (verl Issues #6068).

### Variable Image Tokens and Dynamic Batching

VLM variable image token counts combined with dynamic batch size cause FSDP deadlock. Pass `dp_group` or disable dynamic bsz (verl Issues #6176).

### Text-Only Batches Still Carry Vision Config

Text-only batches of Qwen3.5 still expose `vision_config` in model config. Code must check actual multimodal inputs, not model config alone (verl Issues #6283).

### Video Support and Retokenization

Raw prompt passing for video causes retokenization drift between trainer and inference. Extend the TITO (token-in-token-out) path instead of passing raw prompts (verl Issues #6168).

### Qwen3.5-VL and Qwen3-Omni

Qwen3.5-VL and Qwen3-Omni require `VeOmniEngine` (verl Issues #6146, #6396).

### Four Bug Classes with Ulysses SP + Varlen

When combining Ulysses sequence parallelism with variable-length sequences, four bug classes emerge (verl Issues #6283):

1. Use FA2 (FlashAttention 2), not SDPA, with remove-padding (verl Issues #6283)
2. Call `.contiguous()` before `all_gather` (verl Issues #6283)
3. mRoPE 3D `position_ids` corrupts `cu_seqlens` (verl Issues #6283)
4. Text-only batch ≠ no `vision_config`; check actual inputs (verl Issues #6283)

## Diagnostic Relevance

- **RoPE shape mismatch during SFT/RL**: Verify correct rope index function and transformers ≥4.57 for Qwen3-VL (verl Issues #4483, #6073).
- **CPU/CUDA tensor mismatch on vision inputs**: Check Qwen3-VL + FSDP policy path; apply transformers #41536 fix (verl Issues #4906).
- **Silent quality drop with Ulysses SP>1**: Disable `use_fused_kernels` when using sequence parallelism (verl Issues #6068).
- **FSDP NCCL hang with variable image sizes**: Pass `dp_group` or disable dynamic bsz (verl Issues #6176).
- **Vision path invoked on text-only data**: Check actual multimodal inputs, not `vision_config` presence alone (verl Issues #6283).
- **Reward/token drift on video RL**: Use TITO via vLLM/sglang instead of raw prompt passing (verl Issues #6168).
- **Qwen3-Omni startup failures**: Ensure `VeOmniEngine` is configured (verl Issues #6146, #6396).

## Related Pages

- [FSDP Training](fsdp-training.md) — FSDP deadlock and device placement issues affect VLM training
- [SFT Cold Start](sft-cold-start.md) — VLM SFT requires careful padding and RoPE configuration before RL
- [Training-Inference Mismatch](training-inference-mismatch.md) — retokenization drift between trainer and rollout engines

## Sources

- [verl Issues](../sources/verl-issues.md) — RoPE/position_ids fragility (#4483, #6073), Qwen3-VL CPU/CUDA placement (#4906), fused kernels + Ulysses SP regression (#6068), dynamic bsz deadlock (#6176), text-only vision_config (#6283), video TITO (#6168), VeOmniEngine (#6146, #6396), Ulysses SP + varlen bug classes (#6283)
- [verl-docs](../sources/verl-docs.md) — VLM post-training infrastructure context within verl
