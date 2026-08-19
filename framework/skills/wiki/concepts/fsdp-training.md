# FSDP/FSDP2 Training

FSDP (Fully Sharded Data Parallel) is PyTorch's native distributed training backend and the recommended choice for most verl post-training workloads. FSDP2 offers approximately 7% lower memory than FSDP1 while preserving the same programming model (verl-docs).

## Mechanism

### CPUOffloadPolicy and Weight Updates

Enabling `CPUOffloadPolicy` breaks `update_weights`: the path `get_per_tensor_param()` → `state_dict()` crashes with CPU/CUDA device mismatch (verl Issues #5995). Workaround: set `offload_policy=false` while keeping `param_offload=true` and `optimizer_offload=true` (verl Issues #5995).

### NCCL Deadlock on New Architectures

On new architectures such as Gemma4, heterogeneous buffer sizes combined with non-deterministic `named_buffers()` ordering across ranks cause NCCL deadlock during weight broadcast (verl Issues #6365). Fix: sort buffers by name before broadcast (verl Issues #6365).

### Wrap Policy and Partial `_no_split_modules`

Wrap policy fails when `_no_split_modules` lists layer names that do not exist in the installed transformers version — e.g., Qwen3.5 references a layer absent in transformers 5.2 (verl Issues #6289). The policy should skip unresolved module names rather than error (verl Issues #6289).

### Silent Dtype Divergence in `forward_step()`

`forward_step()` hardcodes bf16 autocast, ignoring `mixed_precision.param_dtype` configuration (verl Issues #5932). This causes silent dtype divergence between configured and actual compute precision (verl Issues #5932).

### Dynamic Batching Deadlock

Dynamic batching without passing `dp_group` causes deadlock: ranks create different micro-batch counts, leading to NCCL hang (verl Issues #6176, #6178). Pass `dp_group` explicitly or disable dynamic batch size (verl Issues #6176, #6178).

### fp32 Weights with FlashAttention

FSDP autocasts compute to bf16 while keeping weights in fp32 — this is intentional (verl Issues #252). FlashAttention may warn about fp32 weights; forcing fp16 weights causes NaN `grad_norm` (verl Issues #252).

## Diagnostic Relevance

- **`update_weights` crash with CPU/CUDA mismatch**: Check `offload_policy` setting; disable it while retaining param/optimizer offload (verl Issues #5995).
- **NCCL hang on new model architectures**: Suspect buffer ordering; verify fix for sorted buffer broadcast is applied (verl Issues #6365).
- **Wrap policy error on startup**: Check `_no_split_modules` against installed transformers version; unresolved names should be skipped (verl Issues #6289).
- **Unexpected dtype in loss/grad**: Compare `mixed_precision.param_dtype` config against actual autocast in `forward_step()` (verl Issues #5932).
- **Hang with dynamic batching**: Ensure `dp_group` is passed or disable dynamic bsz; symptom is NCCL timeout with variable micro-batch counts across ranks (verl Issues #6176, #6178).
- **NaN grad_norm after forcing fp16**: Revert to fp32 weights with bf16 compute autocast (verl Issues #252).

## Related Pages

- [verl](../entities/verl.md) — library providing FSDP/FSDP2 training backends
- [Batch Size Tuning](batch-size-tuning.md) — dynamic batching parameters interact with FSDP deadlock risks
- [Training-Inference Mismatch](training-inference-mismatch.md) — precision mismatches between FSDP training and rollout engines

## Sources

- [verl-docs](../sources/verl-docs.md) — FSDP/FSDP2 as recommended backend, ~7% memory improvement with FSDP2
- [verl Issues](../sources/verl-issues.md) — CPUOffloadPolicy crash (#5995), Gemma4 NCCL deadlock (#6365), wrap policy partial modules (#6289), bf16 autocast hardcoding (#5932), dynamic bsz deadlock (#6176, #6178), fp32 weights + FlashAttention (#252)
