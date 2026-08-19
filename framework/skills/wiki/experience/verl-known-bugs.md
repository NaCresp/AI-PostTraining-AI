# verl Known Bugs Quick Reference

Consolidated list of known verl bugs and workarounds. Check here first when encountering unexplained errors.

## Silent Correctness Bugs (Training Runs but Produces Wrong Results)

| Bug | Symptom | Fix | Issue |
|-----|---------|-----|-------|
| `use_remove_padding=False` + FSDP2 | `loss_mask` used as `attention_mask`, clipfrac >40%, inflated KL at step 0 | Use `use_remove_padding=True` | #6278 |
| `transformers>=5.0` tokenizer | `apply_chat_template` returns `BatchEncoding` not `list[int]`, rewards silently zero | Call `normalize_token_ids()` or pin transformers<5 | #6080 |
| LoRA `lora_alpha=0` in export | Merged model = base model despite training improvement | Verify `adapter_config.json` after merge | #6380 |
| `forward_step()` bf16 hardcode | Ignores `mixed_precision.param_dtype` → silent dtype divergence | Check actual vs configured precision | #5932 |
| Fused kernels + Ulysses SP>1 | Silent quality regression on VLMs | Disable fused kernels with SP>1 | #6068 |

## Crash Bugs

| Bug | Error Message | Fix | Issue |
|-----|---------------|-----|-------|
| `CPUOffloadPolicy` + `update_weights` | CPU/CUDA storage mismatch in `state_dict()` | `offload_policy=false`, keep `param_offload=true` | #5995 |
| FSDP wrap policy + Qwen3.5 | `_no_split_modules` lists nonexistent layer | Skip unresolved module names (needs verl patch) | #6289 |
| Dynamic batching without dp_group | NCCL deadlock (ranks create different micro-batch counts) | Pass `dp_group` explicitly or disable dynamic bsz | #6176 |
| Gemma4 buffer ordering | NCCL deadlock during weight broadcast | Sort buffers by name before broadcast | #6365 |
| FP8 rollout weight sync | Shape mismatch during `load_quanted_weights` | FP8 not yet plug-and-play; avoid or match quantization | #6112 |
| Forcing fp16 weights | NaN `grad_norm` | Keep fp32 weights with bf16 compute autocast | #252 |
| VLM remove-padding + uniform seq_len | RoPE shape mismatch | Fixed in verl PR #6127 | #6073 |
| TransferQueue mixed tensors | Nested/stacked tensor mismatch | Use TransferQueue ≥0.1.7 with jagged default | #6407 |

## Performance/Stability Bugs

| Bug | Impact | Workaround | Issue |
|-----|--------|-----------|-------|
| vLLM cascade attention (non-Hopper) | Precision mismatch `rollout_probs_diff_mean > 0.01` | `disable_cascade_attn=True` | — |
| CUDA graph with DAPO/LoRA | Model performance degradation (cause unknown) | `enforce_eager=True` | — |
| Megatron-Bridge LoRA | ~10% underperformance or collapse to 0 | Use FSDP2, or pin Megatron-Bridge commit `6259ae83+` | #5094 |
| Colocated vLLM IPC collision | Weight sync to wrong process | Include job ID in socket path | #6233 |
| DAPO dynamic sampling hard masks | All-zero or all-one filters → unstable accept ratio | Feature request for band-pass soft thresholds | #2791 |
| Multimodal data filter with `get()` | ~1000× slower (serializes full images) | Use `pop()` for image/video fields | #6145 |
| Sync generation dump | Blocks training for seconds (n=16, bs=1024) | Use async dump | — |

## Version Coupling Matrix

| Change | What Breaks |
|--------|-------------|
| transformers ≥5.0 | tokenizer API, nested tensors, RoPE functions |
| transformers 5.2 | Qwen3.5 `_no_split_modules` resolution |
| transformers <4.57 | Qwen3-VL RoPE index incompatible |
| flash_attn (old) | varlen bugs with long sequences |
| TransferQueue <0.1.7 | Mixed tensor layout crashes |

## Debugging Principle

See [RL Training Principles](rl-training-principles.md) — change one knob at a time.
