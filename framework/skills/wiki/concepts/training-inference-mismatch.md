# Training-Inference Mismatch

Numerical differences between the training engine (FSDP/Megatron) and the inference engine (vLLM/SGLang) when using identical model weights. A common but under-recognized cause of RL training instability and collapse.

## Mechanism

Even with identical weights, different engines produce different token probabilities due to:

1. **Precision differences**: FP8 vs BF16 vs FP32 between rollout and training
2. **Attention implementation**: Flash Attention variants, cascade attention bugs
3. **Kernel differences**: Different fused kernel implementations
4. **Numerical accumulation order**: Different parallelism strategies change floating-point accumulation order

These differences accumulate across long sequences — for MoE models, probability ratios for low-probability tokens become highly volatile (orders of magnitude variation).

### Known Bug: vLLM Cascade Attention

On non-Hopper GPUs (A100, L20, B200), vLLM's cascade attention has a Flash Attention bug that causes significant precision mismatch, especially with long inputs/outputs and reasoning models like Qwen3.

**Diagnosis:** Check `training/rollout_probs_diff_mean` — should be < 0.005.

**Workaround:**
```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True
```

The root cause fix exists in `flash-attention` but hasn't been released in vLLM as of v0.10.2.

## Diagnostic Relevance

**Primary diagnostic metric:**
```yaml
actor_rollout_ref.rollout.calculate_log_probs: true
```
This enables `training/rollout_probs_diff_mean` in the logs.

| rollout_probs_diff_mean | Status |
|------------------------|--------|
| < 0.005 | Healthy |
| 0.005 - 0.01 | Borderline — monitor closely |
| > 0.01 | Problematic — precision issue from inference engine |

**Symptom pattern:**
- `actor/grad_norm` continuously increases
- Policy appears to learn (reward improves) but test-time performance degrades
- PPO ratio r(θ) is volatile, especially for low-probability tokens

**When all three conditions are met, investigate mismatch:**
1. Non-Hopper GPU architecture
2. vLLM with the cascade attention bug
3. Long input+output sequences (especially with reasoning models)

## Related Pages

- [Rollout Correction](rollout-correction.md) — the framework for correcting distribution mismatch
- [verl](../entities/verl.md) — library architecture with hybrid rollout/training engines
- [Clip Ratio and Trust Region](clip-ratio-and-trust-region.md) — DPPO addresses ratio volatility for MoE models

## Sources

- [verl-docs](../sources/verl-docs.md) — FAQ on precision mismatch, diagnostic procedure, cascade attention workaround
