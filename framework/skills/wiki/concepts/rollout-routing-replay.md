# Rollout Routing Replay (R3)

Technique that captures MoE expert routing decisions during inference and replays them during training, ensuring bit-identical expert allocation between rollout and trainer forward passes.

## Mechanism

For each token, an MoE router selects `top-k` experts via a learned linear layer and top-k operation. In production:
- Rollout uses SGLang with FP8 quantization and non-deterministic kernels
- Training uses Megatron with potentially different precision

Tiny numerical differences flip expert routes at the per-layer, per-token level. Without R3:
- Rollout selects experts `{2, 7}` for token 314
- Training selects experts `{2, 8}` for the same token
- Gradients are computed against the wrong expert, causing policy divergence over thousands of steps

R3 workflow:
1. SGLang returns `routed_experts` in response `meta_info` (shape: `(seq_len-1, num_layers, top_k)`, int32)
2. Stored in `sample.rollout_routed_experts`
3. Trainer's forward pass uses recorded routes instead of recomputing them

Memory cost: `(num_tokens - 1) × num_layers × top_k × 4 bytes`. For 32K tokens, 60 layers, top_k=8: ~60 MB per sample.

### When R3 Is Not Required

- Dense (non-MoE) models
- `reinforce_plus_plus` with `--use-tis` (masks off-policy term)

## Diagnostic Relevance

- MoE RL with GRPO showing divergence despite stable rewards → check if routes differ between rollout and training
- 0% accuracy after weight sync on new MoE architecture → may indicate routing mismatch, not dummy weights
- Enable with `--use-rollout-routing-replay` in Miles/slime

## Related Pages

- [Miles](../entities/miles.md) — primary implementation
- [Low-Precision RL](low-precision-rl.md) — precision drift that causes route flips; R3 is the mitigation
- [Training-Inference Mismatch](training-inference-mismatch.md) — broader mismatch category
- [GRPO](../entities/grpo.md) — algorithm where R3 is required for MoE

## Sources

- [Miles Documentation](../sources/miles-docs.md) — R3 mechanism, memory cost, configuration
