# Low-Precision RL

Unified low-precision training pipelines where rollout and training share the same quantization logic on the forward pass, while backward passes and master weights remain in BF16.

## Mechanism

A common MoE RL failure mode: train in BF16, serve in FP8 → per-layer numerical disagreement compounds into divergent log-probabilities and misdirected gradients.

Miles/slime unified approach:

| Stage | Precision |
|-------|-----------|
| Rollout forward | FP8 / MXFP8 / NVFP4 GEMM |
| Trainer forward | Same quant config as rollout |
| Trainer backward | BF16 gradients |
| Optimizer | BF16 master weights |
| Weight sync | Re-quantize on each sync to SGLang |

### Supported Formats

| Format | Block Layout | Hardware | Maturity |
|--------|-------------|----------|----------|
| BF16 | — | All | Baseline |
| FP8 block-wise | 128×128, FP32 scales | Hopper, Blackwell | GA |
| MXFP8 | 1×32, UE8M0 scales | Blackwell only | Beta |
| NVFP4 | 1×16, two-level scales | Blackwell | Experimental |

### Rollout × Training Compatibility

- BF16 rollout + BF16 train: baseline
- FP8 rollout + FP8 train: unified (recommended for MoE)
- BF16 train + FP8 inference: lower friction but has precision drift on MoE — pair with R3

Rules: `--rollout-mxfp8` and `--rollout-fp8` are mutually exclusive; `--train-mxfp8` requires `--rollout-mxfp8`.

## Diagnostic Relevance

- MoE RL diverges with bf16-train + fp8-inference → enable R3 or switch to unified FP8
- Log-probability mismatch increases after enabling FP8 rollout → check quant config alignment
- Blackwell hardware: MXFP8 available; Hopper: FP8 block-wise only

## Related Pages

- [Rollout Routing Replay](rollout-routing-replay.md) — mitigates route flips from precision drift
- [Training-Inference Mismatch](training-inference-mismatch.md) — precision as mismatch source
- [Miles](../entities/miles.md) — unified low-precision implementation
- [slime](../entities/slime.md) — bf16 train + fp8 inference mode

## Sources

- [Miles Documentation](../sources/miles-docs.md) — format table, compatibility matrix, recipes
- [slime Documentation](../sources/slime-docs.md) — bf16 train + fp8 inference quick path
