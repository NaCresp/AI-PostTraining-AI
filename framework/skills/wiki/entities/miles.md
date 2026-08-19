# Miles

Enterprise-ready RL framework for large-scale LLM/VLM post-training, forked from and co-evolving with [slime](slime.md). Integrates SGLang (rollout) with Megatron-LM (training) for trillion-parameter MoE workloads.

## Details

### Differentiators over slime

| Feature | Description |
|---------|-------------|
| **R3 (Rollout Routing Replay)** | Captures MoE expert routing during inference, replays during training |
| **Unified low-precision** | Matching FP8/MXFP8/NVFP4 forward passes across rollout and training |
| **Online MTP-SFT** | Keeps speculative decoding draft model aligned during RL |
| **LoRA in SGLang** | Adapters load directly into rollout engine without merge |
| **INT4 QAT** | W4A16 quantization-aware training recipe |
| **20+ plug-points** | Replace rollout, reward, loss, or filter without forking trainer |

### Supported Models

Dense: Qwen3.6/3.5/3, GLM4, Nemotron-3-Nano, MiMo, GPT-OSS.
MoE: DeepSeek-V4/V3/R1, Qwen3 MoE series, GLM4.5/4.7/5, Kimi K2/K2.5, Nemotron-3-Super.

### Hardware

NVIDIA: GB300, GB200, B200, B100, H200, H100, A100.
AMD: MI300X, MI325, MI350, MI355X (ROCm).

### Four Core Objects

Every Miles job loops over: **prompt dataset → SGLang rollout → reward model → Megatron actor** (+ frozen reference for KL).

## Relevance

Miles bridges research-grade RL (slime) and production deployment. Essential reference for MoE RL at scale where train-inference alignment (R3, unified FP8) and cross-hardware support matter.

## Related Pages

- [slime](slime.md) — upstream research framework
- [Rollout Routing Replay](../concepts/rollout-routing-replay.md) — MoE routing stability
- [Low-Precision RL](../concepts/low-precision-rl.md) — unified FP8/MXFP8 pipelines
- [Training-Inference Mismatch](../concepts/training-inference-mismatch.md) — precision drift causes
- [GRPO](grpo.md) — primary algorithm

## Sources

- [Miles Documentation](../sources/miles-docs.md) — R3, low-precision, core concepts, CLI reference
