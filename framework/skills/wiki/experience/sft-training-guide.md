# SFT Training Guide

Supervised fine-tuning before RL (math, tools, VLM). Universal pitfalls: [RL Training Principles](rl-training-principles.md).

## Checklist

1. **Mode:** full FT (quality) vs LoRA (`rank≈64`, `alpha≈128`). LoRA checkpoint export must be validated — see [GRPO + LoRA Config](grpo-lora-config.md) for verl-specific traps.
2. **Data:**
   - Chat **message lists**, not opaque API blobs, when the RL loader expects them.
   - Tool trajectories: valid code fences + parsed execution results.
   - VLM filters: `pop()` image/video fields, not `get()` — avoids serializing full media every row.
3. **Training:** gradient checkpointing; dynamic batching or packing for variable length (watch uniform-length VLM batches with packing).
4. **VLM:** pin `transformers` to model-family requirements; avoid wrong RoPE helper for the generation; prefer token-in-token-out inference path before RL.
5. **Post-train:** eval SFT checkpoint; confirm valid outputs for the RL task (syntax, tools, answer format).

## Pre-RL Gate

Do not start GRPO until SFT can produce **usable** rollouts — SFT quality is the ceiling for tool/multi-turn RL.

→ [SFT Cold Start](../concepts/sft-cold-start.md), [Multi-Turn Tool RL Guide](multi-turn-tool-rl-guide.md)

## Derived From

- [LoRA](../entities/lora.md), [VLM Post-Training](../concepts/vlm-post-training.md), [Data Pipeline](../concepts/data-pipeline.md)
