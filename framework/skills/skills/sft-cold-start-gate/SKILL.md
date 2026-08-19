---
name: sft-cold-start-gate
description: SFT checklist and pre-RL gate before GRPO (data format, eval, valid rollouts). Use before starting RL on a new task or checkpoint.
---

# SFT cold start gate

**Re-read the Pre-RL gate section after SFT eval completes**—not only before you write the training script. If eval is weak, invoke `revise-before-commit` before scheduling GRPO.

## SFT checklist

1. **Mode** — Full FT (quality) vs LoRA; if LoRA, validate checkpoint export merges adapter scale correctly.
2. **Data** — Chat **message lists** when the RL loader expects them; tool trajectories need valid code fences + parsed execution results.
3. **Training** — Gradient checkpointing; dynamic batching or packing for variable length.
4. **VLM (if applicable)** — Pin `transformers` to model family; avoid wrong RoPE helper; prefer token-in-token-out path before RL.
5. **Post-train eval** — Run task benchmark on SFT checkpoint; confirm valid outputs for the RL task.

## Pre-RL gate

Do **not** start GRPO until SFT can produce **usable** rollouts. SFT quality is the ceiling for subsequent RL.

- Near-zero task accuracy after SFT → fix data, template, or training before RL.
- Invalid tool/format syntax → RL will not converge.

## Verification

- Small held-out sample: outputs match expected format for your reward extractor.
- If using verl: parquet has top-level `data_source` and `reward_model.ground_truth` (see skill `data-parquet-schema`).

## Pitfalls

- RL alone rarely teaches new output structure from a random base.
- Weak SFT limits RL ceiling regardless of algorithm.