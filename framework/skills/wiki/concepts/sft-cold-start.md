# SFT Cold Start

SFT (Supervised Fine-Tuning) is the prerequisite phase before RL training that establishes a baseline policy capable of producing task-valid outputs. Without SFT, RL receives no useful signal because the model cannot generate syntactically valid responses for the target task.

## Mechanism

### Why SFT Precedes RL

RL optimizes a reward signal over model-generated trajectories. If the model cannot produce valid tool calls, code blocks, or reasoning formats, rewards remain sparse or zero and policy gradients provide no directional signal (verl-recipe). SFT teaches the format and behavior that RL then refines.

### Tool-Calling Cold Start

For multi-turn tool-calling, SFT teaches: (a) tool-call format, (b) when to insert code blocks, (c) how to parse execution results from sandbox feedback (verl-recipe, ReTool pipeline). The ReTool SFT baseline on Qwen2.5-32B achieves 0.24 acc/mean@30 on AIME 2025; GRPO improves this to 0.60 after 150 steps (verl-recipe).

### Math Reasoning Cold Start

For math reasoning, SFT establishes a baseline policy that RL can improve via reward shaping. The same ReTool pipeline demonstrates a 2.5× gain from SFT baseline to GRPO (verl-recipe).

### Minimal RLVR Example

The char_count recipe shows SFT baseline 0.435 → GRPO 0.6 after 2 epochs on a 135M model with a single consumer GPU (verl-recipe). This confirms that even small models benefit from a cold-start SFT phase before RL.

### SFT Quality Matters

SFT quality is critical — garbage in, garbage out for RL. A weak SFT baseline limits the ceiling of subsequent RL regardless of reward design or algorithm choice (verl-recipe).

### VLM SFT Considerations

For VLM SFT, use `use_remove_padding=True` with care around uniform-length micro-batches; mismatched batching can cause silent failures (verl Issues #4483, #6073). Qwen3-VL SFT requires recent verl plus transformers ≥4.57; using the wrong RoPE index function causes RoPE shape mismatch errors (verl Issues #4483, #6073).

## Diagnostic Relevance

- **Pre-RL baseline eval**: Run task-specific benchmarks before starting RL. ReTool uses AIME 2025 acc/mean@30; char_count uses its own accuracy metric. If baseline is near zero, fix SFT before RL (verl-recipe).
- **Format validity**: For tool-calling, inspect SFT outputs for syntactically valid code blocks and tool invocations. Invalid format → RL will not converge (verl-recipe).
- **VLM RoPE errors**: Shape mismatch on `position_ids` or mRoPE tensors during SFT indicates wrong rope index function or outdated transformers/verl pairing (verl Issues #4483, #6073).
- **Padding mode**: When using `use_remove_padding=True` with VLM SFT, verify micro-batch lengths are consistent; heterogeneous lengths within a micro-batch cause training instability (verl Issues #6073).

## Related Pages

- [Multi-Turn Tool RL](multi-turn-tool-rl.md) — SFT→RL pipeline for tool-calling agents
- [ReTool](../entities/retool.md) — reference implementation of SFT cold start for strategic tool use
- [LoRA](../entities/lora.md) — parameter-efficient method used in both SFT and RL phases

## Sources

- [verl-recipe](../sources/verl-recipe.md) — ReTool SFT baseline (0.24) → GRPO (0.60) on AIME 2025, char_count SFT baseline (0.435) → GRPO (0.6), SFT quality importance, tool-call format teaching
- [verl Issues](../sources/verl-issues.md) — VLM SFT padding and RoPE issues (#4483, #6073), Qwen3-VL transformers version requirements
