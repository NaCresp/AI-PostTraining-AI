# TRL GRPO Guide

Use this guide when running GRPO training with HuggingFace TRL on standard GPU setups.

## Checklist / Steps

### 1. Minimal Setup

```python
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

Launch: `accelerate launch train_grpo.py` (8 GPUs ≈ 1 day for 0.5B).

### 2. Reward Function

Use built-in rewards from `trl.rewards` or custom callable:

```python
def my_reward(completions, **kwargs):
    return [1.0 if correct(c) else 0.0 for c in completions]
```

### 3. Advantage Scaling

| Setting | Effect | When to Use |
|---------|--------|-------------|
| Default (std scaling) | Normalizes by group std | Standard GRPO |
| `scale_rewards=False` | No std scaling | Avoid difficulty bias |
| `scale_rewards="batch"` | Local mean + global std | More robust shaping |

### 4. Memory Optimization

- Enable PEFT/LoRA via `peft_config` for large models
- Use `reducing_memory_usage.md` techniques: gradient checkpointing, smaller batch
- Liger Kernel integration for fused GRPO loss

### 5. Speed Optimization

- **vLLM colocation**: co-locate vLLM with training for zero idle GPU time
- DeepSpeed ZeRO for multi-GPU training
- See `vllm_integration.md` for setup

### 6. Loss Type

TRL uses token-level loss aggregation (no per-sequence `1/|o_i|` scaling) to avoid response length bias. Do not add manual length normalization.

## Key Pitfalls

- `scale_rewards` default may cause difficulty bias on heterogeneous datasets → try `scale_rewards="batch"`
- vLLM colocation requires compatible transformers/vLLM versions
- TRL v1 API changes — verify against latest docs if migrating from older TRL
- Large models (>32B) may exceed single-node capacity even with LoRA — consider verl/slime for scale

## Derived From

- [TRL](../entities/trl.md) — trainer API and integrations
- [GRPO](../entities/grpo.md) — algorithm and TRL-specific scaling
- [Loss Aggregation](../concepts/loss-aggregation.md) — token-level loss in TRL
- [LoRA](../entities/lora.md) — PEFT integration
