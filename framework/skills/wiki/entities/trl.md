# TRL (Transformers Reinforcement Learning)

HuggingFace's full-stack post-training library integrated with transformers. Provides trainers for SFT, GRPO, DPO, PPO, RLOO, and more with vLLM/DeepSpeed/PEFT integrations.

## Details

### Trainer Taxonomy

**Online (RL) methods** (⚡ = vLLM support):
- `GRPOTrainer`, `RLOOTrainer`, `OnlineDPOTrainer`, `PPOTrainer`, `XPOTrainer`, `NashMDTrainer`

**Offline methods**:
- `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `ORPOTrainer`, `BCOTrainer`, `CPOTrainer`

**Distillation**:
- `GKDTrainer`, `MiniLLMTrainer`

### GRPOTrainer API

Minimal usage:

```python
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

### GRPO Configuration Highlights

| Parameter | Effect |
|-----------|--------|
| `scale_rewards=False` | Disable std scaling (avoids difficulty bias) |
| `scale_rewards="batch"` | Local mean + global std (more robust shaping) |
| Loss type | Token-level (no per-sequence `1/|o_i|` scaling) |

### Integrations

- **vLLM**: Co-located generation for throughput
- **DeepSpeed**: Distributed training
- **PEFT/LoRA**: Parameter-efficient fine-tuning
- **Liger Kernel**: Fused kernels for GRPO loss
- **Trackio**: Experiment tracking

## Relevance

TRL is the lowest-friction entry point for HuggingFace ecosystem RL. Best for models that fit in standard HF+DeepSpeed infrastructure (<70B). For large MoE or Megatron-scale training, prefer [verl](verl.md), [slime](slime.md), or [NeMo RL](nemo-rl.md).

## Related Pages

- [GRPO](grpo.md) — algorithm details and TRL-specific scaling options
- [LoRA](lora.md) — PEFT integration for parameter-efficient RL
- [Loss Aggregation](../concepts/loss-aggregation.md) — TRL token-level loss avoids length bias
- [SFT Cold Start](../concepts/sft-cold-start.md) — SFTTrainer prerequisite

## Sources

- [TRL Documentation](../sources/trl-docs.md) — trainer taxonomy, GRPOTrainer API, integrations
