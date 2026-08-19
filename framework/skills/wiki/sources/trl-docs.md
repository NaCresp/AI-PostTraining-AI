# TRL Documentation

- **Type**: documentation
- **Raw location**: `raw/trl-docs/`
- **Ingested**: 2026-05-29

## Key Takeaways

1. TRL is HuggingFace's full-stack post-training library integrated with transformers; v1 marks a major API shift.
2. **GRPOTrainer** supports vLLM-backed generation; minimal API: `GRPOTrainer(model=..., reward_funcs=..., train_dataset=...)`.
3. GRPO advantage normalization has configurable scaling: `scale_rewards=False` removes std scaling (avoids difficulty bias); `scale_rewards="batch"` uses local mean + global std.
4. TRL deliberately avoids per-sequence length scaling in loss (no `1/|o_i|`) to prevent response-level length bias.
5. Online methods: GRPO, RLOO, OnlineDPO, NashMD, PPO, XPO (vLLM support on several).
6. Offline methods: SFT, DPO, BCO, CPO, KTO, ORPO. Distillation: GKD, MiniLLM.
7. Integrations: DeepSpeed, PEFT/LoRA, Liger Kernel, vLLM colocation, Trackio, Unsloth.
8. Co-located vLLM in TRL enables training+inference on same GPUs without idle time.

## Pages Created or Updated

- [TRL](../entities/trl.md) — created
- [Loss Aggregation](../concepts/loss-aggregation.md) — updated with TRL length-bias note
- [GRPO](../entities/grpo.md) — updated with TRL scale_rewards options
- [TRL GRPO Guide](../experience/trl-grpo-guide.md) — created

## Notes

- TRL targets HuggingFace ecosystem users; less suited for large MoE Megatron-scale training compared to slime/verl/NeMo RL.
- Experimental trainers marked 🧪 in source taxonomy.
