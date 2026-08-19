# Miles Documentation

- **Type**: documentation
- **Raw location**: `raw/miles-docs/`
- **Ingested**: 2026-05-29

## Key Takeaways

1. Miles is an enterprise fork of [slime](../entities/slime.md), optimized for trillion-parameter MoE post-training with SGLang + Megatron-LM.
2. **Rollout Routing Replay (R3)** captures MoE expert routing during inference and replays it during training, eliminating route-flip instability from precision/kernel differences.
3. **Unified low-precision RL** aligns rollout and training forward passes in FP8/MXFP8/NVFP4; backward and master weights stay BF16.
4. bf16-train + fp8-inference is a lower-friction path but has precision drift on MoE — pair with R3 and optionally TIS.
5. **Speculative decoding with online MTP-SFT** keeps draft acceptance rate high as the policy evolves during RL.
6. LoRA adapters load directly into SGLang for rollout without merge/conversion.
7. Four core objects in every job: prompt dataset, SGLang rollout, reward model, Megatron/FSDP actor (+ frozen reference).
8. Supports INT4 W4A16 QAT, multi-agent co-evolution, and 20+ plug-points for custom rollout/reward/loss/filter.

## Pages Created or Updated

- [Miles](../entities/miles.md) — created
- [Rollout Routing Replay](../concepts/rollout-routing-replay.md) — created
- [Low-Precision RL](../concepts/low-precision-rl.md) — created
- [Training-Inference Mismatch](../concepts/training-inference-mismatch.md) — updated with R3 and low-precision causes
- [slime/Miles Training Guide](../experience/slime-miles-training-guide.md) — created

## Notes

- Miles and slime share the four-knob batch invariant and colocated/disaggregated modes — documented once in slime entity and colocated concept.
- NVFP4 training recipe marked experimental in source.
