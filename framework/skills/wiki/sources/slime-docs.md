# slime Documentation

- **Type**: documentation
- **Raw location**: `raw/slime-docs/`
- **Ingested**: 2026-05-29

## Key Takeaways

1. slime connects **Megatron-LM** (training) with **SGLang** (rollout) for large-scale RL; it powers GLM-4.5/4.6/4.7 training.
2. The training loop is governed by a **four-knob invariant**: `rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout`.
3. Supports **colocated** (train+rollout share GPUs) vs **disaggregated** (separate GPU pools) deployment; colocated requires lowering `--sglang-mem-fraction-static` (~0.8).
4. **Dynamic sampling** (DAPO-style) uses `over_sampling_batch_size` + custom filter functions to discard zero-variance groups.
5. **Partial rollout** caches interrupted generations to avoid wasting compute during dynamic sampling.
6. Multi-turn agent RL uses custom `--custom-generate-function-path` and `--custom-rm-path`; **loss_mask** must be 1 for model tokens, 0 for tool/environment tokens.
7. **Delta weight sync** ships only changed parameter positions (~2-3% density) for cross-datacenter training/inference disaggregation.
8. **Speculative decoding** with online MTP-SFT keeps draft model aligned with target during RL (`--enable-mtp-training`).
9. Supports GRPO, GSPO, Reinforce++, PPO; `--calculate-per-token-loss` maps to token-mean aggregation.
10. bf16 train + fp8 inference supported; Megatron checkpoint stays bf16 while SGLang loads FP8 weights.

## Pages Created or Updated

- [slime](../entities/slime.md) — created
- [Delta Weight Sync](../concepts/delta-weight-sync.md) — created
- [Colocated vs Disaggregated Training](../concepts/colocated-vs-disaggregated.md) — created
- [Agent Loss Masking](../concepts/agent-loss-masking.md) — created
- [Batch Size Tuning](../concepts/batch-size-tuning.md) — updated with four-knob invariant
- [Multi-Turn Tool RL](../concepts/multi-turn-tool-rl.md) — updated with slime agent extension points
- [Async Training](../concepts/async-training.md) — updated with partial rollout
- [slime/Miles Training Guide](../experience/slime-miles-training-guide.md) — created

## Notes

- Miles is a production fork of slime; many concepts overlap. Miles-specific features (R3, unified FP8) documented separately from miles-docs.
- Ignored Docker install steps and hardware-specific AMD tutorial details.
