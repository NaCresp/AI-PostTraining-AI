# NeMo RL Documentation

- **Type**: documentation
- **Raw location**: `raw/nemo-rl-docs/`
- **Ingested**: 2026-05-29

## Key Takeaways

1. NeMo RL is NVIDIA's open-source post-training library for LLM/VLM RL, built on Ray with DTensor and Megatron Core backends.
2. Supports GRPO, DAPO, SFT, DPO, RM, and on-policy distillation with YAML-driven configs (`uv run python examples/run_grpo.py`).
3. **Custom reward environments** via `nemo_rl.environments` — plug in math verifiers, code sandboxes, VLM environments.
4. **Async GRPO** guide for decoupling rollout and training; Eagle3 speculative decoding for faster rollouts.
5. **Sequence packing and dynamic batching** design doc covers throughput optimization for variable-length sequences.
6. FP8 quantization support; DTensor tensor-parallel accuracy issues documented separately.
7. Multi-node via Slurm (`ray.sub` launcher); Docker containers for reproducible environments.
8. HuggingFace model integration for ease of use; Megatron backend for large-scale MoE.

## Pages Created or Updated

- [NeMo RL](../entities/nemo-rl.md) — created
- [NeMo RL GRPO Guide](../experience/nemo-rl-grpo-guide.md) — created
- [Batch Size Tuning](../concepts/batch-size-tuning.md) — updated with NeMo sequence packing reference

## Notes

- NeMo RL overlaps significantly with verl in algorithm coverage; wiki focuses on NeMo-specific config patterns and environment API.
- Cluster/Slurm setup details omitted per wiki scope (environment setup).
