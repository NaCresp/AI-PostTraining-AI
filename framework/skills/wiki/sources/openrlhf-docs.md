# OpenRLHF Documentation

- **Type**: documentation
- **Raw location**: `raw/openrlhf-docs/` (repo assets + ReadTheDocs HTML mirror)
- **Ingested**: 2026-05-29

## Key Takeaways

1. OpenRLHF uses **Ray + vLLM + DeepSpeed** distributed architecture with native HuggingFace models.
2. **Agent-based execution paradigm** decouples experience collection (single-turn, multi-turn, tool use) from RL algorithm (PPO, GRPO, REINFORCE++, RLOO).
3. All modes produce **token-in-token-out** trajectories — no re-tokenization between rollout and training, eliminating chat-template drift bugs.
4. **Hybrid Engine** (sleep-mode time-sharing) colocates vLLM and DeepSpeed on the same GPUs by putting vLLM to sleep during training.
5. **Async + partial rollout** overlaps weight sync with generation for throughput; `--train.async_enable` + `--train.agent_func_path` for async agent RL.
6. Supports VLM RLHF (Qwen3.5 with images), multi-turn VLM, off-policy correction (TIS/ICEPOP/Seq-Mask-TIS), and dynamic sampling (DAPO).
7. Algorithms selectable via `--algo.advantage.estimator`; Muon optimizer supported per entity.
8. CLI reorganized in v0.10.2 with namespaced flags (`--train.*`, `--algo.*`, `--rollout.*`).

## Pages Created or Updated

- [OpenRLHF](../entities/openrlhf.md) — created
- [Agent-Based RL Pipeline](../concepts/agent-based-rl-pipeline.md) — created
- [Colocated vs Disaggregated Training](../concepts/colocated-vs-disaggregated.md) — updated with Hybrid Engine
- [Multi-Turn Tool RL](../concepts/multi-turn-tool-rl.md) — updated with OpenRLHF agent executors
- [Async Training](../concepts/async-training.md) — updated with OpenRLHF partial rollout
- [OpenRLHF Training Guide](../experience/openrlhf-training-guide.md) — created

## Notes

- Sphinx source not in GitHub repo; ingested from ReadTheDocs HTML mirror (16 pages).
- DeepSpeed-specific memory tuning details summarized at high level only.
