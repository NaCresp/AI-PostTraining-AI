# verl-recipe Collection

- **Type**: codebase + documentation
- **Raw location**: `raw/verl-recipe/`
- **Ingested**: 2026-05-19

## Key Takeaways

1. FAPO (Flawed-Aware Policy Optimization) addresses the "flawed positive" problem — responses that reach the correct answer through incorrect reasoning steps. It uses a Generative Reward Model (GenRM) to detect these and penalizes them, improving reliability beyond raw accuracy.
2. ReTool demonstrates the SFT→RL pipeline for multi-turn tool-calling: first cold-start SFT to learn tool-call format, then RL with sandbox interaction. GRPO achieves 0.6 acc/mean@30 on AIME 2025 after 150 steps (vs SFT baseline 0.24), outperforming PPO (0.55 after 250 steps).
3. Generative Reward Models (GenRM) can be integrated either as external vLLM/SGLang services with a router, or inside verl's single controller. External service mode is more stable for large-scale training.
4. The GKD (Generalized Knowledge Distillation) recipe reveals practical async distillation engineering: one-step-off and two-step-off schedulers for overlapping rollout/teacher/training, with batched NCCL weight sync providing ~12× speedup over naive streaming.
5. SPO (Single-stream Policy Optimization) generates only 1 response per prompt (vs GRPO's 8), using offline value estimates and Thompson Sampling for prompt selection. Compensates with 8× larger batch size.
6. GVPO eliminates importance sampling entirely, using an MSE loss that matches log-ratio deviations to advantage deviations. This removes a source of training instability but requires all responses from the same prompt group to be processed within one iteration.
7. The char_count recipe serves as a minimal RLVR tutorial — SFT baseline 0.435 → GRPO 0.6 after 2 epochs on a 135M model with a single consumer GPU.
8. Consistent finding across recipes: **modify only one thing at a time** when debugging RL training.

## Pages Created or Updated

- [ReTool](../entities/retool.md) — multi-turn tool-calling RL pipeline
- [FAPO](../entities/fapo.md) — flawed-aware policy optimization with GenRM
- [Reward Design](../concepts/reward-design.md) — reward functions, GenRM, flawed-positive detection
- [Multi-Turn Tool RL](../concepts/multi-turn-tool-rl.md) — SFT→RL pipeline for tool-calling agents
- [Entropy Collapse](../concepts/entropy-collapse.md) — updated with recipe-level experimental details
- [verl](../entities/verl.md) — updated supported algorithms list

## Notes

- Skipped hardware-specific recipes (r1_ascend, flash_rl_ascend, grpo_mindspeed_mm, dance_grpo_mindspeed_mm).
- Skipped environment setup and Docker instructions.
- DanceGRPO (diffusion model RL) is specialized to image generation and not directly relevant to LLM RL stability.
- SPO and GVPO are noted for their architectural insights but not given full entity pages as they are less mature recipes.
