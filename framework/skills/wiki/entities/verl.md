# verl (Volcano Engine RL for LLMs)

Production-grade RL library for LLM post-training, supporting PPO, GRPO, DAPO, DPPO, GPG, OPO, RLOO, SPPO, SPIN, and On-Policy Distillation (OPD). Originally from ByteDance Seed / HybridFlow.

## Details

### Supported Algorithms

| Algorithm | Advantage Estimator | Critic Required | Key Feature |
|-----------|-------------------|-----------------|-------------|
| PPO | GAE | Yes | Actor-critic with clipped surrogate |
| GRPO | grpo | No | Group relative rewards |
| DAPO | grpo + filter | No | Clip-higher, dynamic sampling |
| DPPO | grpo | No | Divergence-based trust region |
| GPG | gpg | No | Corrected advantage, no KL/ref |
| OPO | opo | No | Length-weighted optimal baseline |
| OTB | optimal_token_baseline | No | Token-level variance reduction |
| RLOO | rloo | No | Leave-one-out baseline |
| SPIN | DPO loss | No | Self-play iterative DPO |
| OPD | distillation | No | On-policy teacher distillation |
| FAPO | grpo + GenRM | No | Flawed-positive detection |
| GVPO | grpo (MSE) | No | No importance sampling, MSE loss |
| SPO | grpo (offline) | No | Single-stream with offline values |

### Training Backends

- **FSDP / FSDP2**: PyTorch native, recommended for most use cases. FSDP2 offers ~7% lower memory.
- **Megatron**: For large MoE models (DeepSeek-V3 671B, Qwen3-235B). Supports PP/TP/EP/CP.

### Rollout Backends

- **vLLM**: Primary inference backend, supports CUDA graph, chunked prefill.
- **SGLang**: Alternative backend with different memory semantics for `gpu_memory_utilization`.

### Key Architecture Concepts

- **Hybrid Engine**: Actor, rollout, and reference share the same process; weights are synced between training and inference phases.
- **Single Controller**: Global batch sizes (train_batch_size, ppo_mini_batch_size) are automatically normalized per worker.
- **Config convention**: `*_per_gpu` suffixed parameters are local; unsuffixed parameters are global.
- **Default entropy_coeff = 0.0** since v0.3.x (changed from nonzero in earlier versions).

### Config Key Map

The most commonly adjusted knobs for stable RL training:

| Config | Purpose | Typical Value |
|--------|---------|---------------|
| `data.train_batch_size` | Prompts per iteration | 512-1024 |
| `actor_rollout_ref.rollout.n` | Responses per prompt | 8-64 |
| `actor_rollout_ref.actor.clip_ratio` | PPO clip range | 0.2 |
| `actor_rollout_ref.actor.optim.lr` | Learning rate | 1e-6 |
| `actor_rollout_ref.actor.optim.clip_grad` | Gradient clipping | 1.0 |
| `actor_rollout_ref.actor.loss_agg_mode` | Loss aggregation | token-mean |
| `actor_rollout_ref.actor.use_kl_loss` | KL regularization | True (GRPO) |
| `actor_rollout_ref.actor.kl_loss_coef` | KL weight | 0.001 |
| `actor_rollout_ref.rollout.temperature` | Sampling temperature | 1.0 |

## Relevance

verl is the primary RL training framework for the AI4AI project. Understanding its config system, supported algorithms, and engineering conventions is essential for running and debugging RL experiments.

## Related Pages

- [GRPO](grpo.md) — primary algorithm used with verl
- [DAPO](dapo.md) — best-practice recipe
- [LoRA](lora.md) — parameter-efficient fine-tuning for SFT and RL
- [Batch Size Tuning](../concepts/batch-size-tuning.md) — train/mini/micro batch relationships
- [Training-Inference Mismatch](../concepts/training-inference-mismatch.md) — precision issues between rollout and training
- [FSDP Training](../concepts/fsdp-training.md) — recommended training backend with known issues
- [On-Policy Distillation](../concepts/on-policy-distillation.md) — OPD algorithm support
- [SFT Cold Start](../concepts/sft-cold-start.md) — prerequisite SFT phase before RL
- [VLM Post-Training](../concepts/vlm-post-training.md) — vision-language model support
- [Data Pipeline](../concepts/data-pipeline.md) — dataset format, tokenization, and TransferQueue
- [ReTool](retool.md) — multi-turn tool-calling RL recipe
- [FAPO](fapo.md) — flawed-positive detection with GenRM
- [Reward Design](../concepts/reward-design.md) — reward function patterns and GenRM integration
- [Multi-Turn Tool RL](../concepts/multi-turn-tool-rl.md) — SFT→RL pipeline for tool-calling

## Sources

- [verl-docs](../sources/verl-docs.md) — full documentation, config reference, baseline results
- [verl-recipe](../sources/verl-recipe.md) — recipe implementations and experimental results
