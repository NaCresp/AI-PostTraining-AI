# Wiki Index

## Entities

- [GRPO](entities/grpo.md) — Critic-free RL algorithm using group sampling and relative reward normalization (5 sources)
- [DAPO](entities/dapo.md) — GRPO extension with separated clip, dynamic sampling, token-level loss, overlong reward shaping (2 sources)
- [verl](entities/verl.md) — Production RL library for LLM post-training, supporting PPO/GRPO/DAPO and more (2 sources)
- [slime](entities/slime.md) — Megatron+SGLang RL framework; powers GLM-4.x training (1 source)
- [Miles](entities/miles.md) — Enterprise slime fork with R3, unified FP8, trillion-param MoE support (1 source)
- [OpenRLHF](entities/openrlhf.md) — Ray+vLLM+DeepSpeed RLHF with agent-based execution paradigm (1 source)
- [TRL](entities/trl.md) — HuggingFace post-training library with GRPOTrainer and vLLM integration (1 source)
- [NeMo RL](entities/nemo-rl.md) — NVIDIA multimodal RL library with DTensor/Megatron backends (1 source)
- [ReTool](entities/retool.md) — Multi-turn tool-calling RL with sandbox interaction; SFT→RL pipeline (1 source)
- [FAPO](entities/fapo.md) — Flawed-Aware Policy Optimization; detects incorrect reasoning in correct answers via GenRM (1 source)
- [LoRA](entities/lora.md) — Parameter-efficient fine-tuning for SFT and RL; checkpoint export fragility and backend-specific issues (1 source)

## Concepts

### RL Training
- [Entropy Collapse](concepts/entropy-collapse.md) — Sharp entropy decline leading to overconfidence and performance saturation; R = −a·exp(H) + b (1 source)
- [KL Divergence Control](concepts/kl-divergence-control.md) — KL loss vs KL-in-reward mechanisms to prevent policy divergence; estimator types k1/k2/k3 (1 source)
- [Loss Aggregation](concepts/loss-aggregation.md) — token-mean vs seq-mean-token-mean; critical for long-CoT stability; DrGRPO length bias fix (3 sources)
- [Clip Ratio and Trust Region](concepts/clip-ratio-and-trust-region.md) — PPO clip, DAPO clip-higher, DPPO divergence-based masking (1 source)
- [Rollout Correction](concepts/rollout-correction.md) — IS weights + rejection sampling to handle off-policy drift from training-inference mismatch (1 source)
- [Reward Design](concepts/reward-design.md) — Reward functions, Generative Reward Models (GenRM), flawed-positive detection, reward shaping (3 sources)
- [Rollout Routing Replay](concepts/rollout-routing-replay.md) — MoE expert routing capture and replay for train-inference alignment (1 source)
- [Low-Precision RL](concepts/low-precision-rl.md) — Unified FP8/MXFP8 forward passes across rollout and training (1 source)
- [Agent-Based RL Pipeline](concepts/agent-based-rl-pipeline.md) — Decouple experience collection from policy optimization (2 sources)

### SFT and Data
- [SFT Cold Start](concepts/sft-cold-start.md) — Prerequisite SFT phase before RL; cold start for tool-calling, math, and VLM tasks (2 sources)
- [Data Pipeline](concepts/data-pipeline.md) — Data preparation, tokenization, TransferQueue tensor layout, multimodal filtering (1 source)

### Infrastructure
- [Training-Inference Mismatch](concepts/training-inference-mismatch.md) — Numerical differences between rollout and training engines; cascade attention bug (4 sources)
- [Batch Size Tuning](concepts/batch-size-tuning.md) — train/mini/micro batch hierarchy; four-knob invariant; dynamic batching (3 sources)
- [Async Training](concepts/async-training.md) — One-step off-policy and fully async pipelines; partial rollout (4 sources)
- [FSDP Training](concepts/fsdp-training.md) — PyTorch FSDP/FSDP2 backend; CPUOffloadPolicy, NCCL deadlock, wrap policy, dtype issues (2 sources)
- [Colocated vs Disaggregated Training](concepts/colocated-vs-disaggregated.md) — Resource allocation: shared GPUs vs separate pools; Hybrid Engine (2 sources)
- [Delta Weight Sync](concepts/delta-weight-sync.md) — Sparse cross-DC weight updates for train/inference disaggregation (1 source)

### Advanced Topics
- [Multi-Turn Tool RL](concepts/multi-turn-tool-rl.md) — SFT→RL pipeline for tool-calling agents; sandbox interaction; loss masking (5 sources)
- [Agent Loss Masking](concepts/agent-loss-masking.md) — Per-token mask rules for multi-turn agent training (2 sources)
- [On-Policy Distillation](concepts/on-policy-distillation.md) — Teacher-student distillation during on-policy training; GKD async schedulers (3 sources)
- [VLM Post-Training](concepts/vlm-post-training.md) — Vision-language model SFT/RL; RoPE fragility, fused kernels, TITO path (2 sources)

## Experience

### Universal (Framework-Agnostic)
- [RL Training Principles](experience/rl-training-principles.md) — Defaults, monitoring, silent failures, throughput vs algorithm knobs (all stacks)
- [Stable GRPO/DAPO Launch](experience/stable-grpo-training.md) — Algorithm checklist + optional verl YAML
- [Training Collapse Triage](experience/training-collapse-triage.md) — Entropy collapse, mismatch, KL, gradients
- [Performance Tuning](experience/performance-tuning.md) — Throughput without changing algorithm behavior
- [Async Training Guide](experience/async-training-guide.md) — Off-policy overlap, staleness, resource split
- [Reward Function Design](experience/reward-function-design.md) — Rule-based, GenRM, routing, overlong shaping
- [SFT Training Guide](experience/sft-training-guide.md) — Pre-RL supervised fine-tuning checklist
- [Advanced Algorithms](experience/advanced-algorithms.md) — DAPO, DrGRPO, DPPO, FAPO, OPD selection
- [Multi-Turn Tool RL](experience/multi-turn-tool-rl-guide.md) — SFT→RL for tool-calling agents
- [Distillation Guide](experience/distillation-guide.md) — On-policy teacher→student (GKD patterns)
- [VLM Post-Training Guide](experience/vlm-post-training-guide.md) — Vision-language training pitfalls
- [Infrastructure Debugging](experience/infrastructure-debugging.md) — NCCL, dtype, weight sync (not algorithmic)

### Framework-Specific
- [verl Data Format](experience/verl-data-format.md) — Parquet schema, data_source, custom rewards
- [verl Known Bugs](experience/verl-known-bugs.md) — Symptoms, fixes, issue numbers
- [GRPO + LoRA (verl)](experience/grpo-lora-config.md) — verl LoRA config and export trap
- [slime/Miles Training Guide](experience/slime-miles-training-guide.md) — Megatron+SGLang batch invariant, MoE R3
- [OpenRLHF Training Guide](experience/openrlhf-training-guide.md) — Hybrid Engine, agent executors, async
- [TRL GRPO Guide](experience/trl-grpo-guide.md) — GRPOTrainer, scale_rewards, vLLM colocation
- [NeMo RL GRPO Guide](experience/nemo-rl-grpo-guide.md) — YAML overrides, environments, Slurm

### Redirect
- [Diagnosing Training Collapse](experience/diagnosing-training-collapse.md) → [Training Collapse Triage](experience/training-collapse-triage.md)

## Sources

- [verl Documentation](sources/verl-docs.md) — documentation, ingested 2026-05-19
- [verl-recipe Collection](sources/verl-recipe.md) — recipe implementations, ingested 2026-05-19
- [verl GitHub Issues](sources/verl-issues.md) — closed issues from verl-project/verl, ingested 2026-05-27
- [slime Documentation](sources/slime-docs.md) — Megatron+SGLang RL framework docs, ingested 2026-05-29
- [Miles Documentation](sources/miles-docs.md) — enterprise slime fork docs, ingested 2026-05-29
- [OpenRLHF Documentation](sources/openrlhf-docs.md) — Ray+vLLM agent-based RLHF docs, ingested 2026-05-29
- [TRL Documentation](sources/trl-docs.md) — HuggingFace TRL library docs, ingested 2026-05-29
- [NeMo RL Documentation](sources/nemo-rl-docs.md) — NVIDIA NeMo RL docs, ingested 2026-05-29
