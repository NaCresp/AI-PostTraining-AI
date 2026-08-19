# OpenRLHF Training Guide

Use this guide when running RLHF with OpenRLHF (Ray + vLLM + DeepSpeed + HuggingFace).

## Checklist / Steps

### 1. First Run (RLVR with Qwen3-4B)

Follow the quick start: prepare dataset, download pretrained model, launch with default sync pipeline.

### 2. Choose Execution Mode

| Goal | Configuration |
|------|--------------|
| Max simplicity, limited GPUs | Hybrid Engine (colocated) |
| Max throughput, enough GPUs | Disaggregated (separate rollout/train pools) |
| Max throughput + some staleness | Async + partial rollout |

**Hybrid Engine** (colocated):
```bash
--train.colocate_all
# vLLM sleeps during training; tune memory carefully
```

**Async + partial rollout**:
```bash
--train.async_enable
--train.agent_func_path examples/python/your_agent.py  # if agent-based
```

### 3. Choose Algorithm

```bash
--algo.advantage.estimator grpo    # or ppo, reinforce++, rloo
```

Algorithm is independent of execution mode — same flag regardless of single-turn or multi-turn.

### 4. Multi-Turn / Agent Training

```bash
--train.agent_func_path path.to.agent_executor
```

Implement `AgentExecutorBase` subclass returning token-level trajectories. No re-tokenization between rollout and training.

For custom single-turn rewards:
```bash
--train.agent_func_path path.to.custom_reward_agent
```

### 5. VLM RLHF

Use VLM-specific training scripts (e.g., `train_vlm_math_hybrid_engine.sh`). Images in both prompts and environment feedback supported in v0.10+.

### 6. Off-Policy Correction

For async or precision mismatch:
```bash
# TIS / ICEPOP / Seq-Mask-TIS options in agent_training guide
```

### 7. Dynamic Sampling (DAPO)

Configure in training script alongside `--algo.advantage.estimator grpo`.

## Key Pitfalls

- Retokenization drift → always use token-in-token-out agent executors, never string concatenation
- Hybrid Engine OOM → vLLM memory not fully released during training; reduce model size or use ZeRO-3
- Async without partial rollout → weight sync interrupts all in-flight generations, wasting compute
- CLI flag migration: v0.10.2 uses namespaced flags (`--train.*`, `--algo.*`); check migration guide for old scripts
- v0.10.2 reorganized all CLI — verify flag names against `common_options.html`

## Derived From

- [OpenRLHF](../entities/openrlhf.md) — architecture and CLI
- [Agent-Based RL Pipeline](../concepts/agent-based-rl-pipeline.md) — execution/algorithm decoupling
- [Colocated vs Disaggregated Training](../concepts/colocated-vs-disaggregated.md) — Hybrid Engine
- [Async Training](../concepts/async-training.md) — async + partial rollout
- [Multi-Turn Tool RL](../concepts/multi-turn-tool-rl.md) — multi-turn executors
