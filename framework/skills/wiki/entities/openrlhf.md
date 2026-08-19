# OpenRLHF

Production-grade open-source RLHF framework built on Ray + vLLM + DeepSpeed with native HuggingFace model support. First framework to decouple execution mode from RL algorithm via agent-based pipeline.

## Details

### Architecture Stack

| Component | Role |
|-----------|------|
| **Ray** | Distributed scheduler and controller |
| **vLLM** | High-throughput inference engine |
| **DeepSpeed** | Memory-efficient training (ZeRO) |
| **Transformers** | Model interface (native HF weights) |
| **NCCL / CUDA IPC** | Weight sync between train and rollout |

### Agent-Based Pipeline

All training runs flow through `AgentExecutorBase`:
- **SingleTurnExecutor** — standard one-shot generation + reward
- **MultiTurnExecutor** — multi-step reasoning, tool calls, environment interaction

Algorithms (PPO, GRPO, REINFORCE++, RLOO, Dr.GRPO) consume the same token-level trajectory format.

### Execution Modes

| Mode | Description |
|------|-------------|
| **Sync + Hybrid Engine** | vLLM sleeps during training; same GPUs time-shared |
| **Async + Partial Rollout** | Rollout and training concurrent; in-flight generations survive weight sync |
| **Disaggregated** | Separate GPU pools for rollout and training |

### Supported Algorithms

PPO, GRPO, REINFORCE++, REINFORCE++-baseline, RLOO, Dr.GRPO. Selected via `--algo.advantage.estimator`.

### Key CLI Namespaces (v0.10.2)

- `--train.*` — training config, colocation, async
- `--algo.*` — algorithm, advantage estimator, KL, clipping
- `--rollout.*` — generation parameters
- `--train.agent_func_path` — custom agent executor

## Relevance

OpenRLHF is the most accessible framework for HuggingFace-native RLHF. Its agent paradigm and token-in-token-out design directly address the retokenization drift problem that causes silent RL bugs. Good choice for models up to ~70B on commodity GPU clusters.

## Related Pages

- [Agent-Based RL Pipeline](../concepts/agent-based-rl-pipeline.md) — execution/algorithm decoupling
- [GRPO](grpo.md) — supported algorithm
- [Multi-Turn Tool RL](../concepts/multi-turn-tool-rl.md) — multi-turn agent executors
- [Async Training](../concepts/async-training.md) — async + partial rollout
- [Colocated vs Disaggregated Training](../concepts/colocated-vs-disaggregated.md) — Hybrid Engine
- [verl](verl.md) — alternative framework (FSDP + vLLM)

## Sources

- [OpenRLHF Documentation](../sources/openrlhf-docs.md) — architecture, agent paradigm, training guide, async/hybrid engine
