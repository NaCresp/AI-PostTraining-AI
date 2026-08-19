# Multi-Turn Tool-Calling RL Guide

SFT → RL for code/tools/sandbox tasks. Core RL rules: [RL Training Principles](rl-training-principles.md).

## Pipeline

```
SFT (format + parsing)  →  GRPO/DAPO (outcome reward on final answer)
```

SFT must teach: when to call tools, syntax, using stdout/errors. Without it, RL has no signal.

## RL Notes

- Outcome-only reward — strategic tool use emerges from final accuracy.
- Rollouts are slow (long tail) — consider [async](async-training-guide.md) + partial rollout.
- Sandbox latency is part of rollout budget.
- GRPO typically beats PPO for tool RL in published ReTool-style runs.

## Framework Issues (verl examples)

- ReMax + multiturn batch union — use GRPO.
- MoE routing tensors must match truncated `response_ids`.
- Per-row tool configs may require global tool config path.
- Async: `AsyncPartialToolAgentLoop` + `partial_rollout`.

→ [Multi-Turn Tool RL](../concepts/multi-turn-tool-rl.md), [ReTool](../entities/retool.md)
