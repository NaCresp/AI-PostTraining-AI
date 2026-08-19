# Agent-Based RL Pipeline

Design pattern that decouples experience collection (how trajectories are generated) from policy optimization (which RL algorithm updates the model).

## Mechanism

Traditional RLHF conflates two concerns:
- **Experience collection**: single-turn generation, multi-step tool interaction, environment feedback
- **Policy update**: PPO, GRPO, REINFORCE++, etc.

When entangled, every new execution mode requires trainer changes, and every new algorithm must be re-implemented per mode.

### OpenRLHF's Solution

```
AgentExecutorBase (token-in-token-out)
├── SingleTurnExecutor  → standard RLHF, custom reward functions
└── MultiTurnExecutor   → multi-step reasoning, tool calls, external envs
         ↓
   Token trajectories + rewards
         ↓
   RL Algorithms (PPO, GRPO, REINFORCE++, RLOO, ...)
```

Core principles:
- **Token-in-token-out**: trajectories stored as token IDs + log-probs, never re-detokenized
- **Unified interface**: switching single-turn ↔ multi-turn is a one-flag change
- **Algorithm-agnostic**: algorithms selected independently of execution mode
- **Pipeline-agnostic**: sync (Hybrid Engine) and async both feed the same loss layer

### slime/Miles Equivalent

slime achieves similar decoupling via plug-points:
- `--custom-generate-function-path` replaces experience collection
- `--custom-rm-path` replaces reward computation
- Algorithm selected via `--advantage-estimator`

## Diagnostic Relevance

- Retokenization drift between rollout and training → use token-in-token-out frameworks (OpenRLHF, verl TITO)
- Adding multi-turn to existing single-turn setup → should not require algorithm changes
- Chat-template mismatch causing reward=0 → symptom of text-level (not token-level) pipeline

## Related Pages

- [OpenRLHF](../entities/openrlhf.md) — primary implementer of this pattern
- [Agent Loss Masking](agent-loss-masking.md) — per-token mask rules in agent trajectories
- [Multi-Turn Tool RL](multi-turn-tool-rl.md) — multi-turn execution mode
- [Training-Inference Mismatch](training-inference-mismatch.md) — retokenization as mismatch source
- [slime](../entities/slime.md) — plug-point based decoupling

## Sources

- [OpenRLHF Documentation](../sources/openrlhf-docs.md) — agent paradigm, AgentExecutorBase, design principles
- [slime Documentation](../sources/slime-docs.md) — custom generate/reward function extension points
