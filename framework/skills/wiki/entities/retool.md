# ReTool (Reinforcement Learning for Strategic Tool Use)

SFT→RL pipeline that trains LLMs to strategically invoke tools (e.g., code execution) during multi-turn reasoning, using sandbox interaction for real-time feedback.

## Details

### Two-Phase Pipeline

**Phase 1: Cold Start SFT**
- Train on code-enhanced reasoning trajectories (ReTool-SFT dataset)
- Model learns tool-call format: when to insert code blocks, how to parse execution results
- SFT baseline on Qwen2.5-32B-Instruct: 0.24 acc/mean@30 on AIME 2025

**Phase 2: RL with Dynamic Interaction**
- Model generates hybrid trajectories (natural language + code snippets)
- Code blocks are detected and sent to sandbox for asynchronous execution
- Execution results (stdout/errors) are fed back to guide subsequent reasoning
- Reward: accuracy of the final boxed answer

### RL Results (Qwen2.5-32B)

| Method | Steps | AIME 2025 acc/mean@30 | Avg turns |
|--------|-------|-----------------------|-----------|
| SFT baseline | - | 0.24 | 7.2 |
| GRPO (DAPO recipe) | 150 | **0.60** | 10 |
| PPO | 250 | 0.55 | 8.3 |

GRPO significantly outperforms PPO in both accuracy and convergence speed for tool-calling RL.

### Sandbox Architecture

The `ToolAgentLoop` in verl handles multi-turn interaction:
1. Model generates until a code termination marker is detected
2. Code is sent to a sandbox environment (e.g., SandboxFusion, Daytona)
3. Execution results are appended to context
4. Generation continues from the new context
5. The cycle repeats until EOS or max turns

### Configuration

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: True
      max_user_turns: 16
      max_assistant_turns: 16
      tool_config_path: recipe/retool/sandbox_fusion_tool_config.yaml
```

## Relevance

ReTool demonstrates that RL can teach strategic tool use — not just when to call tools, but what code to write and how to interpret results. The avg turns increase (7.2→10) shows the model learns to use more computation when beneficial. This is directly relevant for building agentic RL systems.

## Related Pages

- [Multi-Turn Tool RL](../concepts/multi-turn-tool-rl.md) — the SFT→RL paradigm for tool-calling
- [SFT Cold Start](../concepts/sft-cold-start.md) — prerequisite SFT phase for tool-call format learning
- [GRPO](grpo.md) — the underlying RL algorithm
- [DAPO](dapo.md) — the recipe used for RL phase
- [Async Training](../concepts/async-training.md) — AsyncPartialToolAgentLoop supports fully async with partial rollout
- [Data Pipeline](../concepts/data-pipeline.md) — RLHFDataset format and tool_config_path requirements

## Sources

- [verl-recipe](../sources/verl-recipe.md) — ReTool implementation, results, sandbox configuration
