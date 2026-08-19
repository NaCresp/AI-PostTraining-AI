# Multi-Turn Tool RL

Training LLMs to strategically use tools (code execution, search, APIs) through multi-turn RL interaction with sandbox environments.

## Mechanism

### The SFT→RL Pipeline

Multi-turn tool-calling RL requires a two-phase approach:

**Phase 1: Cold Start SFT**
- Train on curated trajectories showing proper tool-call format
- Model learns: (a) when to insert code blocks, (b) how to format tool calls, (c) how to interpret execution results
- Establishes a baseline policy that can produce valid tool interactions

Without SFT, the model cannot generate syntactically valid tool calls, and RL has no useful signal to work with. The reward is too sparse if the model never successfully invokes a tool.

**Phase 2: RL with Dynamic Interaction**
- Model generates hybrid trajectories: interleaved natural language reasoning + code snippets
- At each code block boundary, execution is sent to a sandbox
- Results (stdout, errors) are appended to the context
- "Think → Execute → Feedback" cycle continues until EOS or max turns

The RL signal comes from the final answer accuracy, but the multi-turn structure means the policy must learn strategic decisions: when to compute, what to compute, and how to use results.

### verl's ToolAgentLoop

verl implements multi-turn interaction through the `AgentLoop` architecture:

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: True
      max_user_turns: 16
      max_assistant_turns: 16
      tool_config_path: sandbox_fusion_tool_config.yaml
```

The loop detects code termination markers in the generated text, sends code to the sandbox, receives results, and continues generation.

**AsyncPartialToolAgentLoop** extends this for fully async training — supports interrupting in-flight multi-turn rollouts during parameter sync, saving state, and resuming after sync.

### Sandbox Environments

- **SandboxFusion**: Primary sandbox used in ReTool recipe
- **Daytona**: Alternative sandbox environment
- Custom implementations via `agent.custom_async_server.path`

### Experimental Results

ReTool on Qwen2.5-32B (AIME 2025):

| Phase | Method | Score | Avg turns |
|-------|--------|-------|-----------|
| SFT only | - | 0.24 | 7.2 |
| SFT → GRPO | 150 steps | **0.60** | 10.0 |
| SFT → PPO | 250 steps | 0.55 | 8.3 |

Key observations:
- GRPO outperforms PPO for tool-calling RL (faster convergence, higher final score)
- Average turns increase from 7.2 to 10 — the model learns to use more computation steps
- The 2.5× accuracy improvement (0.24 → 0.60) demonstrates RL's value beyond SFT

## Diagnostic Relevance

**Training considerations:**
- SFT quality is critical — if the model can't generate valid tool calls, RL will fail
- Multi-turn rollouts are much slower than single-turn — consider async training
- Long-tail problem is exacerbated: multi-turn responses can be very long, causing significant GPU idle time
- Tool execution latency adds to rollout time — sandbox responsiveness matters

**When to use multi-turn RL:**
- Tasks where intermediate computation (math, data analysis) improves accuracy
- Tasks where iterative refinement based on feedback is beneficial
- Agentic tasks with environment interaction

**Async training integration:**
The fully async pipeline's `AsyncPartialToolAgentLoop` enables interrupting multi-turn rollouts at parameter sync boundaries. With `partial_rollout=True`, in-flight tool interactions are saved and resumed, reducing sync overhead.

## Related Pages

- [ReTool](../entities/retool.md) — primary implementation of multi-turn tool-calling RL
- [Async Training](async-training.md) — fully async supports multi-turn with partial rollout
- [Reward Design](reward-design.md) — reward based on final answer accuracy after tool interaction
- [Batch Size Tuning](batch-size-tuning.md) — multi-turn rollouts require careful memory management

## Sources

- [verl-recipe](../sources/verl-recipe.md) — ReTool pipeline, experimental results
- [verl-docs](../sources/verl-docs.md) — ToolAgentLoop architecture, async multi-turn support
