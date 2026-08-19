# Agent Loss Masking

Rules for setting per-token loss masks in multi-turn agent RL, determining which tokens participate in policy gradient computation.

## Mechanism

In multi-turn agent training (tool calling, search, code execution), the response contains tokens from two sources:
1. **Model-generated** tokens (reasoning, action commands, final answers)
2. **Environment/tool** tokens (API results, execution stdout, search snippets)

Only model-generated tokens should receive gradients. Environment tokens are observations, not policy outputs.

### Mask Rules

| Token Source | `loss_mask` | Rationale |
|-------------|-------------|-----------|
| Model reasoning/action | `1` | Policy should learn to generate these |
| Tool/environment output | `0` | External — not part of policy |
| Prompt tokens | `0` (implicit) | Not in response |

Implementation in slime/Miles custom generate functions:

```python
loss_masks += [1] * len(model_tokens)   # model output
loss_masks += [0] * len(tool_tokens)    # tool/environment output
```

`loss_mask` must be the same length as `response` token sequence.

### Framework Support

- **slime/Miles**: `--custom-generate-function-path` with manual `loss_mask` in Sample object
- **OpenRLHF**: Agent executors produce token trajectories with built-in masking via `AgentExecutorBase`
- **verl**: `ToolAgentLoop` handles multi-turn with loss mask in training batch

## Diagnostic Relevance

- Agent RL not learning tool use → check if tool output tokens have `loss_mask=0`
- Loss exploding in multi-turn training → verify mask length matches response length
- Model memorizing tool outputs → tool tokens incorrectly masked as `1`

## Related Pages

- [Multi-Turn Tool RL](multi-turn-tool-rl.md) — full SFT→RL pipeline for agents
- [Agent-Based RL Pipeline](agent-based-rl-pipeline.md) — OpenRLHF token-in-token-out design
- [slime](../entities/slime.md) — custom generate function extension point
- [ReTool](../entities/retool.md) — verl multi-turn tool recipe

## Sources

- [slime Documentation](../sources/slime-docs.md) — loss masking rules in multi-turn adaptation guide
- [OpenRLHF Documentation](../sources/openrlhf-docs.md) — token-in-token-out agent executors
