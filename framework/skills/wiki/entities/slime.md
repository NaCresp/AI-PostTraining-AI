# slime

LLM post-training RL framework connecting Megatron-LM (training) with SGLang (rollout). Powers GLM-4.5/4.6/4.7 training at Z.ai. Designed for RL scaling with flexible custom data generation.

## Details

### Architecture

- **Training backend**: Megatron-LM (primary), with FSDP support for smaller models
- **Rollout backend**: SGLang via `sgl-router` for multi-server scheduling
- **Orchestration**: Ray for distributed job submission
- **Data format**: JSONL prompts with `--input-key`, `--label-key`, optional `--metadata-key`

### Supported Algorithms

| Estimator | Notes |
|-----------|-------|
| `grpo` | Default for math reasoning; supports clip-higher (`--eps-clip-high 0.28`) |
| `gspo` | Group sequence policy optimization |
| `reinforce_plus_plus` | REINFORCE++ variant |
| `reinforce_plus_plus_baseline` | With baseline |
| `ppo` | Standard PPO |

Also supports `--use-tis` (Truncated Importance Sampling) for off-policy correction.

### Key Configuration Groups

| Group | Purpose |
|-------|---------|
| `MODEL_ARGS` | Megatron architecture (layers, hidden size, rotary base) |
| `CKPT_ARGS` | HF checkpoint, ref-load, save paths |
| `ROLLOUT_ARGS` | Dataset, batch knobs, sampling, reward type |
| `GRPO_ARGS` | Algorithm, KL, clipping, entropy |
| `PERF_ARGS` | TP/PP/CP/EP parallelism, dynamic batching |
| `SGLANG_ARGS` | Engine TP, memory fraction, `--sglang-*` passthrough |

### Batch Invariant

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

Set any three; slime validates the fourth.

### Extension Points

- `--custom-generate-function-path` — multi-turn agent rollout
- `--custom-rm-path` — custom reward function
- `--dynamic-sampling-filter-path` — DAPO-style group filtering
- `--buffer-filter-path` — partial rollout cache strategy

### Weight Sync Modes

- Default: full NCCL broadcast every step
- Delta: sparse position-only updates for cross-DC disaggregation (`--update-weight-mode delta`)

## Relevance

slime is the research-grade foundation for large-scale MoE RL. [Miles](miles.md) is its enterprise fork. Compared to [verl](verl.md), slime uses Megatron+SGLang natively rather than FSDP+vLLM, making it better suited for 100B+ MoE training.

## Related Pages

- [Miles](miles.md) — enterprise fork with R3, unified FP8, production stability
- [GRPO](grpo.md) — primary algorithm
- [Colocated vs Disaggregated Training](../concepts/colocated-vs-disaggregated.md) — resource allocation modes
- [Delta Weight Sync](../concepts/delta-weight-sync.md) — sparse cross-DC weight updates
- [Agent Loss Masking](../concepts/agent-loss-masking.md) — multi-turn training mask rules
- [Batch Size Tuning](../concepts/batch-size-tuning.md) — four-knob invariant

## Sources

- [slime Documentation](../sources/slime-docs.md) — architecture, quick start, agent adaptation, delta sync, speculative decoding
