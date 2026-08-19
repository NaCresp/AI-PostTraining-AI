# Colocated vs Disaggregated Training

Resource allocation pattern for RL systems: whether training (actor) and inference (rollout) share the same GPU pool or use separate dedicated pools.

## Mechanism

### Disaggregated (Default)

Training and rollout use separate GPU allocations:

```
Actor:  actor_num_nodes × actor_num_gpus_per_node GPUs
Rollout: rollout_num_gpus GPUs
```

Both phases run in parallel. Best when total GPU count allows dedicated pools.

Framework examples:
- **slime/Miles**: separate `--actor-num-gpus-per-node` and `--rollout-num-gpus`
- **verl**: separate trainer and rollout worker groups
- **OpenRLHF**: separate Ray placement groups

### Colocated

Training and rollout share the same GPUs, time-multiplexing between phases:

```
--colocate  (slime/Miles)
--train.colocate_all  (OpenRLHF Hybrid Engine)
```

OpenRLHF Hybrid Engine puts vLLM to **sleep mode** during training, waking it for rollout. slime colocated mode requires lowering SGLang memory: `--sglang-mem-fraction-static 0.8` (Megatron occupies GPU memory before offload).

### Trade-offs

| Aspect | Disaggregated | Colocated |
|--------|--------------|-----------|
| GPU efficiency | Needs 2× GPUs minimum | Works on single pool |
| Pipeline bubbles | Possible idle time | Sequential phases |
| Memory pressure | Lower per pool | Higher — must tune memory fractions |
| Weight sync | NCCL broadcast or delta | CUDA IPC (fastest) |
| Best for | Large clusters (64+ GPUs) | Constrained GPU budgets |

## Diagnostic Relevance

- OOM during colocated training → reduce `--sglang-mem-fraction-static` or switch to disaggregated
- High trainer idle ratio with disaggregated → allocate more rollout GPUs
- Cross-DC setup → must use disaggregated + delta weight sync (colocated impossible)

## Related Pages

- [slime](../entities/slime.md) — colocated flag and memory tuning
- [OpenRLHF](../entities/openrlhf.md) — Hybrid Engine sleep mode
- [Delta Weight Sync](delta-weight-sync.md) — incompatible with colocated mode
- [Async Training](async-training.md) — async reduces colocated pipeline bubbles

## Sources

- [slime Documentation](../sources/slime-docs.md) — colocated vs disaggregated configuration
- [OpenRLHF Documentation](../sources/openrlhf-docs.md) — Hybrid Engine sleep mode
