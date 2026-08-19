# Delta Weight Sync

Sparse weight synchronization that ships only changed parameter positions between training and rollout engines, instead of broadcasting the full model every step.

## Mechanism

Default sync broadcasts every parameter every step — cost scales linearly with model size. In cross-datacenter disaggregation (trainer and rollout in different datacenters over shared FS at ~100-300 MB/s), full broadcast is infeasible.

Delta sync pipeline:
1. **Diff** current weights against pinned-CPU snapshot (bytewise compare)
2. **Encode** changed (position, value) pairs (`indices`, `deltas`, or `deltas_zstd`)
3. **Bucket and flush** via NCCL or disk (safetensors files)
4. **Receiver applies** via NaN-masked overwrite (lossless — exact bytes at changed positions)

Typical density: ~2-3% of weights change per RL step. For 355B model: ~5 GB delta vs full broadcast.

### Encoding Choice

| Encoding | When to Use |
|----------|-------------|
| `indices` | NCCL or fast FS (≥600 MB/s) |
| `deltas` | Medium FS (~300-500 MB/s) |
| `deltas_zstd` | Cross-DC shared FS (≤300 MB/s) |

### Why Not Colocated

Colocated sync uses CUDA IPC (~64 B handle). Delta encoding adds overhead with zero wire savings. slime rejects `--update-weight-mode delta --colocate`.

## Diagnostic Relevance

- Cross-DC training with slow weight sync dominating step time → consider delta sync
- Check `timing/sync_rollout_weights` — if disproportionately large, delta may help even intra-DC
- Colocated mode: use default full sync or CUDA IPC, not delta

## Related Pages

- [slime](../entities/slime.md) — delta sync implementation
- [Colocated vs Disaggregated Training](colocated-vs-disaggregated.md) — when delta sync applies
- [Async Training](async-training.md) — cross-DC async setups benefit most

## Sources

- [slime Documentation](../sources/slime-docs.md) — delta sync mechanism, encoding table, configuration
