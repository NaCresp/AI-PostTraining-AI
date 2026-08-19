# Async Training

Decoupling rollout and training to overlap GPU-idle time during long-tail sample generation. Achieves 1.5–2.7× speedup at the cost of increased off-policy drift.

## Mechanism

### The Problem

In synchronous training, rollout dominates runtime (e.g., ~70% for DAPO 32B). GPUs sit idle waiting for the longest response in the batch to complete. Adding more GPUs for rollout doesn't help because long-tail samples are the bottleneck, not total compute.

### One-Step Off-Policy

The simplest async approach: train on the previous step's rollout data while generating new data in parallel.

- Previous step's data is used for training (one step stale)
- Rollout and training resources are separately allocated
- NCCL-based parameter sync between actor and rollout (~300ms latency)
- 23-40% speedup with comparable quality

```yaml
trainer:
  nnodes: 1
  n_gpus_per_node: 6   # training GPUs
rollout:
  nnodes: 1
  n_gpus_per_node: 2   # rollout GPUs
```

### Fully Async Policy

More aggressive: supports multi-step asynchrony, streaming sample generation, and partial rollout.

**Key parameters:**
| Parameter | Purpose | Recommendation |
|-----------|---------|----------------|
| `staleness_threshold` | Max fraction of stale samples | < 1.0 (0.5 typical) |
| `trigger_parameter_sync_step` | Local updates between sync | Balance efficiency vs on-policy |
| `require_batches` | Mini-batches per training step | 1 for streaming, 4 for stability |
| `partial_rollout` | Interrupt in-flight rollouts at sync | True for max overlap |

**Supported modes:**
1. **On-policy pipeline** (staleness=0, sync_step=1): same as sync but resource-isolated
2. **Stream off-policy** (staleness=0, sync_step>1): streaming but synchronous
3. **Async stream + stale** (staleness>0, partial=False): async with wait for in-flight
4. **Async stream + partial** (staleness>0, partial=True): async with interrupt — fastest

### Experimental Results (Qwen2.5-Math-7B, DAPO)

| Mode | 32 GPUs | 64 GPUs | 128 GPUs |
|------|---------|---------|----------|
| Colocate sync | 3d 17h | 1d 17h | 1d 16h |
| Fully async | 1d 9h (2.66×) | 21h (1.92×) | 17h (2.35×) |

Quality: max accuracy comparable (0.33 vs 0.35 on AIME24), marginal degradation acceptable.

## Diagnostic Relevance

**Resource allocation:** The ideal setup makes rollout time ≈ training time, minimizing pipeline bubbles.

Monitor:
- `trainer/idle_ratio` — if high, allocate more rollout resources
- `rollouter/idle_ratio` — if high, allocate more training resources
- `fully_async/count/stale_samples_processed` — track staleness impact

**Staleness calibration:**
- `staleness_threshold=0`: synchronous (no stale data)
- `staleness_threshold=0.5`: recommended starting point
- `staleness_threshold=1.0`: approximately equivalent to one-step off-policy
- `staleness_threshold>1.0`: not recommended, too much stale data

**Streaming batch size (`require_batches`):** Smaller values = less bubble time but can cause training instability and longer response lengths due to data ordering effects. Start with 4, decrease cautiously.

**When to use [rollout correction](rollout-correction.md):** In async training, metrics may become unstable in later training stages. Enable rollout IS to correct for parameter staleness:
```yaml
algorithm.rollout_correction.bypass_mode: False  # compute log_prob with training engine
```

## Related Pages

- [Rollout Correction](rollout-correction.md) — corrects off-policy drift from staleness
- [Training-Inference Mismatch](training-inference-mismatch.md) — exacerbated in async setups
- [Batch Size Tuning](batch-size-tuning.md) — batch parameters interact with async settings

## Sources

- [verl-docs](../sources/verl-docs.md) — one-step off-policy implementation, fully async architecture, experimental results
