# Performance Tuning for Throughput

Maximize RL training throughput **without** changing algorithmic behavior. For stability defaults see [RL Training Principles](rl-training-principles.md).

## Checklist

### 1. Micro-Batch and Memory

- Enable **gradient checkpointing** on the actor.
- Increase per-GPU micro-batch until OOM, then back off one step.
- **Forward-only** stages (reference log-prob, critic forward) can often use **~2×** actor micro-batch — no backward graph.

### 2. Dynamic Batching (Variable-Length RL)

Essential when response lengths vary (long CoT, tool traces).

- Cap **max tokens per GPU** (or equivalent); raise until OOM.
- Rule of thumb: cap ≥ **2 × (max_prompt + max_response)**.
- With distributed training, all ranks must agree on batch splits — missing `dp_group` causes NCCL deadlock (especially VLMs).

→ [Batch Size Tuning](../concepts/batch-size-tuning.md)

### 3. Sequence Packing

- Pack or strip padding so compute goes to real tokens, not pads.
- Often **50–70%** wasted compute without packing on skewed length distributions.
- Packing and dynamic batching solve the same problem differently — many stacks allow only one at a time (e.g. NeMo RL).

### 4. Activation Offloading

- Trade PCIe bandwidth for memory when FSDP-style training is memory-bound.
- Typically requires gradient checkpointing.

### 5. Async Training

If rollout is **> ~50%** of step time:

| Mode | Typical gain | Risk |
|------|-------------|------|
| One-step off-policy | ~23–40% | Low |
| Fully async + staleness cap | ~1.5–2.7× | Higher — needs correction + tuning |

Start synchronous → one-step off → fully async. Details: [Async Training Guide](async-training-guide.md).

### 6. Weight Sync (Disaggregated / Colocated Hybrid)

- Batch-and-bulk weight broadcast beats per-tensor streaming (GKD reports ~12×).
- If sync dominates step time: larger buckets, check NCCL / network.
- After sync, **spot-check generations** — silent wrong weights happen on new architectures.

→ [Delta Weight Sync](../concepts/delta-weight-sync.md)

### 7. GPU Split

**Goal:** rollout wall time ≈ training wall time.

| Observation | Action |
|-------------|--------|
| Trainer idle | More rollout GPUs |
| Rollout workers idle | More training GPUs |

## Key Pitfalls

- Micro-batch / max-tokens knobs → throughput only.
- Global batch, mini-batch, epochs, `n`, clip, KL, loss agg → **convergence** — tune separately.
- Async adds debugging surface — do not skip a healthy sync baseline.

## Framework Examples

| Stack | Doc |
|-------|-----|
| verl | [verl](../entities/verl.md), `perf/best_practices` in raw |
| slime/Miles | [slime/Miles Training Guide](slime-miles-training-guide.md) |
| NeMo RL | `design-docs/sequence-packing-and-dynamic-batching` in raw |

## Derived From

- [Batch Size Tuning](../concepts/batch-size-tuning.md), [Async Training](../concepts/async-training.md)
- [Colocated vs Disaggregated](../concepts/colocated-vs-disaggregated.md)
