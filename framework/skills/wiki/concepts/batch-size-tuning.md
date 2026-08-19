# Batch Size Tuning

Understanding the three-level batch hierarchy in verl and how to tune each level for throughput without affecting algorithmic behavior.

## Mechanism

### Three Batch Size Levels

```
train_batch_size (global, prompts per iteration)
  └── ppo_mini_batch_size (global, per optimization step)
       └── ppo_micro_batch_size_per_gpu (local, per forward pass)
```

1. **`data.train_batch_size`**: Total prompts sampled per training iteration. The number of rollout responses = `train_batch_size × rollout.n`. Larger values reduce rollout frequency but increase off-policy drift.

2. **`actor_rollout_ref.actor.ppo_mini_batch_size`**: Global mini-batch size for PPO updates. The sampled trajectories are split into mini-batches. This is an algorithmic parameter that affects convergence.

3. **`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`**: Samples processed per forward/backward pass on each GPU. This is a performance parameter (like gradient accumulation) that should not change algorithmic behavior. Always use the `_per_gpu` suffix version.

### Key Principle

- **Algorithmic parameters** (train_batch_size, ppo_mini_batch_size) are **global** — automatically normalized across workers
- **Performance parameters** (micro_batch_size_per_gpu, max_token_len_per_gpu) are **local** — per-GPU allocations

### Dynamic Batching

When `use_dynamic_bsz: True`, micro_batch_size is replaced by token-level limits:

```yaml
actor_rollout_ref.actor:
  use_dynamic_bsz: True
  ppo_max_token_len_per_gpu: 16384  # ≥ 2 × (max_prompt_length + max_response_length)
```

This adapts batch composition to process similar numbers of tokens per forward pass regardless of sequence length variation — significantly improving throughput when response lengths vary widely.

## Diagnostic Relevance

**Tuning tips:**

1. **Enable gradient checkpointing** (`enable_gradient_checkpointing: True`) to allow larger micro-batch sizes.

2. **Forward-only parameters can be 2× larger** — ref/rollout log_prob computation and critic forward don't need backward pass memory:
   ```yaml
   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu: 32  # 2× actor's
   critic.forward_micro_batch_size_per_gpu: 32
   ```

3. **Critic/Reward can be larger than Actor** — no vocab-size output layer:
   ```yaml
   critic.ppo_micro_batch_size_per_gpu: 16  # > actor's
   ```

4. **For dynamic batching**, increase `ppo_max_token_len_per_gpu` until OOM, starting from 2× the max sequence length.

5. **Sequence packing** (`use_remove_padding: True`) — supported for LLaMA, Mistral, Gemma, Qwen architectures. Eliminates padding overhead.

6. **Activation offloading** (`enable_activation_offload: True`) — works with gradient checkpointing for even larger micro-batch sizes (FSDP only).

**Common pitfall:** Using `ppo_micro_batch_size` (without `_per_gpu`) is deprecated. Always use the `_per_gpu` variant to avoid confusion about normalization.

## Related Pages

- [verl](../entities/verl.md) — config system and normalization conventions
- [Async Training](async-training.md) — batch size interacts with async parameter sync

## Sources

- [verl-docs](../sources/verl-docs.md) — performance tuning guide, config explanation, best practices
