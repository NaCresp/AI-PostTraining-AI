# Loss Aggregation

How per-token policy gradient losses are combined into a scalar loss. The choice significantly affects training stability, especially for long chain-of-thought (CoT) responses.

## Mechanism

verl supports three aggregation modes via `actor_rollout_ref.actor.loss_agg_mode`:

### `token-mean` (Recommended)

```python
loss = masked_mean(loss_mat, loss_mask)
```

Mean over all tokens across all sequences in the mini-batch. Each token contributes equally regardless of which sequence it belongs to. **This is the default and recommended setting.**

### `seq-mean-token-sum`

```python
seq_losses = sum(loss_mat * loss_mask, dim=-1)  # sum per sequence
loss = mean(seq_losses)                          # mean over sequences
```

Sum tokens within each sequence, then average across sequences. Longer sequences contribute more to the loss.

### `seq-mean-token-mean`

```python
seq_losses = sum(loss_mat * loss_mask, dim=-1) / sum(loss_mask, dim=-1)
loss = mean(seq_losses)
```

Average tokens within each sequence, then average across sequences. Each sequence contributes equally regardless of length. **This is the original GRPO paper's approach** but is noted as potentially unstable for long-CoT.

### DrGRPO: `seq-mean-token-sum-norm`

DrGRPO (2025) identifies an optimization bias in GRPO that artificially lengthens responses. It normalizes with a global constant:

```yaml
actor_rollout_ref.actor:
  loss_agg_mode: seq-mean-token-sum-norm
  loss_scale_factor: null      # optional: set to max response length
  use_kl_loss: False
algorithm:
  norm_adv_by_std_in_grpo: False
```

If `loss_scale_factor` is not set, the current batch's response length is used for normalization.

## Diagnostic Relevance

**When to suspect loss aggregation issues:**
- Response lengths growing unboundedly → may indicate length bias from seq-level aggregation
- Short responses disproportionately dominating training → token-mean may over-weight them if they are much more numerous
- Training instability with long-CoT → try switching from `seq-mean-token-mean` to `token-mean`

**Key insight:** The DAPO paper and verl's default scripts all use `token-mean`. The original GRPO paper uses `seq-mean-token-mean`, but this "may be unstable in long-CoT scenarios" per the verl documentation.

## Related Pages

- [GRPO](../entities/grpo.md) — algorithm where loss aggregation choice matters most
- [DAPO](../entities/dapo.md) — uses token-mean as one of its key innovations
- [Entropy Collapse](entropy-collapse.md) — loss aggregation affects how entropy declines

## Sources

- [verl-docs](../sources/verl-docs.md) — loss aggregation modes, DrGRPO specification
