# DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

RL algorithm that extends GRPO with four key innovations for more stable and efficient training. Achieved 50% on AIME 2024 with Qwen2.5-32B, outperforming DeepSeek-R1-Zero with 50% fewer training steps.

## Details

### Innovation 1: Separated Clip Ratios (Clip-Higher)

DAPO decouples the lower and upper clip bounds, allowing more aggressive upward policy updates:

```yaml
actor_rollout_ref.actor:
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
```

This asymmetry allows the policy to explore more aggressively while still constraining downward updates.

### Innovation 2: Dynamic Sampling with Group Filtering

Filters out groups where all outputs have the same reward (all correct or all incorrect), since these produce zero advantage and waste compute:

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc
    max_num_gen_batches: 10
```

The trainer repeats sampling until enough qualified groups (with mixed rewards) fill `train_batch_size`.

### Innovation 3: Token-Level Loss (via loss_agg_mode)

Uses `token-mean` loss aggregation, which means the loss across all tokens in all sequences. This avoids the length bias of per-sequence averaging.

### Innovation 4: Overlong Reward Shaping

Penalizes outputs that approach the hard context limit with a linear penalty:

```yaml
data:
  max_response_length: 20480  # 16384 + 4096 buffer
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```

Penalty increases linearly from 0 to `penalty_factor` as length exceeds `max_response_length - buffer_len`.

### Reproduction Results (AIME 2024)

| Setup | AIME 2024 | Hardware |
|-------|-----------|----------|
| DAPO | 52% | 16×8×H800 |
| DAPO w/o Dynamic Sampling | 50% | 16×8×H800 |
| DAPO w/o Token-level Loss & DS | 44% | 16×8×H20 |

### Known Issues

- Enabling CUDA graph (`enforce_eager=False`) might cause model performance degradation — cause under investigation.
- "RL infrastructures nowadays still have inherent unrobustness" — modify one thing at a time when experimenting.
- Dynamic sampling uses hard {0,1} masks that can yield all-zero or all-one filters, causing unstable accept ratios. Feature request: band-pass soft thresholds (e.g., keep reward in [0.2, 0.8]) (verl issue #2791).
- Gemma-4 on FSDP: shows exploding grad norms (10–50+) under GRPO; 31B fails FSDP load with FQN mismatch (verl issue #5999). New architectures need buffer-order and weight-sync validation.

## Relevance

DAPO represents the current best-practice recipe for math-reasoning RL. Its innovations directly address common training instabilities. The `reward_manager: dapo` setting enables the full DAPO pipeline in verl.

## Related Pages

- [GRPO](grpo.md) — base algorithm that DAPO extends
- [Loss Aggregation](../concepts/loss-aggregation.md) — token-mean vs seq-mean-token-mean
- [Clip Ratio and Trust Region](../concepts/clip-ratio-and-trust-region.md) — clip-higher strategy
- [Entropy Collapse](../concepts/entropy-collapse.md) — Clip-Cov/KL-Cov strategies integrate with DAPO
- [FSDP Training](../concepts/fsdp-training.md) — infrastructure issues affecting DAPO training

## Sources

- [verl-docs](../sources/verl-docs.md) — algorithm specification, config, reproduction results
- [verl Issues](../sources/verl-issues.md) — dynamic sampling hard masks (#2791), Gemma4 instability (#5999)
