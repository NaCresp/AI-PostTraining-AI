# slime/Miles Training Guide

Use this guide when starting an RL training run with slime or Miles (Megatron + SGLang stack).

## Checklist / Steps

### 1. Environment

- Use official Docker: `slimerl/slime:latest` (slime) or `radixark/miles:latest` (Miles)
- Convert HF weights to Megatron `torch_dist` format before training
- Verify model config script matches your model version (`source scripts/models/<model>.sh`)

### 2. Batch Size Invariant

Ensure this equation holds:

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

Example: `--rollout-batch-size 16 --n-samples-per-prompt 8 --global-batch-size 128 --num-steps-per-rollout 1`

### 3. GRPO Config

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --eps-clip 0.2
   --eps-clip-high 0.28        # DAPO-style clip-higher
)
```

### 4. Performance

```bash
PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608   # increase until OOM
   --enable-gradient-checkpointing  # via Megatron recompute flags
)
```

### 5. MoE-Specific (Miles)

```bash
--use-rollout-routing-replay    # R3 — required for MoE + GRPO
```

For low-precision MoE:
```bash
--rollout-fp8 --train-fp8       # unified FP8 (recommended)
# OR bf16 train + fp8 inference (lower friction, needs R3)
```

### 6. Resource Allocation

**Disaggregated** (recommended for 8+ GPUs):
```bash
--actor-num-gpus-per-node 4 --rollout-num-gpus 4
```

**Colocated** (limited GPUs):
```bash
--colocate --sglang-mem-fraction-static 0.8
```

### 7. Dynamic Sampling (DAPO-style)

```bash
--over-sampling-batch-size 64 \
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

Pair with `--partial-rollout` to cache interrupted generations.

### 8. Multi-Turn Agent

```bash
--custom-generate-function-path your_module.generate
--custom-rm-path your_module.reward_func
--metadata-key metadata
```

Ensure `loss_mask`: 1 for model tokens, 0 for tool/environment tokens.

## Key Pitfalls

- Model config mismatch (e.g., wrong `--rotary-base`) → silent training failure; verify against HF config.json
- Colocated OOM → reduce `--sglang-mem-fraction-static` to 0.7-0.8
- MoE without R3 → route flips cause divergence; always enable for MoE GRPO
- Embedding padding during HF→Megatron conversion → set `--vocab-size` manually if needed
- Delta weight sync incompatible with colocated mode

## Derived From

- [slime](../entities/slime.md) — architecture, config groups, extension points
- [Miles](../entities/miles.md) — R3, low-precision, enterprise features
- [Colocated vs Disaggregated Training](../concepts/colocated-vs-disaggregated.md) — resource allocation
- [Rollout Routing Replay](../concepts/rollout-routing-replay.md) — MoE stability
- [Agent Loss Masking](../concepts/agent-loss-masking.md) — multi-turn mask rules
- [Batch Size Tuning](../concepts/batch-size-tuning.md) — four-knob invariant
