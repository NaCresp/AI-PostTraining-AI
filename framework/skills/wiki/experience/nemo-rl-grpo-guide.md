# NeMo RL GRPO Guide

Use this guide when running GRPO with NVIDIA NeMo RL (Ray + DTensor/Megatron backends).

## Checklist / Steps

### 1. Quick Start (Single GPU)

```bash
uv run python examples/run_grpo.py
# Default: Qwen2.5-1.5B, OpenMathInstruct-2, config in examples/configs/grpo_math_1B.yaml
```

Multi-GPU:
```bash
uv run python examples/run_grpo.py cluster.gpus_per_node=8
```

### 2. Choose Backend

| Backend | When to Use |
|---------|-------------|
| DTensor (default) | Prototyping, models up to ~30B |
| Megatron Core | Large MoE, multi-node, long context |

Megatron:
```bash
uv run python examples/run_grpo.py \
  --config examples/configs/grpo_math_1B_megatron.yaml
```

### 3. Custom Reward Environment

Implement or use built-in environments:
- `math_environment` — math answer verification
- `code_environment` — code execution scoring
- `dapo_math_verifier` — DAPO-style verification
- Custom: extend `nemo_rl.environments.interfaces`

See `guides/environments.md` for environment API.

### 4. Config Overrides

```bash
uv run python examples/run_grpo.py \
  policy.model_name="meta-llama/Llama-3.2-1B-Instruct" \
  checkpointing.checkpoint_dir="results/my_run" \
  logger.wandb_enabled=True \
  logger.wandb.name="grpo-experiment"
```

### 5. Multi-Node (Slurm)

```bash
NUM_ACTOR_NODES=2
COMMAND="uv run ./examples/run_grpo.py --config examples/configs/grpo_math_8B.yaml cluster.num_nodes=2" \
sbatch --nodes=${NUM_ACTOR_NODES} --gres=gpu:8 ray.sub
```

Build container first per `docker.md`.

### 6. Advanced Features

- **Async GRPO**: `guides/async-grpo.md` for decoupled rollout/training
- **Eagle3 speculative decoding**: faster rollouts via draft model
- **FP8**: `fp8.md` for quantization
- **Sequence packing**: enabled via dynamic batching config

### 7. Evaluation

Built-in eval pipeline in `guides/eval.md`. Override val samples:
```bash
logger.num_val_samples_to_print=10
```

## Key Pitfalls

- DTensor TP accuracy issues on some models → check `guides/dtensor-tp-accuracy.md`
- Megatron backend requires correct parallelism config for model size
- Download models before multi-node job to avoid training-time download delays
- Environment must match Docker container for reproducibility

## Derived From

- [NeMo RL](../entities/nemo-rl.md) — architecture, backends, environments
- [GRPO](../entities/grpo.md) — algorithm reference
- [Batch Size Tuning](../concepts/batch-size-tuning.md) — sequence packing
- [Async Training](../concepts/async-training.md) — async GRPO
