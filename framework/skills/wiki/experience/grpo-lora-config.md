# GRPO + LoRA (verl)

Memory-efficient GRPO via LoRA in **verl**. Universal LoRA LR / monitoring: [RL Training Principles](rl-training-principles.md).

## Validated verl Config

```yaml
actor_rollout_ref:
  model:
    path: <your_sft_checkpoint>
    lora:
      rank: 64
      alpha: 128
  actor:
    optim:
      lr: 1e-5
    use_kl_loss: true
    kl_loss_coef: 0.001
  rollout:
    n: 4
    enforce_eager: true
```

| Parameter | Full FT | LoRA |
|-----------|---------|------|
| LR | 1e-6 | 1e-5 |
| Group `n` | 8–64 | 4–8 |
| CUDA graph | cautious | prefer `enforce_eager=true` |

## Checkpoint Export Trap

Merged `adapter_config.json` may show `lora_alpha=0` while training used 128 → merged weights equal base model.

```python
assert json.load(open("adapter_config.json"))["lora_alpha"] == 128
```

## Backend

- **FSDP2:** preferred for LoRA in verl.
- **Megatron-Bridge:** may underperform or collapse — pin fixed commits if required.

## Derived From

- [LoRA](../entities/lora.md), [verl Known Bugs](verl-known-bugs.md)
