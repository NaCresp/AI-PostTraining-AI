---
name: grpo-lora-config
description: verl GRPO with LoRA (rank, LR, n, eager mode) and checkpoint export validation. Use when full fine-tune OOMs or experimenting with LoRA GRPO in verl only.
---

# GRPO + LoRA configuration

**verl-only.** Universal GRPO defaults: `grpo-defaults`.

## Starting config pattern

| Parameter | Full FT | LoRA |
|-----------|---------|------|
| LR | ~1e-6 | ~1e-5 |
| Group `n` | 8–16+ | 4–8 |
| LoRA rank / alpha | — | rank≈64, alpha≈128 |
| CUDA graph | cautious | prefer `enforce_eager=True` |

Keep KL loss: coef ~0.001, `low_var_kl`, token-mean aggregation.

## Pre-flight

- SFT checkpoint path points to weights that actually load in rollout.
- `use_kl_loss` consistent with algorithm choice (DAPO often disables KL — see `grpo-defaults`).

## Checkpoint export trap

Merged export may write `lora_alpha=0` while training used 128 → merged model equals base.

```python
import json
assert json.load(open("adapter_config.json"))["lora_alpha"] == 128
```

Always eval merged weights on a few prompts before long RL.

## Backend notes

- **FSDP2** preferred for LoRA in verl.
- Megatron-Bridge LoRA paths have reported underperformance or collapse — pin known-good commits if required.

## Pitfalls

- Skipping export validation → “trained” run with base-model eval.
- Same run: LoRA + async + CUDA graph without monitoring — see `posttraining-known-pitfalls`.

## Related skills

- `grpo-defaults`, `posttraining-known-pitfalls`, `sft-cold-start-gate`
