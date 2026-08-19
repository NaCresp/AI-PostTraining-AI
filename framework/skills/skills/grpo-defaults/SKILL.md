---
name: grpo-defaults
description: GRPO baseline defaults plus an executable verl training script template. Use when starting a new GRPO run; edit script args/overrides instead of deriving command from scratch.
---

# GRPO defaults

Use the executable template first, then tune one knob per run.

## Quick start

```bash
bash "${CLAUDE_SKILL_DIR}/scripts/train_grpo_verl.sh" \
  --train_file /workspace/.../train.parquet \
  --val_file /workspace/.../val.parquet \
  --model_path /root/models/Qwen/Qwen3-1.7B-Base \
  --save_dir /workspace/.../checkpoints/grpo_v1
```

The script already sets practical GRPO defaults (token-mean, KL=0.001, n=8, clip=0.2). Modify script flags or `EXTRA_OVERRIDES` only.

## Baseline knobs

| Knob | Starting point | Notes |
|------|----------------|-------|
| Group size `n` | 8–16 | Must be > 1 |
| Clip ε | 0.2 | DAPO: asymmetric 0.2 / 0.28 |
| Loss aggregation | token-mean | Better for long CoT |
| KL loss | `low_var_kl`, coef 0.001 | Do not mix KL-in-reward + KL-in-loss |
| LR | 1e-6 full FT / 1e-5 LoRA | LoRA usually higher |

## DAPO upgrade

- Asymmetric clip (0.2 / 0.28)
- Disable KL loss
- Add dynamic group filtering and overlong shaping

## Verification

- Within-group reward std > 0 after a few steps.
- Entropy does not collapse monotonically.

## Pitfalls

- `n = 1` gives no GRPO relative signal.
- Changing multiple algorithm knobs in one run blocks diagnosis.
