---
name: ppo-defaults
description: PPO baseline defaults plus an executable verl training script template. Use when running classic PPO (single rollout per prompt) and edit script overrides directly.
---

# PPO defaults

Use this when you want PPO-style training instead of GRPO.

## Quick start

```bash
bash "${CLAUDE_SKILL_DIR}/scripts/train_ppo_verl.sh" \
  --train_file /workspace/.../train.parquet \
  --val_file /workspace/.../val.parquet \
  --model_path /root/models/Qwen/Qwen3-1.7B-Base \
  --save_dir /workspace/.../checkpoints/ppo_v1
```

The script uses conservative PPO defaults: `n=1`, KL penalty in reward enabled, token-mean aggregation, and moderate LR.

## PPO baseline knobs

| Knob | Starting point | Notes |
|------|----------------|-------|
| Rollout `n` | 1 | PPO baseline commonly single sample |
| KL control | fixed `kl_coef=0.005` in reward | Increase if policy drifts too fast |
| Actor KL loss | off | Keep one KL mechanism at a time |
| LR | 1e-6 full FT / 1e-5 LoRA | Tune separately from KL |

## Verification

- KL stays bounded (no sustained runaway).
- Reward improves without immediate entropy collapse.

## Pitfalls

- Enabling KL in reward and KL in actor loss simultaneously.
- Treating PPO as GRPO by setting `n>1` without adjusting objective assumptions.
