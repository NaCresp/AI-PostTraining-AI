---
name: data-parquet-schema
description: verl training parquet columns (prompt, reward_model, top-level data_source) and custom reward registration. Use only when training with verl.
---

# Training data parquet schema

**verl-specific** — not required for custom training scripts outside verl.

## Required columns

| Column | Type | Content |
|--------|------|---------|
| `prompt` | list of dicts | `[{"role": "user", "content": "..."}]` |
| `reward_model` | dict | `{"ground_truth": "...", ...}` |
| `data_source` | string | **Top-level** — reward manager reads this directly |

## Critical pitfall

`data_source` nested only inside `reward_model` → `KeyError` at training start. Promote to top-level column.

## Custom reward

```yaml
reward:
  custom_reward_function:
    path: /path/to/reward_function.py
    name: compute_score
```

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return {"score": 1.0, "acc": 1.0}  # or zeros
```