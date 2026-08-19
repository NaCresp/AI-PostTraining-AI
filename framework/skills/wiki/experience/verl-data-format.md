# verl Training Data Format

Prepare parquet data files for verl GRPO/PPO training.

## Required Format

verl expects a parquet file with these columns:

| Column | Type | Content |
|--------|------|---------|
| `prompt` | list of dicts | Chat messages: `[{"role": "user", "content": "..."}]` |
| `reward_model` | dict | Reward metadata: `{"ground_truth": "42", "data_source": "math"}` |
| `data_source` | string | **Must be a top-level column** — verl's reward manager reads it directly |

### Critical: `data_source` Must Be Top-Level

verl's `RewardManager` expects `data_source` as a top-level column in the parquet, NOT nested inside `reward_model`. If your data has it nested:

```python
import pandas as pd
df = pd.read_parquet("train.parquet")
df['data_source'] = df['reward_model'].apply(lambda x: x.get('data_source', 'math'))
df.to_parquet("train_fixed.parquet", index=False)
```

Failure mode: `KeyError: 'data_source'` crash at the start of training.

## Custom Reward Functions

Register via config:

```yaml
reward:
  custom_reward_function:
    path: /path/to/reward_function.py
    name: compute_score
```

The function signature:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # solution_str = model's full response
    # ground_truth = from reward_model.ground_truth in parquet
    return {"score": 1.0, "acc": 1.0}  # or {"score": 0.0, "acc": 0.0}
```

### Reward Function Pitfalls

- Handle edge cases: infinity, NaN, empty strings
- `normalize_answer` must check `math.isfinite(val)` before `int(val)` — large floats cause `OverflowError`
- Return dict must include `"acc"` key if using DAPO dynamic sampling (`filter_groups.metric: acc`)
- Format reward (e.g., 0.1 for correct format but wrong answer) can cause reward hacking — prefer binary 0/1

## Validation Data

`data.val_files` must point to a valid parquet file (same format as train). Passing `null` crashes verl.

## Key Pitfalls

- `data_source` nested in `reward_model` instead of top-level → KeyError crash
- `data.val_files=null` → TypeError crash (use a small validation split instead)
- `transformers>=5.0` changes `apply_chat_template` return type → silently zeros rewards
- Mixed data sources: every `data_source` value must have a matching reward function handler
