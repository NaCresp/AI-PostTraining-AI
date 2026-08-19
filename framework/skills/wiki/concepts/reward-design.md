# Reward Design

How to design reward functions for RL training that produce stable learning signals, avoid reward hacking, and capture the right notion of quality.

## Mechanism

### Reward Function Types

**1. Rule-Based Rewards (Most Common)**

Exact-match verification against ground truth. Used for math, code, and other verifiable tasks.

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # Extract answer from \boxed{...}
    pred = normalize_final_answer(extract_boxed(solution_str))
    gt = normalize_final_answer(ground_truth)
    return 1.0 if pred == gt else 0.0
```

In verl, rewards are computed by the `RewardManager` which dispatches to per-`data_source` reward functions. Custom functions use `custom_reward_function.path` and `custom_reward_function.name`.

**2. Generative Reward Models (GenRM)**

Use an LLM to evaluate response quality. More expensive but captures process-level quality.

- **FAPO's GenRM**: Detects flawed reasoning in correct answers. A separate 4B model reviews each step and identifies the earliest error.
- **External service mode**: Launch GenRM as vLLM/SGLang servers with a router — recommended for large-scale training.
- **Single controller mode**: Run GenRM inside verl's training loop — simpler but less stable at scale.

**3. Model-Based Reward Models**

`AutoModelForSequenceClassification` models that score response quality. Enabled via `reward_model.enable: True`. Used for RLHF alignment tasks (e.g., full_hh_rlhf).

### Reward Shaping Techniques

**Overlong Reward Shaping (DAPO)**

Linear penalty as responses approach the hard context limit:
```yaml
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```

**Score Granularity**

- Binary (0/1): simplest, used for correctness verification
- Ternary (-1/0/1): distinguishes incorrect (-1), neutral (0), and correct (1). FAPO uses this: flawed positives get 0 instead of 1.
- Continuous: reward models output real-valued scores

### Reward Manager Variants

| Manager | Behavior |
|---------|----------|
| `naive` | Sequential reward computation |
| `dapo` | Handles DAPO-specific aggregation |
| `prime` | Parallel verification (requires multiprocessing-safe reward functions) |

### Multi-Source Rewards

When training on mixed datasets (e.g., GSM8K + MATH), the `data_source` field routes each sample to the correct reward function. Ensure `compute_score` handles all data sources or register per-source functions.

## Diagnostic Relevance

**Common pitfalls:**
- **Reward hacking**: model finds shortcuts to get high reward without genuine capability. KL regularization and reward diversity help.
- **Flawed positives**: correct answer via incorrect reasoning. FAPO's GenRM addresses this directly.
- **Sparse rewards**: binary correct/incorrect gives no gradient signal for partially correct responses. Consider intermediate scoring or process-level rewards.
- **Reward scale mismatch**: when combining rule-based rewards with KL penalties, ensure the scales are compatible. `kl_loss_coef: 0.001` is typical.

**Metrics to monitor:**
- Reward mean/std across training — should increase but not saturate
- Fraction of perfect scores — if too high early, the task may be too easy or the model is overfitting
- For GenRM: `is_flawed_positive` rate — tracks how often correct answers have flawed reasoning

### Known Issues from Production

- `transformers>=5.0` can silently zero out rewards: `apply_chat_template(tokenize=True)` returns `BatchEncoding` instead of `list[int]`; if not unwrapped via `normalize_token_ids()`, prompts are malformed (verl issue #6080).
- Fully-async pipeline hardcodes `use_rm=False` — GenRM/LLM-as-judge rewards unavailable in async mode (verl issue #5949).
- DAPO dynamic sampling hard {0,1} masks can yield all-zero or all-one filters — band-pass soft thresholds requested (verl issue #2791).

## Related Pages

- [FAPO](../entities/fapo.md) — flawed-positive detection with GenRM
- [GRPO](../entities/grpo.md) — group-relative advantage depends on reward distribution
- [DAPO](../entities/dapo.md) — overlong reward shaping, dynamic sampling based on reward
- [KL Divergence Control](kl-divergence-control.md) — reward-based KL penalty option
- [Data Pipeline](data-pipeline.md) — tokenizer API changes that silently corrupt rewards
- [Multi-Turn Tool RL](multi-turn-tool-rl.md) — reward based on final answer accuracy after tool interaction

## Sources

- [verl-docs](../sources/verl-docs.md) — reward function API, RewardManager, custom reward setup
- [verl-recipe](../sources/verl-recipe.md) — FAPO GenRM, ReTool reward design, reward function examples
- [verl Issues](../sources/verl-issues.md) — tokenizer API zeroing rewards (#6080), async GenRM limitation (#5949), dynamic sampling hard masks (#2791)
