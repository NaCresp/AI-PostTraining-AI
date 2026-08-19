# FAPO (Flawed-Aware Policy Optimization)

RL algorithm that detects and penalizes "flawed positives" — responses that reach the correct answer through incorrect reasoning steps. Uses a Generative Reward Model (GenRM) to evaluate reasoning quality beyond final-answer accuracy.

## Details

### The Flawed-Positive Problem

Standard rule-based rewards only check the final answer. A response can get reward=1.0 by arriving at the correct answer through flawed reasoning (e.g., algebraic errors that cancel out, lucky guesses, correct answer with wrong justification). Training on these flawed positives reinforces bad reasoning patterns, reducing reliability.

### How FAPO Works

1. **Rule-based check**: verify if the final boxed answer matches ground truth
2. **If correct**: submit the response to a Generative Reward Model (GenRM) that reviews the solution step by step
3. **GenRM identifies**: the index of the earliest error step (or -1 if no error found)
4. **Penalty**: if a flaw is detected in a correct response, subtract `FLAWED_REWARD_PENALTY` (default 1.0) from the reward

```python
reward_score = 1.0 if correct else -1.0
if correct and is_flawed_positive:
    reward_score -= FLAWED_REWARD_PENALTY  # 1.0 - 1.0 = 0.0
```

This means a flawed positive gets reward 0.0 instead of 1.0, effectively treating it as neutral rather than reinforcing incorrect reasoning.

### GenRM Integration

Two deployment modes:

1. **External service** (recommended for large-scale): launch GenRM as separate vLLM/SGLang servers with a router. More stable, scales independently.
2. **Single controller**: start GenRM inside verl's training loop. Simpler setup but "still unstable for large-scale training scenarios."

The GenRM is trained separately (e.g., FAPO-GenRM-4B from a Qwen2.5-4B base) on a dataset of math solutions annotated with error locations.

### Reward Loop Infrastructure

FAPO implements `RewardLoop` in verl, which enables async reward computation during training. This is essential because GenRM inference can be slow (up to 16384 tokens per critique).

## Relevance

FAPO addresses a fundamental limitation of outcome-based reward: it conflates "gets the right answer" with "reasons correctly." For math RL, where chain-of-thought quality matters, detecting flawed positives improves the quality of the learned policy. The GenRM approach generalizes to any domain where process-level evaluation is important.

## Related Pages

- [Reward Design](../concepts/reward-design.md) — reward function design patterns including GenRM
- [GRPO](grpo.md) — base RL algorithm used by FAPO
- [DAPO](dapo.md) — FAPO builds on DAPO training infrastructure

## Sources

- [verl-recipe](../sources/verl-recipe.md) — FAPO implementation, GenRM training, reward function design
