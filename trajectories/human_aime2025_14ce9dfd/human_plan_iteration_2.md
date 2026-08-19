# RQ3 Human Guidance Plan — Iteration 2

## Objective
Train Qwen3-1.7B-Base to maximize accuracy on AIME 2025 (30 problems, integer answers 0–999) within a 10-hour autonomous window using 4 GPUs.

**Key revision from Iteration 1**: SFT is now a minimal formatting-only warm-up. The bulk of the time and effort is dedicated to GRPO with large rollout groups and long generation lengths (~32K tokens) to support the deep multi-step reasoning AIME demands.

---

## Key Challenge Analysis

**Starting point**: Qwen3-1.7B-Base is a pretrained-only model with no instruction-following ability. It needs minimal formatting training to produce parseable `\boxed{N}` answers, but its latent mathematical reasoning should be preserved and amplified through RL, not overwritten by SFT.

**AIME characteristics**: Problems require extended multi-step reasoning (algebra, combinatorics, geometry, number theory). Solutions often involve 10–50 logical steps. A 1.7B model must learn to use its generation budget wisely — 32K tokens gives room for exploration, backtracking, and self-correction during reasoning.

**Core insight from human feedback**: SFT overtraining damages the base model's reasoning distribution. The base model already has mathematical knowledge from pretraining; SFT should only teach output format, not reasoning patterns. GRPO is where real improvement happens — it rewards correct answers while letting the model discover its own reasoning strategies.

---

## Phase 0: Environment & Cheap Evidence (30 min)

Before any training, gather critical information:

1. **GPU type & memory**: Determine batch sizing and whether 32K generation fits in memory.
2. **Installed packages**: Confirm verl, vllm, transformers versions. Check for version coupling issues (see `posttraining-known-pitfalls` skill).
3. **Tokenizer/chat template**: Inspect Qwen3-1.7B-Base's tokenizer config. Determine the right prompt format for SFT data and GRPO rollouts.
4. **Base model probe**: Generate 5 completions on simple math prompts to understand raw output style. This establishes the pre-SFT baseline for reasoning quality.
5. **Data availability**: Identify available math datasets (NuminaMath-CoT, MATH, etc.) and prepare data pipelines.

---

## Phase 1: Minimal SFT Warm-Up (≤1 hour training)

### Goal
Teach the base model **only** to produce the required answer format: `<think>...</think>\n\nThe answer is \boxed{N}.` Stop as soon as format compliance is achieved. Do not optimize for reasoning quality — that's GRPO's job.

### Data
- **Small dataset**: ~1K–3K examples maximum. Use a mix of easy-to-moderate math problems with chain-of-thought solutions.
- **Format**: `<think>step-by-step reasoning</think>\n\nThe answer is \boxed{N}.`
- **Difficulty mix**: Mostly easy/moderate (AMC 10 level, MATH level 1–3). The goal is format learning, not reasoning improvement.
- **Exclusion**: No AIME 2024/2025 problems.

### Training Config
- Full fine-tune (1.7B is small enough)
- **1 epoch only** — stop early if format compliance is reached
- LR: ~2e-5, cosine schedule
- Max sequence length: 4096 tokens (sufficient for format learning)
- Batch size: tuned to GPU memory

### SFT Reasoning Degradation Check
**Critical**: Before proceeding to GRPO, verify SFT did not damage reasoning:
1. **Format compliance test**: Generate on 20 held-out math prompts. Target: ≥80% produce parseable `\boxed{N}` output.
2. **Reasoning quality test**: On 10 moderate-difficulty problems where the base model showed some reasoning ability in Phase 0, compare SFT model outputs qualitatively. If the SFT model produces shorter, more formulaic, or less exploratory reasoning than the base model, SFT was too aggressive.
3. **Perplexity spot-check**: If the SFT model's loss on a small held-out set of competition math text is significantly worse than the base model's, reasoning capacity may be damaged.

**Abort criteria**: If format compliance is <50% after 1 epoch, try a different data format or smaller LR. If reasoning appears degraded, reduce SFT data to 500 examples and retrain for 0.5 epochs.

**Duration estimate**: 30–60 min training + 15 min verification.

---

## Phase 2: GRPO — Main Training Phase (7–8 hours)

This is where the real improvement happens. GRPO lets the model discover effective reasoning strategies by rewarding correct final answers, without constraining the reasoning path.

### Why GRPO Works for AIME
- **Outcome-based reward**: Binary signal (correct integer or not) is clean and perfectly aligned with the eval metric.
- **Group-relative advantages**: Multiple completions per prompt create within-group contrast — the model learns from its own successes vs. failures.
- **Long generation**: 32K tokens allows the model to attempt complex multi-step reasoning, try different approaches, backtrack, and self-correct — all behaviors that emerge under RL but are hard to teach via SFT.

### Data
- **Prompt set**: ~2K–5K competition-level math prompts (AMC 10/12, pre-2024 AIME, Olympiad problems, MATH level 4–5). Only prompts needed — ground truth answers used solely for reward computation.
- **Difficulty**: Skew toward harder problems. Easy problems where the model already gets 100% correct provide no learning signal.
- **Exclusion**: No AIME 2024/2025 problems.

### Training Config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Initialize from | Best SFT checkpoint (format-compliant) | Minimal SFT preserves reasoning |
| Group size (rollouts per prompt) | **16** | Large group = more diverse completions = richer advantage signal. 16 is the sweet spot for a 1.7B model on 4 GPUs. |
| Max generation length | **32,768 tokens** | AIME needs extended reasoning. Let the model think as long as it needs. |
| Temperature | 1.0 | High diversity in rollouts is critical for GRPO signal |
| KL coefficient | 0.001–0.01 | Start low (0.001). Increase if policy drifts too fast. |
| Learning rate | 5e-7 to 1e-6 | Conservative — RL is sensitive to LR |
| LR schedule | Cosine with warmup (5% of steps) | Stable start |
| Batch size | Tune to fit 16 rollouts × 32K tokens across 4 GPUs | May need gradient accumulation |
| Training steps | 100–300 per iteration | Eval every 50–100 steps |
| Reward | Binary: 1.0 if extracted integer matches ground truth, 0.0 otherwise | Clean, aligned with eval |

### Memory Management for 32K Generation
- 32K tokens × 16 rollouts is memory-intensive. Strategies:
  - Use vLLM for rollout generation (efficient KV cache management)
  - If OOM: reduce group size to 8 (minimum viable), or use micro-batched rollouts
  - `use_remove_padding=True` to avoid FSDP2 mask bugs (per `posttraining-known-pitfalls`)

### GRPO Iteration Strategy

**Iteration 1 (hours 1.5–4.5):**
- Conservative settings: KL=0.01, LR=5e-7, 100 steps
- Primary goal: Confirm GRPO is working (rewards improving, no collapse)
- Eval at step 50 and step 100
- Request eval from eval agent at step 100

**Iteration 2 (hours 5–7.5):**
- Adjust based on Iteration 1 results:
  - If reward improving but slowly → increase LR to 1e-6, reduce KL to 0.001
  - If entropy collapsing → increase KL, reduce LR
  - If within-group reward std ≈ 0 → increase temperature, check if problems are too easy/hard
- Train 100–200 more steps from best Iteration 1 checkpoint
- Request eval

**Iteration 3 (hours 7.5–9, if time permits):**
- Fine-tune based on accumulated lessons
- May adjust data mix (harder/easier problems) based on eval analysis
- Request eval

### Monitoring (per `monitor-rl-training` skill)

| Metric | Healthy | Action if unhealthy |
|--------|---------|---------------------|
| Policy entropy | Stable band, not crashing to 0 | Increase KL, reduce LR |
| Rollout prob diff | < 0.005 | > 0.01 → check weight sync, disable cascade attention |
| Grad norm | Stable / slow drift | Sustained rise → reduce LR |
| Within-group reward std | > 0 | = 0 for >10 steps → increase temp, check data difficulty |
| Response length std | Has variance | Uniform → entropy collapse starting |
| Mean reward | Trending up (even slowly) | Flat for >50 steps → adjust hyperparameters |

### Abort Criteria for GRPO
- **Entropy collapse**: Policy entropy crashes to near-zero → stop, increase KL coefficient by 5×, restart from previous checkpoint.
- **Reward stuck at zero**: For >30 steps, mean reward = 0 across all groups → check reward function, check if model outputs are parseable (may need SFT fix).
- **Accuracy regression**: Eval accuracy drops below SFT baseline → revert to SFT checkpoint, reduce LR by 50%, increase KL.
- **Within-group reward std = 0**: For >15 consecutive steps → all completions identical, no learning signal. Increase temperature to 1.2 or resample prompts.

---

## Time Budget (10 hours)

| Phase | Estimated Time | Cumulative |
|-------|---------------|------------|
| Phase 0: Environment setup + cheap evidence | 0.5h | 0.5h |
| Phase 1: SFT data prep + training + verification | 1.0h | 1.5h |
| GRPO data preparation | 0.5h | 2.0h |
| GRPO Iteration 1 (100 steps + eval wait) | 2.5h | 4.5h |
| GRPO Iteration 2 (100–200 steps + eval wait) | 2.5h | 7.0h |
| GRPO Iteration 3 (if time permits) | 2.0h | 9.0h |
| Final eval + submission | 1.0h | 10.0h |

**Buffer strategy**: SFT is capped at 1 hour. If it finishes faster, all extra time goes to GRPO iterations. Always reserve 30 min for final eval + submission. If running behind, skip Iteration 3 and submit best available.

---

## Hypotheses (Ranked by Confidence)

1. **Minimal SFT preserves reasoning better than extensive SFT** — Training only for format compliance avoids overwriting the base model's reasoning distribution. (High confidence — supported by human feedback and RL literature)
2. **32K generation length is critical for AIME** — AIME problems require extended multi-step reasoning that cannot fit in 4K–8K tokens. Longer generation allows exploration, backtracking, and self-correction. (High confidence)
3. **Large rollout groups (16) improve GRPO signal** — More diverse completions per prompt create richer within-group contrast, leading to better advantage estimates and faster learning. (High confidence)
4. **GRPO will improve over SFT-only** — RL with outcome reward lets the model discover reasoning strategies that SFT cannot teach. (Medium-high confidence)
5. **Conservative KL + low LR prevents collapse** — Starting with tight KL and low LR, then relaxing, is safer than starting aggressive and trying to recover. (Medium confidence)
6. **Competition-level prompts transfer to AIME** — Training on AMC/Olympiad-style prompts covers the reasoning patterns needed for AIME. (Medium confidence)

---

## Success Criteria

- **Minimum viable**: Any non-zero accuracy on AIME 2025 (base model likely scores 0)
- **Good outcome**: 2+ problems correct out of 30 (~7%+ accuracy)
- **Strong outcome**: 4+ problems correct (~13%+ accuracy)
- **Exceptional**: 6+ problems correct (~20%+ accuracy)

---

## Experiment-Level Abort Criteria

- If SFT produces no parseable math outputs after 2 attempts with different data/format → skip SFT, try GRPO with a simple prompt template and hope the model learns format through RL.
- If GRPO shows no reward improvement after 150 steps across 2 different hyperparameter settings → submit best available checkpoint.
- If total GPU time consumed exceeds 8 hours with no checkpoint improving on the SFT baseline → freeze and submit best available.
- If all GRPO iterations show entropy collapse regardless of KL/LR settings → the model may be too small for this approach; submit best SFT checkpoint.

---

## Final Submission Criteria

Submit the checkpoint with the highest evaluated AIME accuracy. If multiple checkpoints tie, prefer:
1. Lower INVALID-format rate (more reliable output)
2. More correct answers on harder problems (AIME II > AIME I)
3. Later GRPO checkpoint (more training signal absorbed)
