# RQ3 Human Guidance Plan — Iteration 1

## Objective
Train Qwen3-1.7B-Base to maximize accuracy on AIME 2025 (30 problems, integer answers 0–999) within a 10-hour autonomous window using 4 GPUs.

---

## Key Challenge Analysis

**Starting point**: Qwen3-1.7B-Base is a *base* (pretrained-only) model. It cannot follow instructions, produce structured answers, or reason through multi-step math without further training. A 1.7B model is also small for competition math — every training decision matters.

**AIME characteristics**: Problems require multi-step reasoning across algebra, combinatorics, geometry, and number theory. Answers are integers 0–999. Even strong frontier models find AIME challenging; for a 1.7B model, any non-zero accuracy is a meaningful signal.

---

## Strategy: Two-Phase Training

### Phase 1: Supervised Fine-Tuning (SFT) Warm-Up
**Goal**: Teach the base model to (a) follow a think-then-answer format and (b) perform basic-to-intermediate mathematical reasoning.

**Data**: Curate ~10K–30K examples from publicly available math datasets (NuminaMath-CoT, MetaMathQA, MATH dataset, or similar). Focus on:
- Competition-style problems (AMC 10/12, pre-2024 AIME, Olympiad) for difficulty alignment
- Chain-of-thought format: `<think>step-by-step reasoning</think>\n\nThe answer is \\boxed{N}.`
- Exclude any AIME 2024/2025 problems

**Format**: Use Qwen3's chat template. Each example = user prompt + assistant CoT response.

**Training config**:
- Full fine-tune (not LoRA) — 1.7B is small enough
- 2–3 epochs, LR ~2e-5, cosine schedule
- Max sequence length 2048–4096 tokens (CoT needs room)
- Batch size tuned to fill GPU memory across 4 GPUs

**Duration estimate**: ~1–2 hours

**Success criteria**: Model produces parseable `\boxed{N}` answers on >80% of held-out math prompts.

**Abort criteria**: After 1 epoch, if loss hasn't dropped below 2.0 or outputs are gibberish, revisit data/format.

### Phase 2: GRPO (Group Relative Policy Optimization)
**Goal**: Use RL with outcome-based reward to push the model toward correct reasoning on harder problems.

**Why GRPO**: It's well-suited for this setup — no need for a separate reward model, uses group-relative advantages from multiple completions per prompt, and the verl framework has native support.

**Data**: ~2K–5K competition-level math prompts (harder subset of Phase 1 data, plus additional competition problems). No answers needed in prompts — only used for reward verification.

**Reward function**: Binary outcome reward — 1.0 if the extracted integer matches ground truth, 0.0 otherwise. Simple, clean signal aligned with the eval metric.

**Training config**:
- Initialize from best SFT checkpoint
- Group size: 4–8 completions per prompt
- Max generation length: 4096–8192 tokens (let the model think longer)
- KL coefficient: start at 0.01, may need tuning
- LR: ~1e-6 (lower than SFT)
- Temperature: 0.7–1.0 for diverse rollouts
- Steps: 50–200 (monitor for collapse)

**Duration estimate**: ~2–3 hours per iteration, expect 2–3 GRPO iterations

**Success criteria**: Eval accuracy improves over the SFT baseline.

**Abort criteria**:
- Policy entropy crashes to near zero → entropy collapse, stop and reduce LR or increase KL
- Within-group reward std = 0 for >10 consecutive steps → all completions identical, weak signal
- Accuracy drops below SFT baseline after 50+ steps → revert to SFT checkpoint and adjust hyperparameters

---

## Time Budget (10 hours)

| Phase | Estimated Time | Cumulative |
|-------|---------------|------------|
| Data preparation + SFT setup | 0.5h | 0.5h |
| SFT training | 1.5h | 2.0h |
| Eval SFT checkpoint (wait for eval agent) | 0.5–1h | 3.0h |
| GRPO iteration 1 | 2.5h | 5.5h |
| Eval GRPO v1 (wait) | 0.5–1h | 6.5h |
| GRPO iteration 2 (adjusted) | 2h | 8.5h |
| Eval GRPO v2 + final submission | 1.5h | 10.0h |

**Buffer strategy**: If SFT takes longer than expected, reduce GRPO to 1 iteration but with more steps. Always leave 30 min for final eval + submission.

---

## Hypotheses (Ranked by Confidence)

1. **SFT is necessary before RL** — A base model cannot produce parseable math answers without supervised formatting. (High confidence)
2. **GRPO will improve over SFT alone** — RL with outcome reward encourages the model to self-correct and explore reasoning paths. (Medium-high confidence)
3. **Longer generation length helps** — AIME problems need extended reasoning; allowing 4096+ generation tokens during GRPO will outperform shorter limits. (Medium confidence)
4. **Competition-level training data transfers better than easy math** — Training on AMC/AIME-style problems will yield more improvement per example than GSM8K-level problems. (Medium confidence)
5. **Full fine-tune beats LoRA for 1.7B** — At this model size, LoRA's parameter efficiency savings are marginal and full fine-tune gives better gradient flow. (Medium confidence)

---

## Cheap Evidence to Gather Early

Before committing GPU hours, quickly verify:

1. **Tokenizer/chat template**: Inspect Qwen3-1.7B-Base's tokenizer config and chat template to ensure the SFT data format is compatible.
2. **Available Python packages**: Check what's installed (transformers, verl, vllm, deepspeed, etc.) to determine the training stack.
3. **GPU memory**: Run a quick check of GPU types and memory to size batch correctly.
4. **Base model generation**: Do a quick 5-example generation with the base model to understand its raw capability and output format — this informs how much SFT is needed.

---

## Monitoring During Training

Per the `monitor-rl-training` skill:
- **Policy entropy**: Must stay in a stable band, not crash to 0
- **Rollout prob diff**: Should be <0.005; >0.01 is a red flag
- **Grad norm**: Watch for sustained rises
- **Within-group reward std**: Must be >0 (identical completions = no learning signal)
- **Response length std**: Uniform length = early sign of collapse

---

## Success Criteria for the Experiment

- **Minimum viable**: Any non-zero accuracy on AIME 2025 (base model likely scores 0)
- **Good outcome**: 2+ problems correct out of 30 (~7%+ accuracy)
- **Strong outcome**: 4+ problems correct (~13%+ accuracy)

---

## Abort Criteria (Experiment-Level)

- If SFT produces no parseable math outputs after 3 epochs → fundamentally revisit data/format
- If GRPO shows no reward improvement after 100 steps across 2 different hyperparameter settings → submit best SFT checkpoint
- If total GPU time consumed exceeds 7 hours with no evaluated checkpoint improving on the SFT baseline → freeze and submit best available

---

## Final Submission Criteria

Submit the checkpoint with the highest evaluated AIME accuracy. If multiple checkpoints tie, prefer the one with fewer INVALID-format responses (indicating more reliable output formatting).
