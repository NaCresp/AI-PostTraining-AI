# RL Training Principles (Framework-Agnostic)

Universal practices for stable LLM RL post-training. Apply regardless of stack (verl, TRL, OpenRLHF, slime, NeMo RL).

## Before You Train

1. **SFT cold start** when the task needs format or tool use — RL alone rarely invents valid structure. Run eval on the SFT checkpoint before RL.
2. **Pin dependency versions** — `transformers` major bumps often change tokenizer APIs, RoPE helpers, and nested-tensor behavior without crashing.
3. **Establish a synchronous baseline** — get on-policy (or one-step-off) training healthy before fully async pipelines.
4. **Enable rollout vs training log-prob comparison** — without per-token log probs from both engines, train–infer mismatch is invisible.

## GRPO / Group-Based RL Defaults

| Knob | Recommended starting point | Notes |
|------|---------------------------|-------|
| Group size `n` | 8–16 (up to 64 if memory allows) | Must be >1; variance within group drives the signal |
| Clip ε | 0.2 | DAPO-style exploration: asymmetric low 0.2 / high 0.28 |
| Loss aggregation | **Token-mean** | Seq-mean-token-mean is unstable for long CoT; DrGRPO uses a different norm on purpose |
| KL to reference | `low_var_kl` (k3), coef **0.001** | Do **not** use KL-in-reward and KL-in-loss together |
| Learning rate | ~1e-6 full FT, ~1e-5 LoRA | LoRA updates fewer params — higher LR is normal |
| Entropy bonus | 0 (default in many stacks) | Raise only after other collapse mitigations fail |
| Reward scale | Binary 0/1 with KL coef 0.001 | Rescale KL if rewards are not O(1) |

**TRL-specific:** `scale_rewards=False` reduces question-level difficulty bias; `scale_rewards="batch"` uses group mean + global std (see DrGRPO / Lite PPO literature).

## Runtime Monitoring

| Metric | Healthy | Investigate when |
|--------|---------|------------------|
| Policy entropy | Stable band, not monotonic crash to ~0 | → [Training Collapse Triage](training-collapse-triage.md) |
| `rollout_probs_diff_mean` (or equivalent) | < 0.005 | 0.005–0.01 borderline; > 0.01 severe mismatch |
| `grad_norm` | Stable / slow drift | Sustained rise → mismatch or LR too high |
| Within-group reward std | > 0 | = 0 → identical completions, weak GRPO signal |
| Response length std | Has variance | Uniform length → late entropy collapse |

## Throughput vs Algorithm Parameters

- **Safe to tune for speed (usually no algorithm change):** per-GPU micro-batch, max tokens per GPU, dynamic batching, sequence packing, activation offload, rollout GPU count.
- **Affects convergence — change one at a time:** global train batch, mini-batch size, PPO/GRPO epochs, learning rate, group size `n`, clip range, KL coef, loss aggregation mode.

> **Golden rule:** RL stacks are fragile. Change **one** hyperparameter or feature per experiment; otherwise root cause is unknowable.

## Train–Inference Alignment

Rollout (vLLM/SGLang/HF generate) and training forward often differ in precision, attention kernels, or MoE routing.

1. Log probs from **both** sides on the same tokens.
2. Track mean absolute prob diff; thresholds above apply.
3. Mitigations (in order): match dtype, disable known-bad inference kernels (e.g. cascade attention on non-Hopper), [Rollout Correction](../concepts/rollout-correction.md), reduce async staleness.

**Agent / multi-turn:** prefer token-in-token-out trajectories — re-tokenizing concatenated text breaks on-policy assumptions (OpenRLHF, verl AgentLoop, NeMo chat datasets).

## Async Training (When Stable Sync Works)

- Consider async when rollout wall time **> ~50%** of step time.
- Target **rollout duration ≈ training duration** (minimize pipeline bubbles).
- Start with **low staleness** (e.g. one-step off-policy); increase only with IS correction and health metrics (`rollout_is_mean` ≈ 1, effective sample size > 0.3).
- Partial rollout / interrupt-safe agents matter for multi-turn tools.

Details: [Async Training Guide](async-training-guide.md), [Performance Tuning](performance-tuning.md).

## Reward Design (Summary)

- Verifiable tasks: sparse 0/1 on **final** answer after normalization; for `\boxed{}` math, scan only the **last ~300 characters** of the response.
- Mixed datasets: every `data_source` must map to a reward — missing routes fail silently.
- Overlong shaping: linear penalty near context limit beats hard truncation for length control (DAPO pattern).
- Outcome-only rewards can reinforce **flawed positives** — add GenRM / process checks when reasoning quality matters ([FAPO](../entities/fapo.md)).

Full guide: [Reward Function Design](reward-function-design.md).

## Silent Failures (No Crash, Wrong Training)

Check these before tuning LR or algorithms:

| Symptom | Likely cause |
|---------|----------------|
| Rewards stuck at zero | Tokenizer/chat-template API change; reward routing bug |
| Metrics OK, eval bad | Train–infer mismatch; wrong attention mask / padding path |
| VLM quality drop, no error | Fused kernels + sequence parallel; retokenization on images |
| LoRA “trained” but eval = base | Exported `lora_alpha=0` or merge dropped adapter scale |
| NCCL hang, no traceback | Dynamic batching shards differ across ranks without shared `dp_group` |
| 0% accuracy right after weight sync | Stale or dummy rollout weights — validate generations post-sync |

Framework-specific bug tables: [verl Known Bugs](verl-known-bugs.md).

## Algorithm Escalation Path

| Situation | Consider |
|-----------|----------|
| First math RL run | GRPO + KL |
| Entropy collapse / plateau | DAPO (clip-higher, dynamic sampling, overlong buffer) |
| Runaway response length | DrGRPO loss normalization |
| MoE / rare-token instability | DPPO (divergence mask vs ratio clip) |
| Correct answer, bad reasoning | FAPO / GenRM |
| Strong frozen teacher | On-policy distillation |

Details: [Advanced Algorithms](advanced-algorithms.md).

## Derived From

- Raw: verl-docs (GRPO/PPO/async perf), verl-recipe, TRL grpo_trainer, OpenRLHF agent docs, slime/Miles quick start, NeMo RL GRPO + sequence packing docs
- [GRPO](../entities/grpo.md), [Loss Aggregation](../concepts/loss-aggregation.md), [Training-Inference Mismatch](../concepts/training-inference-mismatch.md)
- [Batch Size Tuning](../concepts/batch-size-tuning.md), [Async Training](../concepts/async-training.md)
