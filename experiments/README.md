# Experiment Harness

One directory per experimental condition. Each holds the two agent prompts, the
launcher scripts, and the configuration used for the runs analysed in the paper.

| Directory | Benchmark | Condition |
|---|---|---|
| [`aime2025/`](aime2025/) | AIME 2025 (30 problems) | autonomous agent |
| [`gsm8k/`](gsm8k/) | GSM8K (1319 test problems) | autonomous agent |
| [`humaneval/`](humaneval/) | HumanEval | autonomous agent |
| [`human_guidance/`](human_guidance/) | AIME 2025 | agent receives a human-authored initial plan |

## Files

| File | What it is |
|---|---|
| `program.md` | the training agent's prompt: task, budget, rules, and the journal / skill protocol |
| `eval_program.md` | the evaluator agent's prompt (see [`../framework/evaluator/`](../framework/evaluator/)) |
| `config.yaml` | base model, GPU assignment, time budget, evaluator paths, agent model |
| `start.sh` | launches the training agent under a wall-clock `timeout`, tees the stream to `trajectory.jsonl`, seeds `.claude/skills/` into the workspace, writes `run_metadata.json` |
| `start_eval.sh` | launches the evaluator agent, polls the request queue, feeds one checkpoint per message |
| `request_eval.sh` | called by the training agent to enqueue a checkpoint for evaluation |
| `timer.sh` | wall-clock budget watchdog |

`human_guidance/` additionally contains `run_pass8_eval.py`,
`validate_human_decision.py`, `pass8_results.json` and `pass8_parts/` — the
post-hoc pass@8 re-evaluation used for the human-guidance comparison.

## How a run executes

1. `start.sh` writes `program.md` into the workspace and starts the agent with
   `--output-format stream-json` and a scoped `--allowedTools` set. The raw
   stream is captured to `trajectory.jsonl`.
2. The agent works inside its workspace for the wall-clock budget
   (`time_budget_hours`, 10 h in all released runs). The budget is enforced
   externally; the agent cannot extend it.
3. When the agent wants a checkpoint scored it calls `request_eval.sh`, then
   polls `experiment.jsonl` until the evaluator agent's `eval_result` appears.
4. `submit_final.sh` ends the run. If it is never called, the last evaluation
   stands.

## Paths

`config.yaml` and the launcher scripts contain the absolute container paths
from our cluster (`/workspace/AI4AI/...`, `/root/models/...`) and the original
directory names (`claude-code-aime`, `experiments/evaluator/aime`). They are
released **verbatim**, because the archived trajectories in
[`../trajectories/`](../trajectories/) refer to those same paths. Adjust them
for your environment before running; the mapping to this repository is:

| Original | Here |
|---|---|
| `experiments/claude-code-aime/` | `experiments/aime2025/` |
| `experiments/claude-code-gsm8k/` | `experiments/gsm8k/` |
| `experiments/claude-code-humaneval/` | `experiments/humaneval/` |
| `experiments/claude-code-human/` | `experiments/human_guidance/` |
| `experiments/evaluator/<bench>/` | `evaluation/<bench>/` |
| `run/claude-code-<bench>/run_N/` | `trajectories/{setting}_{benchmark}_{id}/` |

The agent's API key is read from the environment (`ANTHROPIC_API_KEY`); no
credentials are checked in.

The `training/` tree referenced by `verl_path` is the unmodified upstream
trainer and is not vendored here — see
[`../framework/skills/wiki/sources/`](../framework/skills/wiki/sources/).
