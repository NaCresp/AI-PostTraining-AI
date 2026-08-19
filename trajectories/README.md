# Archived Trajectories

Ten runs of our own controlled experiments — the ones the paper's qualitative
claims point at. Each directory is the agent's working directory as it stood
when the run ended, minus the artifacts that cannot be redistributed.

Directories are named `{setting}_{benchmark}_{id}`. The setting is the condition
the agent ran under — `experience` (the experience-driven framework: experiment
journal, skill library, evaluator agent), `autonomous` (no framework, the
baseline), or `human` (a human reviews the agent's plan before the autonomous
phase). The id is the first 8 hex characters of a manifest digest over the
released directory: `sha256` of the sorted `<relative path>\0<sha256 of file>`
lines for every file it contains. It depends only on released content, so it can
be recomputed from a checkout.

| Run | Setting | Scaffold | Benchmark | Wall clock | Journal entries | Evaluations | Score |
|---|---|---|---|---:|---:|---:|---:|
| [`experience_aime2025_c247e78e/`](experience_aime2025_c247e78e/) | experience | Claude Code | AIME 2025 | 6h 09m | 35 | 5 | 0.033 |
| [`experience_gsm8k_855100f4/`](experience_gsm8k_855100f4/) | experience | Claude Code | GSM8K | 4h 50m | 74 | 11 | 0.773 |
| [`experience_humaneval_d26cfb40/`](experience_humaneval_d26cfb40/) | experience | Claude Code | HumanEval | 7h 12m | 72 | 14 | 0.628 |
| [`human_aime2025_14ce9dfd/`](human_aime2025_14ce9dfd/) | human | Claude Code | AIME 2025 | 8h 57m | 59 | 9 | 0.067 |
| [`autonomous_aime2025_250c7e3e/`](autonomous_aime2025_250c7e3e/) | autonomous | Claude Code | AIME 2025 | 8h 53m | — | — | — |
| [`autonomous_gsm8k_42890926/`](autonomous_gsm8k_42890926/) | autonomous | Claude Code | GSM8K | 10h 06m | — | — | — |
| [`autonomous_humaneval_fc9a969e/`](autonomous_humaneval_fc9a969e/) | autonomous | Claude Code | HumanEval | 10h 06m | — | — | — |
| [`autonomous_aime2025_c7c4a0c4/`](autonomous_aime2025_c7c4a0c4/) | autonomous | Codex | AIME 2025 | 7h 24m | — | — | 0.000 |
| [`autonomous_gsm8k_5dcf0c73/`](autonomous_gsm8k_5dcf0c73/) | autonomous | Codex | GSM8K | 3h 02m | — | — | 0.537 |
| [`autonomous_humaneval_832f94ce/`](autonomous_humaneval_832f94ce/) | autonomous | Codex | HumanEval | 5h 33m | — | — | 0.134 |

All ten runs post-train Qwen3-1.7B-Base on 4 GPUs under a 10-hour budget.
`run_metadata.json` in each directory records the exact start/end times and
configuration. Several runs ended before their budget; where they did, the
reason is in the stream's closing records.

**The score column is not a controlled comparison.** It has three different
meanings. The four framework and human runs report the in-run metric produced by
our own evaluator agent, one sample per problem. The three Codex runs report the
benchmark harness's final evaluation of the model the agent submitted. The three
Claude Code autonomous runs have no official score: the harness's final
evaluation never started for that batch, so nothing was recorded. What those
three do have is the agent's own evaluations — 7, 5 and 9 of them, condensed
into `evals/eval_log_summary.json`, run on subsets the agent chose for itself
(20 to 150 problems) and ending at 0.000 on AIME 2025, 0.647 on GSM8K and 0.050
on HumanEval. Those are the agent's own view of its progress, not benchmark
results; the HumanEval agent's best subset score along the way was 0.35, and it
finished below it.

If you want the autonomous condition held to the same scaffold as the framework
runs, use the three Claude Code rows: the Codex runs differ from everything else
here in scaffold as well as in setting. The paper's quantitative claims rest on
the annotated corpus in [`../analysis/annotations/`](../analysis/annotations/),
not on these ten runs. The post-hoc pass@8 numbers for the AIME comparison are in
[`../experiments/human_guidance/pass8_results.json`](../experiments/human_guidance/pass8_results.json).

## What each directory contains

The framework and human runs came out of our own harness
([`../experiments/`](../experiments/)):

| Path | What it is |
|---|---|
| `trajectory.jsonl` | the training agent's full Claude Code stream (`--output-format stream-json`) — every message, tool call, and result |
| `eval_trajectory.jsonl` | the evaluator agent's stream |
| `experiment.jsonl` | the Experiment Journal — both agents append here (schema: [`../framework/journal/`](../framework/journal/)) |
| `run_metadata.json` | start/end, elapsed seconds, exit code, config |
| `program.md` | the prompt this run was given |
| `.claude/skills/` | the skill library as seeded into this workspace |
| `eval_agent_logs/` | per-checkpoint evaluator logs |
| `logs/`, `outputs/` | training logs and generation dumps, where the run produced them |
| `*.py`, `*.sh` | **agent-authored** training scripts, reward functions, and data preparation |
| `.eval_queue`, `.eval_processed`, `.timer` | the evaluation-request queue and the budget watchdog's state |

`human_aime2025_14ce9dfd/` additionally has `planning_trajectory.jsonl` (the
planning agent's stream), `human_plan_iteration_{1,2}.md` (the two proposed
plans), `human_guidance.jsonl` and `.human_decision_{1,2}.json` (the reviewer's
verdicts and stated reasons). The human reviewer rejected iteration 1 for
over-weighting SFT and accepted iteration 2; the 10-hour budget starts only
after acceptance, and the 793 s of human wait time is excluded from it.

The six autonomous runs came out of the benchmark harness instead, so their
layout is different and deliberately sparser — none of the framework's machinery
is present, which is the point of the condition:

| Path | What it is |
|---|---|
| `trajectory.jsonl` | the agent's stream as the harness recorded it |
| `program.md` | the prompt this run was given |
| `run_metadata.json` | elapsed, start/end, base model, budget, how the run ended, and final metrics where any exist |
| `metrics.json` | the harness's final evaluation of the submitted model — Codex runs only |
| `scripts/` | **agent-authored** data preparation, training and evaluation scripts |
| `logs/` | the training and evaluation logs the agent produced |
| `evals/` | the agent's own evaluation results, and `eval_log_summary.json` condensing its full evaluation logs |
| `*.md` | the agent's own notes, where it wrote any |
| `contamination_judgement.txt`, `disallowed_model_judgement.txt` | the harness's compliance verdicts on the run |
| `timer.sh` | the budget watchdog the agent could call |

There is no journal, no skill library, no evaluator agent, and no queue — an
autonomous agent has only its own context. `autonomous_aime2025_c7c4a0c4/` kept
no evaluation results at all; it submitted a model scoring 0.000.

Checkpoints, training data and generation dumps are excluded from every
directory (see below), so `scripts/` records what the agent wrote rather than
what it produced.

## What was removed

| Removed | Why |
|---|---|
| `checkpoints/`, `final_model/`, `sft_output*/` | model weights, up to 331 GB per run |
| `data/`, `train_data*.jsonl`, `tokenized_data*/` | agent-generated training data, up to 97 GB per run |
| `pass8_eval/` | 16 GB of post-hoc generation dumps |
| `judge_output.*`, `task/logs/` | per-sample judging and evaluation dumps, up to 150 MB per run; the scores in them survive in `evals/eval_log_summary.json` |
| `templates/` | chat templates supplied by the task environment, not written by the agent |
| `solve_parsed.txt`, `system_monitor.log`, harness console logs | derived from the stream, or machine-level telemetry |
| `ANALYSIS.md` | our own post-run notes, not part of the run |
| `.eval_session_id` | agent session identifier, no analytic value |
| `__pycache__/` | build artifacts |

Everything else is byte-for-byte as the run left it, with two exceptions:

**1. Skill renaming.** Three skills were named after the specific trainer used
in our runs and are released under the generic names used in the paper. The
rename is applied to directory names *and* to every reference inside
`trajectory.jsonl`, `eval_trajectory.jsonl`, the evaluator logs, and the skill
files themselves, so the archives stay internally consistent:

| Original | Here |
|---|---|
| `verl-known-pitfalls` | `posttraining-known-pitfalls` |
| `verl-parquet-schema` | `data-parquet-schema` |
| `grpo-lora-verl` | `grpo-lora-config` |

Only the identifier changed; skill contents are untouched. The third skill was
authored in an earlier AIME run and does not appear in these archives, and the
autonomous runs carry no skills at all.

**2. Redactions.** Three kinds, all mechanical:

*Credentials.* Each Codex run's stream opens with the harness echoing the agent
provider and its API key. That one token per file is replaced with
`sk-[REDACTED]`. The other seven runs contain none.

*Account balances.* Our API provider returns `403` quota errors whose message
text quotes the account balance. In
`experience_humaneval_d26cfb40/trajectory.jsonl` the four such messages are
replaced wholesale with
`403 pre-charge authorization failed (insufficient account balance)` — this is
why that run terminated at turn 411 rather than on its time budget. Elsewhere
only the figures are replaced, with `$[REDACTED]`, leaving the message
otherwise intact: 36 figures across 12 files, being `trajectory.jsonl` and
`eval_trajectory.jsonl` for `experience_aime2025_c247e78e` and
`experience_gsm8k_855100f4` plus their per-checkpoint evaluator logs, and 4 more
in `autonomous_aime2025_250c7e3e/trajectory.jsonl`, where the same error ended
the run at 8h 53m. Error types, timings and turn counts are unchanged
throughout.

*Host paths.* `experience_aime2025_c247e78e/trajectory.jsonl` contains a
directory listing of a shared model store in which three symlink targets carry
our cluster account name; those are rewritten to `/home/user`. The autonomous
runs reference the base model by its absolute path on the same store, rewritten
the same way — 36, 24 and 79 occurrences in the Codex runs, 35, 55 and 66 in the
Claude Code ones. Container-internal paths (`/workspace/AI4AI/...`,
`/home/ben/task/...`) are left alone — the evaluator log filenames encode them,
so rewriting them would desynchronize the archive from itself.

## Reading a trajectory

[`../analysis/pipeline/`](../analysis/pipeline/) has a parser for each stream
format. Both return `(records, stats)` and take file *contents*, not a path.

For the framework and human runs, `trajectory.jsonl` is one JSON object per line
in Claude Code's stream format — `system` init, then alternating `assistant` /
`user` records carrying tool calls and their results, then a final `result`
record:

```python
from pathlib import Path
from analysis.pipeline.parse_claude import parse_claude_records

records, stats = parse_claude_records(
    Path("trajectories/experience_aime2025_c247e78e/trajectory.jsonl").read_text()
)
```

The three Claude Code autonomous runs carry the same records, but as the
benchmark harness wrote them: every JSON line is prefixed with a wall-clock
stamp, `[2026-06-09T06:26:38Z] {...}`, and the container's startup banner
occupies the first 15 lines. `parse_claude_records` reads them as they are — it
scans past any prefix to the first `{` and skips lines that hold no JSON object,
reporting them in `stats["warnings"]`.

The Codex runs keep that harness's own event format instead, stamped the same
way, with `thread.started`, `turn.*` and `item.*` events in place of Claude's
message records:

```python
from analysis.pipeline.parse_codex import parse_codex_records

records, stats = parse_codex_records(
    Path("trajectories/autonomous_gsm8k_5dcf0c73/trajectory.jsonl").read_text()
)
```

These are the same formats, and the same parsers, that the pipeline uses for the
wider corpus — `config.json` lists `solve_out.txt` and `trace.txt` as the two
stream sources, and the six autonomous files are the former, renamed.

One archive does not end on a clean JSON line. When our watchdog reaches the
wall-clock budget it kills the agent process, and the shell's `Killed` notice is
teed into the stream: `human_aime2025_14ce9dfd/trajectory.jsonl` line 1003 is
exactly that. Parsers must skip lines that fail to decode — the ones in
`analysis/pipeline/` do — and that line is also how you tell a budget-exhausted
run from one that stopped on its own.

For the decision-level view of a framework run, read `experiment.jsonl` instead
— it is two orders of magnitude smaller and holds the agent's own account of
what it planned, what it expected, and what it concluded. The autonomous runs
have no equivalent; that absence is the finding.

## Scope

These ten runs are the controlled arm of the study. The broader corpus the
statistics in the paper are computed over — 1,338 trajectories across 5 agent
harnesses, 7 benchmarks and 4 base models — is not redistributed here; its
annotations are in [`../analysis/annotations/`](../analysis/annotations/). The
autonomous runs were recorded by the same benchmark harness that produced that
corpus, which is why the pipeline's parsers read them unchanged, but they are
separate local runs and are not among the 1,338.

The AIME campaign ran nine times under the framework. The ninth, with all three
components in place, is the one archived here; the additional ten skills in
[`../framework/skills/skills/`](../framework/skills/skills/) were written by the
agent during runs 3, 4, 6 and 7, which are not included.
