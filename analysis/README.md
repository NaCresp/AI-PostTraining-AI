# Analysis

The measurement code behind the paper's empirical sections, plus the annotation
tables it produces. The trajectory corpus itself (PostTrainBench-Trajectories,
1,338 agent trajectories) is not redistributed here; what is released are the
**annotations derived from it** and the code that derives them.

| Directory | What it is |
|---|---|
| [`pipeline/`](pipeline/) | harness-agnostic trajectory parsing: raw agent logs → events → experiment episodes → objective labels |
| [`pilot_study/`](pilot_study/) | the driver that runs the pipeline over the whole corpus and emits the annotation tables |
| [`annotations/`](annotations/) | the released annotation layers and summary tables |
| [`strategy_lockin/`](strategy_lockin/) | the two follow-up audits: broad-criterion strategy change, and cross-scaffold method lock-in |
| [`token_cost/`](token_cost/) | token accounting over the released trajectories |

## Corpus at a glance

Numbers below are from `annotations/objective_level/tables/validation_report.json`.

| | |
|---|---:|
| Trajectories | 1,338 |
| Benchmarks × base models (matched cells) | 7 × 4 = 28 |
| Harness families | 5 (Claude 463, OpenCode 394, Codex 369, GLM-X 84, Qwen3Max 28) |
| Base models | Qwen3-1.7B-Base, Qwen3-4B-Base, Gemma-3-4B-PT, SmolLM3-3B-Base |
| Parsed events | 847,080 |
| Execution calls | 202,795 |
| Training experiments (executed parameter updates) | 5,111 |
| Adjacent experiment pairs (transitions) | 3,557 |
| Evaluation points | 2,034 |

## Pipeline

`pipeline/` converts heterogeneous agent logs into one schema:

| Module | Role |
|---|---|
| `parse_claude.py`, `parse_codex.py`, `parse_trace.py` | per-harness log readers |
| `inventory.py` | enumerate trajectories, detect duplicates and unparsed runs |
| `extract_events.py` | normalise to a single event stream |
| `extract_experiments.py` | segment the stream into experiment episodes (a launched parameter update, not merely a proposed one) |
| `extract_judgements.py` | recover the agent's own stated judgements |
| `extract_metrics.py` | recover reported scores |
| `filter_cohorts.py` | derive the analysis cohorts |
| `schema.py` | the record definitions all of the above share |

Unit tests: `python -m unittest discover -s analysis/pipeline/tests -v`

## Cohorts

Not every trajectory can answer every question, so each result is reported over
an explicit cohort rather than over the full corpus:

| Cohort | n | Admits |
|---|---:|---|
| `all_inventory` | 1,338 | every task-level trajectory, including failed and unparsed runs |
| `behavior_cohort` | 1,300 | trajectories whose logs parsed |
| `artifact_ready` | 992 | trajectories that produced inspectable training artifacts |
| `objective_ready` | 792 | trajectories with ≥1 objective-labelled executed experiment |
| `strategy_ready` | 792 | trajectories admissible for strategy-level statistics |
| `strategy_outcome_ready` | 654 | the above **and** a valid final score |

Transition statistics compare only **adjacent** experiments where *both*
objectives are known. Unknown experiments are excluded and never bridged, and a
proposed-but-unexecuted change is never counted as a change. Both conventions
make the reported change rates lower bounds.

## Reproducing

The driver expects the trajectory corpus at `data/PostTrainBench-Trajectories`:

```bash
python analysis/pilot_study/run_pilot.py \
  --input  data/PostTrainBench-Trajectories \
  --output analysis/pilot_study \
  --config analysis/pilot_study/config.json
```

It writes `intermediate/`, `tables/` and `annotations/` under the output
directory. The `intermediate/` tree (~2 GB of per-event records) is **not**
released; the `tables/` and `annotations/` it produces are, under
[`annotations/`](annotations/).
