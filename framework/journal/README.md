# Experiment Journal

`experiment.jsonl` is a newline-delimited JSON file in the agent's working
directory. It is the agent's externalised memory: the context window is finite,
the journal is not. The agent is instructed to write to it at every decision
point, and to read it back before planning the next experiment.

The evaluator agent appends to the **same** file, so the journal is also the
channel through which evaluation feedback reaches the training agent.

## Entry types written by the training agent

### `plan` — before any training or data preparation

```json
{"timestamp": "<ISO 8601>", "type": "plan",
 "description": "what I'm about to do",
 "hypothesis": "why I think this will work",
 "success_criteria": "what result would confirm the hypothesis",
 "abort_criteria": "what would make me stop and change direction"}
```

Requiring `hypothesis` / `success_criteria` / `abort_criteria` *before* the GPU
is touched is what makes strategy commitment observable: the paper's
strategy-level annotations are anchored on these entries.

### `lesson` — after reading an evaluation result

```json
{"timestamp": "<ISO 8601>", "type": "lesson",
 "eval_checkpoint": "which checkpoint",
 "lesson": "one sentence: what did this result teach me",
 "next_action": "what I will do differently based on this"}
```

### `reflection` — at milestones, new conclusions, or surprises

```json
{"timestamp": "<ISO 8601>", "type": "reflection", "hours_elapsed": 3.5,
 "what_worked": "...", "what_didnt": "...",
 "new_hypothesis": "why the next approach should be different"}
```

### `skill_created` — whenever a skill is added or updated

```json
{"timestamp": "<ISO 8601>", "type": "skill_created",
 "skill": "name", "reason": "what problem it solves"}
```

### `observation` — optional, for anything unexpected

Free-form; the agent is encouraged to use it liberally.

## Entry types written by the evaluator agent

| Type | Written | Contents |
|---|---|---|
| `eval_prediction` | before running the evaluator | predicted score + 1–3 sentences of reasoning |
| `eval_result` | after running the evaluator | raw evaluator JSON (score, invalid rate, error breakdown) |
| `eval_analysis` | after reading the per-problem log | predicted vs actual, gap reason, failure modes, recommendations |
| `eval_error` | on evaluation failure | error message and the debugging steps attempted |

See [`../evaluator/`](../evaluator/) for the exact schemas.

## Protocol

1. The training agent writes `plan`, then trains.
2. It calls `request_eval.sh <checkpoint>` and **polls `experiment.jsonl`** until
   a matching `eval_result` appears. It must not start the next training run
   before then.
3. It reads `eval_analysis`, writes a `lesson`, and only then plans again.

This serialisation is what turns a sequence of training runs into a sequence of
*hypothesis tests*, and is the unit of analysis for the metrics in
[`../../docs/metrics.md`](../../docs/metrics.md).

## Reading the archived journals

Every archived run in [`../../trajectories/`](../../trajectories/) ships its
`experiment.jsonl`. Parsing helpers live in
[`../../analysis/pipeline/`](../../analysis/pipeline/).
