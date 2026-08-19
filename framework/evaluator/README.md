# Evaluator Agent

Evaluation is delegated to a **second agent** running on a separate machine,
sharing only the working directory. It never trains, never edits training code,
and never touches the evaluator harness — it consumes checkpoints and produces
journal entries.

Two reasons this is an agent rather than a script:

1. **Analysis.** A score alone does not tell the training agent *where* the
   model fails. The evaluator reads the per-problem log and reports failure
   modes (format-invalid vs wrong reasoning vs arithmetic slips) and concrete
   recommendations.
2. **Calibration.** The evaluator commits to a numeric prediction *before*
   running the evaluation. The prediction/outcome gap is itself a measurement,
   and it is available to the analysis in [`../../analysis/`](../../analysis/).

## Per-checkpoint cycle

Exactly one cycle per message, then the agent stops and waits:

| Step | Action | Journal entry |
|---|---|---|
| 1. Predict | read `experiment.jsonl`, training scripts, training logs | `eval_prediction` |
| 2. Evaluate | run `evaluation/<benchmark>/submit_checkpoint.sh` | `eval_result` (or `eval_error`) |
| 3. Analyse | read the per-problem log, compare to prediction and prior checkpoints | `eval_analysis` |

The evaluator agent's conversation is preserved across messages, so it
accumulates context over a run and its predictions become better calibrated.

## Prompt

The evaluator's system prompt is `eval_program.md`, kept per benchmark next to
the harness that uses it:

- [`../../experiments/aime2025/eval_program.md`](../../experiments/aime2025/eval_program.md)
- [`../../experiments/gsm8k/eval_program.md`](../../experiments/gsm8k/eval_program.md)
- [`../../experiments/humaneval/eval_program.md`](../../experiments/humaneval/eval_program.md)
- [`../../experiments/human_guidance/eval_program.md`](../../experiments/human_guidance/eval_program.md)

The launcher (`start_eval.sh`) polls the request queue written by
`request_eval.sh`, and feeds each queued checkpoint to the evaluator agent as a
new message in the same session.

## Constraints imposed on the evaluator

- one predict → evaluate → analyse cycle per message, then stop;
- append to `experiment.jsonl` only — create no other files;
- do not modify training scripts, checkpoints, or configuration;
- do not start training runs;
- keep the analysis short — the training agent reads it in a limited context.
