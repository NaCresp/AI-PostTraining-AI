# Evaluation Harness

The scoring code the evaluator agent invokes. It is deliberately outside the
agent's writable workspace: the agent may read its own results but cannot
modify how it is scored.

| Directory | Benchmark | Metric |
|---|---|---|
| [`aime2025/`](aime2025/) | AIME 2025, 30 problems | accuracy |
| [`gsm8k/`](gsm8k/) | GSM8K test split, 1319 problems | accuracy |
| [`humaneval/`](humaneval/) | HumanEval | pass@1 (generated code executed against the tests) |

Each directory contains:

- `evaluate.py` — loads an FSDP-sharded or merged HF checkpoint, generates with
  vLLM, extracts answers, and emits a JSON score plus a per-problem JSONL log;
- `submit_checkpoint.sh` — the entry point the evaluator agent calls;
- `submit_final.sh` — the training agent's final submission.

## Sampling

All three run **one sample per problem** (`n_samples = 1`) during a run, so the
in-run score is accuracy / pass@1. Answer extraction:

| Benchmark | Extraction | Failure label |
|---|---|---|
| AIME 2025 | regex `(?i)Answer\s*:\s*([^\n]+)` over the last 300 characters | `INVALID` |
| GSM8K | `#### <number>` | `INVALID` |
| HumanEval | completion spliced into the prompt and executed | test failure |

The `INVALID` rate is reported separately from wrong answers, because early
checkpoints frequently fail on format rather than on reasoning — a distinction
the evaluator agent is asked to act on.

## pass@8

pass@8 on AIME 2025 is a **post-hoc** re-evaluation of selected checkpoints,
not part of the in-run signal. It is driven by
[`../experiments/human_guidance/run_pass8_eval.py`](../experiments/human_guidance/run_pass8_eval.py),
which re-invokes `aime2025/evaluate.py` with `n_samples = 8` over multiple
seeds; the shard outputs are in `../experiments/human_guidance/pass8_parts/`
and the aggregate in `pass8_results.json`.

## Benchmark data

The benchmark inputs (AIME 2025 problems, GSM8K test split, HumanEval) are not
redistributed. Point the loaders in `evaluate.py` at your own copies.
