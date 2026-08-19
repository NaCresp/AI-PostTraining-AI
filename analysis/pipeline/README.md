# Trajectory parsing pipeline

This directory contains the first-stage, standard-library-only data pipeline for
the public `PostTrainBench-Trajectories` corpus. It inventories canonical task
directories, parses Claude/Codex JSONL and plain-text traces, normalizes events,
extracts conservative experiment candidates and metrics, and creates layered
cohorts without deleting failed or incomplete behavior.

Run from the repository root:

```bash
python analysis/pipeline/run_pipeline.py \
  --input data/PostTrainBench-Trajectories \
  --output analysis/pipeline/output \
  --config analysis/pipeline/config.json
```

The canonical outputs are `trajectory_metadata.jsonl`, `events.jsonl`,
`experiments.csv`, `metrics.csv`, `judgements.csv`, `cohort_membership.csv`,
`cohort_counts.csv`, and `validation_report.{json,md}`. A deterministic
`run_manifest.json` records the input count, source-content digest, configuration
hash, and pipeline script hash.

Normalized free-text event fields are deterministically capped at 4,000
characters; judgement fields are capped at 20,000 characters. Truncated events record the affected field names in
`truncated_fields`; judgement rows use `judgement_parse_status=parsed_truncated`.
The canonical `task_dir` and source-line span retain the audit trail to the full
raw artifact.

Only direct child directories of each top-level batch are trajectories. A nested
`task/` directory is workspace data and is never scanned as a second trajectory.
Missing or ambiguous judgements remain `missing`/`unknown`; they are not promoted
to `clean`. The first stage does not assign strategy labels, infer intent, pool
scores across benchmarks, or perform causal/statistical analysis.
