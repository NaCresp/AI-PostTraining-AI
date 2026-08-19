# Cross-scaffold method audit

This supplementary analysis separates the supervised update mechanism from
the canonical Pilot Study objective label. The v3 objective analysis merges
full-parameter SFT and LoRA/PEFT under `supervised_likelihood`; this audit uses
the two frozen method-family inputs released in `analysis/annotations/strategy_level/`
(`tables/trajectory_analysis.csv` and
`annotations/episode_annotations.jsonl`) only to compare those
implementations. Other superseded v3 outputs are intentionally not retained.

Run from the repository root:

```bash
python analysis/scaffold_lockin/build_comparison.py
```

The command writes:

- `output/tables/cross_scaffold_comparison.csv`: Claude/Codex by benchmark;
- `output/tables/all_seven_task_summary.csv`: all five harness families
  aggregated over all seven Pilot benchmarks;
- `output/tables/all_cell_consistency.csv`: all 28 matched benchmark-by-base
  model cells;
- `output/tables/all_seven_task_by_base_model.csv`: full fields for all 140
  scaffold-by-benchmark-by-base-model cells;
- `output/tables/matched_three_task_summary.csv`: all five harness families
  aggregated over AIME 2025, GSM8K, and HumanEval, retained as the
  controlled-experiment subset;
- `output/tables/matched_cell_consistency.csv`: the 12 matched benchmark-by-base
  model cells in that controlled subset;
- `output/tables/matched_three_task_by_base_model.csv`: the same five-family
  comparison with the full set of fields for all 60
  scaffold-by-benchmark-by-base-model cells;
- `output/report_zh.md`: a manuscript-facing interpretation and compact table.

Method shares are conditional on a recognized method label, so every table
also reports identification coverage. Objective-switch counts retain the
canonical v3 definition and do not treat a Full-SFT/PEFT change as an objective
change. Final scores are descriptive means for trained trajectories with a
valid final metric and must be compared only within the same benchmark.
