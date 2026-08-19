# Annotations

Two annotation layers over the same 1,338 trajectories. They are separate
passes, not versions of one another — keep them distinct when citing.

| Layer | Question it answers | Directory |
|---|---|---|
| Objective level | *What optimisation objective did this experiment use, and did it change?* | [`objective_level/`](objective_level/) |
| Strategy level | *What full strategy — objective, update mechanism, data regime — did the agent commit to, and how did it develop?* | [`strategy_level/`](strategy_level/) |

## Objective level

The canonical label is the optimisation objective of an **executed**
parameter-update experiment:

`supervised_likelihood` · `reward_optimization` · `preference_optimization` ·
`on_policy_distillation` · `objective_unknown`

Full-parameter and parameter-efficient supervised training share the
`supervised_likelihood` label — switching from Full SFT to LoRA is *not* an
objective change. Data construction, hyperparameter tuning, debugging,
evaluation, and checkpoint operations are not experiments.

| File | Contents |
|---|---|
| `episode_annotations.jsonl` | one record per experiment episode: objective form, evidence, confidence, action categories, position in the trajectory |
| `trajectory_annotations.jsonl` | one record per trajectory: benchmark, base model, harness family, initial objective, experiment count, objective-change count |
| `annotation_summary.csv` | coverage and correction rates for the annotation pass |
| `consistency_check_log.jsonl` | the deterministic consistency checks and their outcomes |
| `tables/` | the aggregate tables (see below) |

`annotation_summary.csv` records the honest caveat about this layer: objectives
are extracted automatically and then checked deterministically — **this is not
independent double annotation**. Coverage is 1,338/1,338, corrections 0, and
8.07% of experiments remain `objective_unknown`.

### Tables

| File | Contents |
|---|---|
| `trajectory_analysis.csv` | the per-trajectory master table, 1,338 rows × 72 columns |
| `summary_by_benchmark.csv` | per-benchmark trajectory counts, activity, experiments, objective-change rate |
| `summary_by_base_model.csv`, `summary_by_harness_family.csv` | the same cut by base model / harness |
| `cohort_counts.csv`, `cohort_membership.csv`, `cell_coverage.csv` | cohort definitions and membership |
| `rq1_initial_strategy.csv` | distribution of initial objectives |
| `rq2_online_development.csv` | objective-change rate per harness family over valid adjacent pairs |
| `rq4_outcome_by_strategy.csv` | final outcome broken down by strategy |
| `objective_transitions.csv` | one row per adjacent experiment pair |
| `evaluation_events.csv` | one row per evaluation point (2,034 total) |
| `validation_report.json` | corpus-level counts and parser statistics for the whole run |

## Strategy level

A second pass that decomposes each episode into its components rather than
collapsing them to the objective:

`strategy_family` · `objective_or_reward` · `update_mechanism` · `data_regime` ·
`method_family` · `proposal_only` / `proposal_reference`

`proposal_only` is what makes proposed-but-never-executed strategies visible —
the gap between what the agent said it would try and what it ran.

| File | Contents |
|---|---|
| `episode_annotations.jsonl` | one record per episode with the decomposed labels |
| `tables/trajectory_analysis.csv` | 1,338 rows × 81 columns |

The 9 columns present here and not at the objective level:

`initial_strategy_family`, `local_refinement_count`, `core_transition_count`,
`proposal_count`, `proposal_to_execution_count`, `proposal_to_execution_rate`,
`never_switched`, `local_development_ratio`, `first_transition_progress`

The distinction between `core_transition_count` and `local_refinement_count` is
the one the paper's strategy-persistence metrics rest on: most observable
activity is local refinement of a committed strategy rather than a move to a
different one.

**The two layers do not agree everywhere.** Over the 72 shared columns, 65 of
1,338 trajectories (4.9%) differ between the two passes. The strategy-level pass
re-derived those labels; it is not a strict superset of the objective-level
pass. Cite whichever layer a given number came from.

## Provenance

Trajectory identifiers refer to paths under `data/PostTrainBench-Trajectories`.
Absolute paths from the annotators' machines have been rewritten to
repository-relative form; nothing else in the data was modified.

The raw corpus is not redistributed. The subset of our own runs that the paper
draws on directly is archived in [`../../trajectories/`](../../trajectories/).
