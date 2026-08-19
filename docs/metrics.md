# Metric definitions

Three metrics carry the paper's argument: an **execution** metric that shows agents are competent,
and two **strategy** metrics that show they do not revise. This document defines all three, states
the cohorts they are computed over, and points at the released files that reproduce them.

---

## Cohorts

Not every trajectory can answer every question. A trajectory that crashed in its first ten minutes
has behavior worth counting but no checkpoint to score; a trajectory whose training configuration
cannot be resolved cannot contribute a strategy label. Rather than silently dropping rows per
statistic, the pipeline materializes named cohorts and each result states which one it uses.

| Cohort | Size | Admission criterion |
|---|---:|---|
| `all_inventory` | 1,338 | Every trajectory in the corpus |
| `behavior_cohort` | 1,300 | Trajectory produced a parseable event stream |
| `artifact_ready` | 992 | Produced at least one durable artifact (checkpoint, dataset, or score) |
| `objective_ready` | 792 | Carries a resolved training-objective label |
| `strategy_ready` | 792 | Carries a complete strategy state `s = (k, d, g)` |
| `strategy_outcome_ready` | 654 | Complete strategy state **and** a scored outcome |
| `matched_cell` | 1,338 | Belongs to one of the 28 matched (scaffold × model × benchmark) cells |

The cohorts nest: `strategy_outcome_ready` ⊂ `strategy_ready` ⊆ `objective_ready` ⊂
`artifact_ready` ⊂ `behavior_cohort` ⊂ `all_inventory`. `matched_cell` is a separate partition of
the full inventory into 28 cells, run over 47 batches, used for like-for-like scaffold comparisons.

`filter_cohorts.py` in `analysis/pipeline/` defines these; the cohort of each trajectory is a
column in the released annotation tables, so any statistic can be re-scoped.

---

## Execution: per-strategy success rate

For a scaffold and a training strategy, the success rate is the fraction of launched experiments
that complete and produce a usable checkpoint:

```
success_rate = (experiments that completed and produced a checkpoint) / (experiments launched)
```

This is deliberately generous. It asks only "did the job run and yield something scoreable," not
"did it help." That is the point: it isolates mechanical competence from judgment.

| Scaffold | Strategy | Success rate |
|---|---|---|
| Claude Code | Full supervised fine-tuning | **163 / 202 = 80.7%** |
| Codex | Parameter-efficient fine-tuning | **268 / 299 = 89.6%** |

Agents configure distributed trainers, fit models onto a single 80GB card, convert datasets into
the right on-disk schema, and recover from crashes, at rates in the eighties and nineties. Nothing
in the corpus suggests execution is the bottleneck.

---

## Strategy: convergence of the initial strategy

How much do independent agents agree on where to start? For a group of trajectories `G`, let
`k_{i,1}` be the training strategy of trajectory `i`'s **first** launched experiment. Then

```
                1
C_G  =  max_k  ───  Σ   1[ k_{i,1} = k ]
               |G|  i∈G
```

`C_G` is the share of the group taking the single most common opening move. It ranges from `1/|K|`
(agents spread evenly over the available strategies) to `1` (every agent opens identically).

`C_G` is computed over the **783 trajectories with a recognized initial strategy** (of the 900 that
launch training at all). It is reported per group — per scaffold, per base model, per benchmark —
because the interesting question is whether the convergence is a property of the agent or of the
task.

A high `C_G` is not by itself a defect. If one opening is genuinely best, agreeing on it is
correct. `C_G` becomes evidence only in combination with the next metric: converging on an opening
*and then never revising it* is what makes the convergence a lock-in rather than a consensus.

---

## Strategy: change and persistence rates

Let `P_G` be the set of temporally adjacent experiment pairs `(i, t-1) → (i, t)` within
trajectories in `G`. Then

```
              | { (i,t) ∈ P_G : s_{i,t} ≠ s_{i,t-1} } |
R_change(G) = ─────────────────────────────────────────
                             | P_G |
```

```
R_persist(G) = 1 − R_change(G)
```

`R_change` is the probability that an agent's strategy state differs from one experiment to the
next. `R_persist` is the complement — the probability it carries the previous state forward
unchanged.

Over the full corpus:

| Quantity | Value |
|---|---:|
| Adjacent experiment pairs `|P_G|` | 3,557 |
| Strategy changes | 74 |
| `R_change` | **2.1%** |
| `R_persist` | **97.9%** |
| Trajectories contributing ≥1 change | 44 of 792 |

Decomposed by which component of `s = (k, d, g)` moved:

| Component | Changes |
|---|---:|
| Training objective `k` | 35 |
| Data source `d` | 38 |
| Stage structure `g` | **1** |

A pair may change in more than one component, so the decomposition is not a partition of 74.

The stage-structure figure is the sharpest result in the paper. Across 1,338 trajectories, ten
hours each, 5,111 launched experiments, agents altered the *shape* of their training pipeline
exactly once. Whatever pipeline an agent commits to in its first hour is, with one exception, the
pipeline it dies with.

### What `R_change` is not measuring

`R_change` is defined over the strategy state only. It is deliberately blind to hyperparameter
edits, data-volume changes, checkpoint selection, and prompt-format fixes — all of which agents do
constantly and competently. The 97.9% persistence figure does **not** say agents are idle between
experiments. It says the thing they change is never the strategy.

Nor is it a measure of *correct* revision. It counts changes, not improvements. A change is
counted whether it helped or hurt; the paper's claim is about the near-total absence of changes,
which no accounting of their quality can explain away.

---

## Recomputing these numbers

```bash
# Per-scaffold lock-in breakdown, from the released annotations
python analysis/strategy_lockin/scaffold_lockin/build_comparison.py

# Objective transitions and evaluation events; needs the regenerated intermediate/ tree
python analysis/strategy_lockin/broad_criterion/build_broad_transitions.py
```

The first reads `analysis/annotations/strategy_level/` and writes its tables and `report.md` under
`scaffold_lockin/output/`. The second also needs `analysis/pilot_study/intermediate/`, which is not
redistributed; regenerate it with `run_pilot.py` first. Note that
`broad_criterion/report.md` reports 73 changes over 43 trajectories rather than 74 over 44; that
gap is one adjudicated stage-structure candidate and is explained in
[`annotation-protocol.md`](annotation-protocol.md#reconciling-73-vs-74).

To recompute cohorts and tables from raw trajectory logs:

```bash
python -m analysis.pipeline.run_pipeline --config analysis/pipeline/config.json
```

This requires the raw logs, which are terabyte-scale and not released. The ten runs in
`trajectories/` are complete enough to run the parsers end-to-end and verify the extraction stages
against real input, in both of the stream formats the pipeline accepts.
