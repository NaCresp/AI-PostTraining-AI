# Strategy Lock-In

Two audits that test whether the low objective-change rate measured by the main
annotation pass is an artifact of a narrow definition, and whether the strategy
an agent commits to is explained by the task or by the agent's own priors.

## `broad_criterion/`

Widens the definition of a strategy change from *paradigm only* to
*paradigm ∪ data source ∪ stage structure*, over the same denominator of 3,557
adjacent experiment pairs.

Result: 2.05% (73/3,557) under the broad criterion versus 0.98% (35/3,557) under
the paradigm criterion. Widening the definition roughly doubles the rate and
leaves the conclusion intact.

See [`broad_criterion/report.md`](broad_criterion/report.md), which also records
the two earlier rule sets that were tried and rejected, and why.

## `scaffold_lockin/`

Holds benchmark, base model, and compute budget fixed and varies only the
scaffold. If task demand alone determined the reasonable strategy, matched cells
should push different scaffolds towards the same implementation.

Result: they do not. Claude's initial method is Full SFT in 163/202 (80.7%) of
identifiable cases; Codex's is LoRA/PEFT in 268/299 (89.6%). The direction holds
in 28/28 matched cells. Both scaffolds nonetheless concentrate on
`supervised_likelihood` at the objective layer — the divergence is at the
update-mechanism layer.

See [`scaffold_lockin/output/report.md`](scaffold_lockin/output/report.md).

Both audits are **observational**. Agent model, interface, prompt and scaffold
are not independently randomised, and method labels do not cover every training
command. The reports state the admissible claim explicitly.

## Running them

Both read the pipeline's `intermediate/` tree, which is not redistributed
(~2 GB). Regenerate it first with `analysis/pilot_study/run_pilot.py`, then:

```bash
python analysis/strategy_lockin/broad_criterion/build_broad_transitions.py
python analysis/strategy_lockin/scaffold_lockin/build_comparison.py
```

The `output/` directories in this repository hold the results of the runs the
paper reports.

Tests: `python -m unittest discover -s analysis/strategy_lockin/scaffold_lockin/tests -v`
