# What is Missing from AI Post-Training AI: An Empirical Analysis

<div align="center">

[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://arxiv.org/abs/2608.19072)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

</div>

Code and data for **[What is Missing from AI Post-Training AI: An Empirical Analysis](https://arxiv.org/abs/2608.19072)**.

## Overview

We give coding agents a base language model, a benchmark, one GPU, and ten hours, and ask them to
post-train the model. Repeating this 1,338 times, we find that agents are strong *executors* and
weak *strategists*: they configure trainers, fit models onto a single card, and recover from
crashes reliably, but the training objective, data source, and stage structure they commit to in
their **first** experiment is almost never revised afterwards — even when their own abort criteria
say it should be. We call this **strategy lock-in**. Adding an experience-driven framework
(experiment journal, skill library, evaluator agent) markedly improves execution and leaves
lock-in intact.

This repository contains the harness that ran the experiments, the annotations behind the paper's
numbers, and ten archived agent runs covering all three conditions.

## Repository Structure

```
├── framework/       the experience-driven scaffold
│   ├── journal/       experiment journal: plan / lesson / reflection schemas
│   ├── evaluator/     evaluator agent: predict → evaluate → analyze
│   └── skills/        agent-authored skills, and the reference wiki they draw on
├── experiments/     agent prompts, configs, and launchers, per benchmark
├── evaluation/      AIME 2025 / GSM8K / HumanEval scoring harnesses
├── analysis/        trajectory parsing pipeline, lock-in analyses, and the annotations
├── trajectories/    ten agent runs: framework, baseline, and human-guided
└── docs/            framework design, annotation protocol, metric definitions
```

Each directory has its own `README.md`.

## Installation

```bash
git clone --recurse-submodules https://github.com/JoylimJY/AI-PostTraining-AI.git
cd AI-PostTraining-AI
pip install -r requirements.txt
```

The analysis code runs on the standard library alone. The pinned versions matter for the training
and evaluation harnesses, where `torch` and `vllm` may need to be matched to your CUDA stack.

The submodules pin the upstream post-training libraries the agents read from, at the commits they
actually saw. They are only needed to regenerate the wiki source pages — drop
`--recurse-submodules` to skip them.

## Usage

Reproduce the lock-in numbers from the released annotations:

```bash
python analysis/strategy_lockin/scaffold_lockin/build_comparison.py
```

The broader data-source and stage audit
(`analysis/strategy_lockin/broad_criterion/build_broad_transitions.py`) additionally reads the
pipeline's `intermediate/` tree, which is too large to redistribute; regenerate it first with
`analysis/pilot_study/run_pilot.py`.

Read a trajectory the way the analysis does:

```python
from pathlib import Path
from analysis.pipeline.parse_claude import parse_claude_records

records, stats = parse_claude_records(
    Path("trajectories/experience_aime2025_c247e78e/trajectory.jsonl").read_text()
)
```

For the decision-level view, read `experiment.jsonl` in the same directory instead — it holds the
agent's own account of what it planned, what it expected, and what it concluded.

Re-running the full extraction (`python -m analysis.pipeline.run_pipeline --config
analysis/pipeline/config.json`) requires the raw logs, which are terabyte-scale and not released.
The ten included runs are complete enough to exercise every parser on real input; they cover
both stream formats the pipeline handles.

## Data

`analysis/annotations/` holds two independent annotation passes over the same trajectories:
`objective_level/` labels the training objective of each experiment, and `strategy_level/` labels
the full strategy state — training strategy, data source, and stage structure.

Labels are assigned from executed evidence only, never from stated intent, and missing labels are
never imputed. Every label carries a reference back to the exact line of the source trajectory.

## Documentation

- [`docs/framework.md`](docs/framework.md) — the experience-driven scaffold and the run loop
- [`docs/annotation-protocol.md`](docs/annotation-protocol.md) — label definitions and coverage
- [`docs/metrics.md`](docs/metrics.md) — metric definitions and cohorts

## Citation

```bibtex
@misc{lim2026missing,
  title         = {What is Missing from {AI} Post-Training {AI}: An Empirical Analysis},
  year          = {2026},
  eprint        = {2608.19072},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.19072}
}
```

## License

[MIT](LICENSE). Upstream libraries under `framework/skills/wiki/sources/upstream/` and the
benchmark datasets keep their own licenses.
