# Annotation protocol

Every claim in the paper about *what an agent decided* rests on labels applied to extracted
training experiments. This document defines those labels, states their coverage, and describes
the two annotation passes released in `analysis/annotations/`.

The governing rule: **labels are assigned from executed evidence, never from intent, and missing
labels are never imputed.** An agent that writes "next I will try DPO" in its journal and then
never launches a DPO job contributes no preference-optimization label. An experiment whose data
provenance cannot be determined from the artifacts is left unlabeled and drops out of any
statistic that needs it, rather than being assigned a default. This is why coverage numbers below
are well under 100%, and why they are reported rather than hidden.

---

## The strategy state

The unit of analysis is the **training experiment**: one launched training job with an identifiable
configuration. 5,111 such experiments were extracted from the 1,338 trajectories.

Each experiment carries a strategy state

```
s = (k, d, g)
```

- **k — training strategy.** What optimization the job performs.
- **d — data source.** Where the training data came from.
- **g — stage structure.** How many training stages the trajectory chains, and in what order.

A **strategy change** is a transition between temporally adjacent experiments in the same
trajectory where `s` differs in at least one component. There are 3,557 such adjacent pairs in the
corpus, and 74 of them (2.1%) are strategy changes.

### k — training strategy labels

| Label | Executed evidence required |
|---|---|
| **Supervised Fine-Tuning** | A supervised likelihood objective over full model parameters; all weights updated |
| **Parameter-Efficient Fine-Tuning** | A supervised likelihood objective with an adapter (LoRA and variants); base weights frozen |
| **Reinforcement Learning** | A policy-gradient or reward-maximizing objective (GRPO, PPO, RLOO, and variants) with a reward signal actually computed |
| **Preference optimization** | A pairwise-preference objective (DPO and variants) over preference-formatted data |
| **Distillation** | An objective fitting a teacher's outputs or distribution, with a teacher present in the run |

The distinction between Supervised Fine-Tuning and Parameter-Efficient Fine-Tuning is drawn on the
*mechanism* (full weights vs. adapter), not on the loss, because that is the distinction agents
actually trade off against GPU memory on a single 80GB card.

At the experiment level, the corresponding objective family is recorded with the enum used
throughout the released CSVs:

```
supervised_likelihood | reward_optimization | preference_optimization
                      | on_policy_distillation | objective_unknown
```

`objective_unknown` is a real label with real rows behind it, not a null. It marks experiments
whose objective could not be established from the artifacts, and it is excluded from
objective-conditioned statistics rather than folded into a majority class.

### d — data source labels

| Label | Meaning |
|---|---|
| **curated** | An external dataset, downloaded or supplied, not produced by the model under training |
| **self-generated** | Data sampled from the model under training (rejection sampling, self-distillation, on-policy rollouts) |
| **mixed** | Both, combined in a single training set |

Data source is identifiable for **1,801 of 4,344** strategy-labeled experiments (41.5%). The
remaining experiments train on files whose provenance the trajectory does not establish. They are
left unlabeled. Data-source change counts in the paper are therefore computed over adjacent pairs
where *both* sides carry a data-source label.

### g — stage structure

The ordered sequence of training stages the trajectory has executed so far — for example a single
SFT stage, or SFT followed by GRPO on the resulting checkpoint. A stage-structure change means the
agent altered the *shape* of its pipeline, not just its contents: adding a stage, removing one, or
reordering them.

This is the rarest change by a wide margin. Across 3,557 adjacent pairs, exactly **1** stage-structure
change occurs. Agents pick a pipeline shape in their first hour and keep it for the remaining nine.

### Initial strategy

900 of the 1,338 trajectories launch at least one training job. Of those, the **initial strategy** —
the strategy state of the first launched experiment — is recognized for **783**. The remaining 117
launch a job whose configuration cannot be resolved to a complete state. Initial-strategy
convergence (`C_G`, see [metrics.md](metrics.md)) is computed over the 783.

---

## Trajectory phase labels

Independently of the strategy state, each segment of a trajectory is labeled with what the agent
is doing, using eight phases:

| Phase | The agent is… |
|---|---|
| **Exploration** | Reading the benchmark, the model card, the wiki; forming a plan |
| **Data/Setup** | Building or converting the training set; environment setup |
| **SFT** | Running supervised fine-tuning |
| **RL/GRPO** | Running reinforcement learning |
| **Evaluation** | Scoring a checkpoint or interpreting a score |
| **Debugging** | Diagnosing a crash, a hang, or a silent failure |
| **Journal/Skill** | Writing journal entries or authoring skills |
| **Checkpoint/Waiting** | Blocked on a running job or a pending evaluation |

Phase labels are what make "the agent spent its budget on X" answerable, and they are the basis
for the normalized-progress statistics below.

---

## The two released annotation passes

`analysis/annotations/` contains two independent passes over the same 1,338 trajectories.

### `objective_level/`

Training-objective labels per experiment, plus derived summary tables:

- `annotations/` — the raw per-experiment objective labels
- `tables/` — 11 summary tables (per-scaffold, per-model, per-benchmark rollups)
- `evaluation_events.csv` — the 2,034 evaluation points, with scores and timestamps
- `objective_transitions.csv` — objective changes between adjacent experiments

### `strategy_level/`

Full strategy-state labels:

- `episode_annotations.jsonl` — per-experiment `(k, d, g)` labels
- `tables/trajectory_analysis.csv` — per-trajectory rollup

### They are passes, not versions

It is tempting to read the strategy-level pass as a superset of the objective-level pass. It is
not. Compared directly:

- both cover the same **1,338 trajectories**
- they share **72 columns**; the strategy-level pass adds **9** more
- **65 rows disagree** on the shared columns

Both are released precisely so the disagreement is inspectable. The paper's headline strategy
numbers use the strategy-level pass; objective-only statistics use the objective-level pass. Where
a number could be computed from either, the paper says which.

---

## Audit of the objective-changing trajectories

Only 16 trajectories in the corpus change their training objective at all. Because the number is
small enough to read by hand, all 16 were audited individually. What that audit found:

- **14 of 16 begin with SFT.** Supervised fine-tuning is the default first move almost everywhere.
- **11 of 16 switch at least twice.** Objective changes are not one-time corrections; the agents
  that change at all tend to oscillate.
- **15 of the 35 objective transitions return to SFT.** The most common "change" is a retreat to
  the starting point — which is why a raw change count overstates genuine strategic exploration.
- **The first switch happens at median normalized progress 0.40**, and across all switches the
  median is **0.67**. Agents that revise do so late, after most of the budget is already spent on
  the initial strategy.

Normalized progress is position within a trajectory's own event sequence, on [0, 1], so it is
comparable across trajectories of different lengths.

---

## Reconciling 73 vs. 74

`analysis/strategy_lockin/broad_criterion/report.md` reports **73** strategy changes over 43
trajectories; the paper reports **74** over 44. The difference is one experiment.

The broad-criterion pass flags stage-structure candidates for author review rather than accepting
them automatically, because stage-structure changes are the label most easily confounded by a
re-launch after a crash. Two candidates were flagged. On review, one was accepted as a genuine
stage-structure change (a Codex trajectory, taking that scaffold's count from 14 to 15) and one
was rejected as a restart. The paper's 74 = 35 objective + 38 data-source + 1 stage reflects that
adjudication; the report reflects the pre-adjudication state. Both are released as-is rather than
retroactively edited.
