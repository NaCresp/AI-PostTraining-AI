# The experience-driven framework

The paper compares two conditions: a **bare agent**, which is given a base model, a benchmark, a
GPU, and ten hours; and a **scaffolded agent**, which is given the same thing plus an
experience-driven framework. This document describes that framework, the run loop it sits in, and
where each piece lives in this repository.

Adding the framework is worth **+12.6 points on GSM8K** and **+40.8 points on HumanEval**. It
makes the agent a better *executor* — fewer wasted hours, fewer repeated crashes, tighter
experiment hygiene. It does not make the agent a better *strategist*: scaffolded trajectories lock
in on their first strategy at essentially the same rate as bare ones. That gap is the paper's
central finding, and this framework is the strongest scaffold we could build that still fails to
close it.

---

## Three components

### 1. Experiment journal — `framework/journal/`

An append-only log the agent writes to as it works. It is the agent's only durable memory across
the run: context is compacted, but the journal is not.

Four entry types, each written as a JSON object:

| Type | Written when | Required fields |
|---|---|---|
| `plan` | Before launching a training experiment | `hypothesis`, `success_criteria`, `abort_criteria` |
| `lesson` | After something fails or surprises the agent | what happened, what it means, what to do differently |
| `reflection` | At a checkpoint in the run | current state, what the evidence supports, what to do next |
| `skill_created` | After distilling a lesson into a reusable skill | which skill, what it covers |

The `plan` schema is the load-bearing one. Requiring `success_criteria` **and** `abort_criteria`
*before* a job launches is what makes strategy revision measurable at all: it commits the agent
in advance to a condition under which it would abandon the current approach. The paper's finding
is that agents write abort criteria faithfully and then, when those criteria are met, revise
hyperparameters rather than strategy.

### 2. Skill library — `framework/skills/`

Two layers:

**`framework/skills/skills/`** — 15 skills, each a directory containing a `SKILL.md` with YAML
frontmatter (`name`, `description`) and a procedural body. Skills are loaded by name on demand,
so the library can grow past what fits in context.

| Skill | Covers |
|---|---|
| `create-skill` | The meta-skill: how to write a new skill from a lesson |
| `rl-experiment-discipline` | One variable at a time; log before launch; never trust an unevaluated checkpoint |
| `sft-cold-start-gate` | When SFT is a prerequisite for RL and when it is a detour |
| `escalate-rl-algorithm` | Moving up the algorithm ladder when a simpler method plateaus |
| `grpo-defaults`, `ppo-defaults` | Known-good starting configurations |
| `grpo-lora-config` | GRPO combined with low-rank adaptation |
| `tune-rl-memory-throughput` | Batch size, sequence length, and offload trade-offs on one 80GB GPU |
| `monitor-rl-training` | What to watch during a run and when to intervene |
| `diagnose-silent-rl-failures` | Runs that complete cleanly and learn nothing |
| `triage-training-collapse` | Loss spikes, reward collapse, degenerate outputs |
| `fix-train-infer-mismatch` | Divergence between training-time and inference-time behavior |
| `rl-chat-tokenizer-pitfalls` | Chat templates, special tokens, and masking errors |
| `posttraining-known-pitfalls` | Cross-cutting failure modes seen across the corpus |
| `data-parquet-schema` | The on-disk schema the training data must conform to |

Skills are **agent-authored**. They were written by agents during real runs via `create-skill`,
then carried forward as seeds for later runs; we did not hand-write the library. Three skills
were named after the specific library the authoring agent happened to be driving, and are
released under generic names matching the paper's terminology:

| Original name | Released as |
|---|---|
| `verl-known-pitfalls` | `posttraining-known-pitfalls` |
| `verl-parquet-schema` | `data-parquet-schema` |
| `grpo-lora-verl` | `grpo-lora-config` |

Two skills reference names that no released skill provides (`async-rl-escalation`,
`design-verifiable-reward`). These dangling references are left in place: they are authentic
artifacts of how agents write skills, and repairing them would misrepresent the corpus.

**`framework/skills/wiki/`** — a reference wiki the agent reads but does not train on: 11
`entities/` pages (libraries, models, benchmarks), 21 `concepts/` pages (algorithms, objectives,
failure modes), 20 `experience/` pages (accumulated findings), and 8 `sources/` pages summarizing
upstream post-training libraries. `AGENT.md` tells the agent how to navigate it; `index.md` and
`log.md` are the entry point and change log.

The upstream repositories the `sources/` pages summarize are pinned as git submodules under
`framework/skills/wiki/sources/upstream/` — verl, TRL, OpenRLHF, NeMo-RL, slime, and Miles — at
the commits the agents actually read. The 908 raw scraped documents behind those summaries are not
released.

### 3. Evaluator agent — `framework/evaluator/`

A second agent, running in its own process against the same workspace, that owns benchmark
scoring. Separating it from the training agent does two things: it keeps GPU-bound evaluation off
the training agent's critical path, and it prevents the training agent from grading its own work.

Its loop is three steps per checkpoint, defined in each experiment's `eval_program.md`:

1. **Predict** — before scoring, write down the expected result and why. This turns every
   evaluation into a falsifiable test rather than a lookup.
2. **Evaluate** — run the benchmark harness (`evaluation/<benchmark>/evaluate.py`) and record the
   score.
3. **Analyze** — compare the prediction to the outcome, and write the delta back to the journal
   as a `lesson` if they disagree.

Each step emits a JSON object against a fixed schema, which is what makes the 2,034 evaluation
points in the corpus machine-readable.

---

## The run loop

```
  training agent                          evaluator agent
  ──────────────                          ───────────────
  read journal + skills + wiki
  write `plan` entry
  launch training job  ──────►  checkpoint
                                   │
                                   ├─► append to .eval_queue
                                   │
  poll .eval_processed  ◄──────────┼──  predict
        │                          │    evaluate  (evaluation/<bench>/evaluate.py)
        │                          │    analyze
        │                          └──  append to .eval_processed
        ▼
  write `lesson` / `reflection`
  maybe write a skill (`create-skill`)
  next experiment  ────────────────►  (repeat until .timer expires)
```

The two agents communicate through the filesystem, not a message bus: `.eval_queue` is a
checkpoint request queue, `.eval_processed` carries results back, and `.timer` holds the run
deadline. This is deliberately crude — it survives either agent crashing and restarting, which
happens over a ten-hour unattended run.

The **polling protocol** matters more than it looks. An agent that blocks on evaluation burns its
budget waiting; an agent that never checks trains blind. `program.md` specifies that the training
agent must poll between experiments and must not launch a follow-up experiment that depends on an
unreturned evaluation.

### Harness

Trajectories are produced by driving a coding agent CLI non-interactively. For the Claude Code
condition:

- Claude Code **2.1.150**, agent model `claude-opus-4-6`
- `--output-format stream-json`, streamed to `trajectory.jsonl` — this file is the raw record the
  entire analysis pipeline consumes
- `--allowedTools` scoped to exactly the tools the run needs, so the trajectory is a closed system
- one base model, one benchmark, 10 hours wall clock, 1× H100 80GB, no human in the loop

The other four scaffolds (OpenCode, Codex, GLM-X, Qwen3-Max) are driven equivalently; their
trajectory formats differ, which is why `analysis/pipeline/` carries a parser per format
(`parse_claude.py`, `parse_codex.py`, `parse_trace.py`).

---

## Where things are

| Component | Prompts / config | Released trajectory |
|---|---|---|
| Training agent | `experiments/<benchmark>/program.md`, `config.yaml`, `start.sh` | `trajectories/<benchmark>/run_*/` |
| Evaluator agent | `experiments/<benchmark>/eval_program.md`, `start_eval.sh`, `request_eval.sh` | `trajectories/<benchmark>/run_*/eval_agent_logs/` |
| Journal | schemas in `program.md` | `trajectories/<benchmark>/run_*/` journal files |
| Skill library | `framework/skills/skills/` | `trajectories/<benchmark>/run_*/.claude/skills/` |
| Wiki | `framework/skills/wiki/` | — |
| Scoring | `evaluation/<benchmark>/evaluate.py` | `trajectories/<benchmark>/run_*/outputs/` |

The `human_guidance` condition in `experiments/human_guidance/` is the one exception to the
no-human rule: a human reviews the agent's plan at fixed points and returns a `keep` or `revert`
decision. It is the paper's probe for whether lock-in is a capability limit or an inertia limit.
See `trajectories/human_aime2025_14ce9dfd/` for a complete example, including the two human decisions
recorded during that run.
