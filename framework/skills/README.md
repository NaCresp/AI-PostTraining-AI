# Skill Library

Two layers, both agent-authored:

| Layer | Directory | What it is |
|---|---|---|
| Skills | [`skills/`](skills/) | 15 procedural entries, each a `SKILL.md` with a `name`/`description` front matter block. Loaded by name when the agent hits a matching problem. |
| Wiki | [`wiki/`](wiki/) | The knowledge base the skills were distilled from: entities, concepts, experience notes, and per-source summaries of six open-source post-training frameworks. |

## Skills

A skill is written only after the agent has *personally validated* the
procedure in a session — it contains tested commands and configs, not
speculation. The prompt instructs the agent to create or update a skill
whenever it writes a `lesson` that involves a configuration change, a fix, or a
non-obvious finding, and to **empty** a skill that gave bad advice.

Five skills (`create-skill`, `diagnose-silent-rl-failures`,
`monitor-rl-training`, `posttraining-known-pitfalls`, `data-parquet-schema`)
were seeded before the runs; the other ten were written by the agent during the
AIME campaign.

### Renaming

Three skills were named after the specific trainer used in our runs. They are
released under the generic names used in the paper. The rename is applied
consistently to directory names, front matter, cross-references, and to the
copies archived under [`../../trajectories/`](../../trajectories/):

| Original name | Released as |
|---|---|
| `verl-known-pitfalls` | `posttraining-known-pitfalls` |
| `verl-parquet-schema` | `data-parquet-schema` |
| `grpo-lora-verl` | `grpo-lora-config` |

Skill *contents* are otherwise unmodified. Two cross-references
(`async-rl-escalation`, `design-verifiable-reward`) point at skills that never
existed — the agent invented the names while writing. They are left in place as
authentic artifacts.

## Wiki

`wiki/AGENT.md` is the schema and maintenance protocol the agent follows.
`wiki/index.md` and `wiki/log.md` are its table of contents and ingestion log.

| Directory | Pages | Contents |
|---|---|---|
| `wiki/entities/` | 11 | frameworks, models, datasets |
| `wiki/concepts/` | 21 | algorithms and failure modes |
| `wiki/experience/` | 20 | cross-cutting lessons |
| `wiki/sources/` | 8 | one summary per ingested upstream source |

The raw ingested corpus (908 documents) is not redistributed. `wiki/sources/`
instead pins the upstream repositories as git submodules — see
[`wiki/sources/README.md`](wiki/sources/README.md).
