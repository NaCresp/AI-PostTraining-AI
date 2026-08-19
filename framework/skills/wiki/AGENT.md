# Wiki Agent Schema

This file defines the structure, conventions, and operation protocols for the RL Engineering Wiki.
The wiki is entirely maintained by the LLM. The user sources materials into `raw/`, asks questions,
and guides emphasis. The LLM does all summarizing, cross-referencing, filing, and bookkeeping.

## Domain

LLM post-training engineering — covering the full pipeline from SFT (Supervised Fine-Tuning)
through RL (GRPO, PPO, DPO, DAPO, etc.) to on-policy distillation (OPD). Includes training
infrastructure (verl, FSDP, Megatron, vLLM, SGLang), data preparation, reward design,
multi-turn/tool-calling agent training, VLM (Vision-Language Model) post-training, failure
diagnosis, hyperparameter tuning, and deployment considerations.

---

## Directory Structure

```
wiki/
  AGENT.md              # This file — schema and rules
  index.md              # Content catalog (all pages, organized by type)
  log.md                # Append-only chronological activity record
  entities/             # One page per concrete thing (model, algorithm, tool, dataset, etc.)
  concepts/             # One page per abstract idea (failure mode, technique, pattern, etc.)
  experience/           # Actionable guides synthesized from entities + concepts
  sources/              # One summary page per ingested source
  raw/                  # Unstructured source materials — read-only input
```

### Naming Conventions

- Filenames: `kebab-case.md` (e.g. `entropy-collapse.md`, `grpo.md`)
- Cross-references: standard markdown links with relative paths (e.g. `[GRPO](../entities/grpo.md)`)
- One file per entity or concept — never combine multiple topics into one page

---

## Page Templates

### Entity Page (`entities/`)

An entity is a concrete, nameable thing: a model, algorithm, library, dataset, benchmark, hyperparameter.

```markdown
# {Entity Name}

{1-3 sentence definition.}

## Details

{Core technical content. What it does, how it works, key parameters, known behaviors.
Updated incrementally as new sources are ingested.}

## Relevance

{Why this entity matters in the context of RL engineering for LLMs.}

## Related Pages

- [Page Name](relative/path.md) — {one-line relationship description}

## Sources

- [Source Title](../sources/source-slug.md) — {what this source contributed to this page}
```

### Concept Page (`concepts/`)

A concept is an abstract idea, pattern, failure mode, or technique.

```markdown
# {Concept Name}

{1-3 sentence definition.}

## Mechanism

{How it works, why it happens, under what conditions. Causal chains if applicable.}

## Diagnostic Relevance

{How to detect this in practice — what metrics, symptoms, or patterns to look for.}

## Related Pages

- [Page Name](relative/path.md) — {one-line relationship description}

## Sources

- [Source Title](../sources/source-slug.md) — {what this source contributed to this page}
```

### Experience Page (`experience/`)

An experience page is a concise, actionable guide synthesized from multiple entity and concept pages.
It answers "what should I actually do?" rather than "how does this work?".
Each page covers a specific operational scenario (e.g., starting a training run, debugging a failure).

```markdown
# {Scenario Title}

{1-2 sentence description of when to use this guide.}

## Checklist / Steps

{Numbered or bulleted actionable items. Each item should be concrete and directly executable.
Include config snippets, metric thresholds, and decision criteria where applicable.}

## Key Pitfalls

{Common mistakes and how to avoid them. Drawn from entity/concept pages.}

## Derived From

- [Page Name](relative/path.md) — {what this experience page draws from}
```

Experience pages do not introduce new knowledge — they reorganize existing wiki knowledge into
operational form. When entities or concepts are updated, the corresponding experience pages should
be reviewed for consistency.

### Source Summary Page (`sources/`)

One page per ingested source, summarizing what was extracted and where it went.

```markdown
# {Source Title}

- **Type**: {paper | tutorial | documentation | training log | config | codebase | conversation | other}
- **Raw location**: `raw/{path}`
- **Ingested**: {YYYY-MM-DD}

## Key Takeaways

{Numbered list of the most important things learned from this source.}

## Pages Created or Updated

- [Page Name](relative/path.md) — {what was added/changed}

## Notes

{Any caveats, open questions, contradictions with existing wiki content.}
```

---

## Operations

### Ingest

Triggered when the user adds a source to `raw/` and asks the LLM to process it.

1. Read the source material in `raw/`.
2. Discuss key takeaways with the user (unless batch mode is requested).
3. Write a source summary page in `sources/`.
4. For each entity mentioned: create its page in `entities/` if new, or update it if existing.
5. For each concept mentioned: create its page in `concepts/` if new, or update it if existing.
6. Add cross-references between pages — both directions.
7. Update `index.md` — add new pages, update summaries and source counts for changed pages.
8. Append an entry to `log.md`.

When updating an existing page, preserve prior content. Add new information, note contradictions
explicitly (do not silently overwrite), and update the Sources section.

### Query

Triggered when the user asks a question against the wiki.

1. Read `index.md` to identify relevant pages.
2. Read those pages and synthesize an answer with citations.
3. If the answer has lasting value (analysis, comparison, synthesis), offer to file it as a new
   page in the appropriate directory.
4. Append a query entry to `log.md`.

### Lint

Periodic health check of the wiki. Run when the user requests it or after significant growth.

1. **Contradictions**: find pages that make conflicting claims.
2. **Stale content**: claims superseded by newer sources.
3. **Orphan pages**: pages with no inbound links from other pages.
4. **Missing pages**: concepts or entities mentioned in text but lacking their own page.
5. **Missing cross-references**: related pages that don't link to each other.
6. **Data gaps**: questions the wiki should be able to answer but can't — suggest sources to find.
7. Append lint results to `log.md`.

---

## Quality Standards

- Every page must have at least one entry in Sources (no unsourced claims).
- Cross-references should be bidirectional — if A links to B, B should link to A.
- When new information contradicts existing content, flag it explicitly with
  `> **Contradiction**: {description}` rather than silently replacing.
- Prefer splitting a page that covers too many topics over keeping a monolithic page.
- Keep summaries concise. Detail goes in the Mechanism / Details sections, not in the definition.

---

## Index Format

`index.md` is organized by page type. Each entry has a link, a one-line summary, and a source count.

```markdown
## Entities
- [Name](entities/slug.md) — {one-line summary} ({N} sources)

## Concepts
- [Name](concepts/slug.md) — {one-line summary} ({N} sources)

## Experience
- [Title](experience/slug.md) — {one-line summary of the scenario}

## Sources
- [Title](sources/slug.md) — {source type}, ingested {YYYY-MM-DD}
```

## Log Format

`log.md` is append-only. Each entry uses the format:

```markdown
## [YYYY-MM-DD] {verb} | {subject}

{1-3 sentence summary of what happened. For ingests, list pages touched.}
```

Verbs: `ingest`, `query`, `lint`, `restructure`, `merge`, `split`.

This format is parseable with grep: `grep "^## \[" log.md | tail -5` gives the last 5 entries.
