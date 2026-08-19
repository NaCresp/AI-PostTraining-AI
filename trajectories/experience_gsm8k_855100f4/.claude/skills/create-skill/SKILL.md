---
name: create-skill
description: Create or update Claude Code skills under workspace .claude/skills. Use when a reproducible training workflow is validated, after a kept experiment iteration, or when the user asks to persist experience as a skill.
---

# Create skill (persist reproducible experience)

During optimization, **encapsulate reusable, reproducible experience as skills** — not one-off run notes. Skills live under your working directory and are invoked via the **Skill** tool.

## When to create a skill

Create or update a skill when **all** apply:

1. The workflow succeeded or failed in a **repeatable** way (you could follow the same steps again).
2. It generalizes beyond this run (not a single checkpoint path, token ID, or lucky hyperparameter from one try).
3. It saves future context (checklist, script, or decision tree).

**Do not** skill-ify: one-off EOS patches for a specific bug, "step 47 was best", framework version accidents, or evaluator-specific hacks unless they are stable protocol for your training stack.

## Where to write

```
.claude/skills/<skill-name>/
├── SKILL.md          # required
└── scripts/          # optional — repeatable commands
```

- `name` in YAML frontmatter **must equal** the directory name (`kebab-case`, ≤64 chars).
- `description`: third person; state **what** and **when** (drives auto-invocation).
- Use `${CLAUDE_SKILL_DIR}` for bundled script paths.

## SKILL.md template

```markdown
---
name: my-workflow
description: One-line what + when to use (third person).
---

# Title

## When to use
...

## Procedure
1. ...

## Verification
- How you know it worked

## Pitfalls
- What not to do
```

Keep the body **under ~80 lines**; put long reference in `reference.md` only if needed.

## After creating

1. Log in `experiment.jsonl` that you added/updated `<skill-name>` and why.
2. On the next similar task, invoke **Skill** (`/my-workflow` or tool) instead of re-deriving from scratch.

## Related seed skills

Browse existing skills under `.claude/skills/` before writing a duplicate. Extend an existing skill if the workflow fits; create a new directory only for a distinct scenario.
