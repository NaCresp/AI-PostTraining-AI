# Wiki Log

## [2026-05-19] init | Wiki skeleton

Created wiki structure: AGENT.md, index.md, log.md, entities/, concepts/, sources/, raw/.
No content yet — awaiting first source in raw/.

## [2026-05-19] ingest | verl Documentation

Ingested `raw/verl-docs/` (109 files). Focused on RL engineering knowledge for stable training; ignored hardware architecture and environment setup.

Pages created:
- sources/verl-docs.md (source summary)
- entities/grpo.md, entities/dapo.md, entities/verl.md
- concepts/entropy-collapse.md, concepts/kl-divergence-control.md, concepts/loss-aggregation.md
- concepts/clip-ratio-and-trust-region.md, concepts/rollout-correction.md, concepts/training-inference-mismatch.md
- concepts/batch-size-tuning.md, concepts/async-training.md

Total: 1 source summary + 3 entity pages + 8 concept pages = 12 pages.

## [2026-05-19] ingest | verl-recipe Collection

Ingested `raw/verl-recipe/` (463 files). Extracted new practical knowledge from FAPO, ReTool, GKD distillation, SPO, GVPO recipes. Skipped hardware-specific recipes (Ascend, AMD) and environment setup.

Pages created:
- sources/verl-recipe.md (source summary)
- entities/retool.md, entities/fapo.md
- concepts/reward-design.md, concepts/multi-turn-tool-rl.md

Pages updated:
- entities/verl.md — added FAPO, GVPO, SPO to algorithm table; added cross-references

Total: 1 source summary + 2 entity pages + 2 concept pages = 5 new pages, 1 updated.

## [2026-05-19] restructure | Add experience layer

Added `experience/` directory — actionable guides synthesized from entities and concepts. Updated AGENT.md with page template and conventions.

Pages created:
- experience/stable-grpo-training.md — config checklist + runtime monitoring
- experience/diagnosing-training-collapse.md — step-by-step collapse triage
- experience/performance-tuning.md — throughput tuning (batch/async/packing)
- experience/reward-function-design.md — reward function design patterns

Total: 4 experience pages. AGENT.md and index.md updated.

## [2026-05-29] restructure | experience layer deduplication

Re-analyzed `raw/` (verl-docs, verl-recipe, slime, miles, openrlhf, trl, nemo-rl) and refactored `experience/`:

- Created `experience/rl-training-principles.md` — framework-agnostic consolidated guide
- Created `experience/training-collapse-triage.md` — replaced duplicated collapse content
- Refactored performance-tuning, stable-grpo, async, sft, advanced, reward, infra, vlm, distillation, multi-turn, grpo-lora to link to principles and remove duplicate pitfalls
- `diagnosing-training-collapse.md` → redirect stub
- `index.md` — split Experience into Universal / Framework-Specific sections
