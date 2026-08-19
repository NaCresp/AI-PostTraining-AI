---
name: rl-chat-tokenizer-pitfalls
description: Avoid silent zero rewards from chat templates and dataset format (message lists, transformers 5 BatchEncoding, data_source routing). Use when building RL parquet or custom data loaders.
---

# RL chat and tokenizer pitfalls

Applies to any stack using `apply_chat_template` + rule-based rewards; verl users also need `data-parquet-schema`.

## Dataset format

- Store **chat message lists** `[{"role": "user", "content": "..."}]`, not opaque API blobs, when the trainer expects RLHF-style rows.
- Every `data_source` value must map to a reward handler — missing routes → **silent zero reward**, no crash.

## Transformers 5 tokenizer API

`apply_chat_template(..., tokenize=True)` may return `BatchEncoding` instead of `list[int]`.

**Symptom:** training runs, rewards stuck at zero.

**Fix:** unwrap to token id lists (`normalize_token_ids()` or equivalent) before training; or pin `transformers<5` after regression test.

## Reward extraction alignment

- Training reward extractor must match eval protocol (same final-answer location, normalization).
- For `\boxed{}` math: scan **last ~300 characters** only; ignore intermediate boxes in CoT.

## Verification

- Single batch: print one prompt’s token ids length and type (must be `list[int]`).
- One known-correct completion scores 1.0 in reward fn before full RL.

## Pitfalls

- Valid generations but zero reward → check tokenizer + routing before tuning LR.
- Mixing train reward on full text with eval on truncated tail without intending to.

## Related skills

- `design-verifiable-reward`, `data-parquet-schema`, `diagnose-silent-rl-failures`
