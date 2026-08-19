# Data Pipeline and Tokenization

Data preparation and tokenization form the foundation of post-training for both SFT and RL. Incorrect dataset format, tokenizer API changes, or tensor layout mismatches can silently corrupt rewards or crash training.

## Mechanism

### RLHFDataset Format

`RLHFDataset` expects message lists, not OpenAI API blob format (verl Issues #6096). Per-row `tools`/`tool_choice` fields are not supported; tools come from the global `tool_config_path` (verl Issues #6096).

### Transformers 5 Tokenizer API Change

In transformers 5, `apply_chat_template` returns `BatchEncoding` instead of `list[int]`. Code must unwrap via `normalize_token_ids()` — failure to unwrap silently zeros rewards (verl Issues #6080).

### TransferQueue Tensor Layout

TransferQueue with uniform-shape prompts and variable-length responses produces mixed nested/stacked tensors (verl Issues #6407, #6261). Use TransferQueue ≥0.1.7 with jagged tensors as default (verl Issues #6407, #6261).

### Multimodal SFT Data Filtering Performance

For multimodal SFT data filtering, `pop()` vs `get()` matters: using `get()` for image/video fields makes `Dataset.filter` approximately 1000× slower because HuggingFace serializes full image binaries on access (verl Issues #6145). Use `pop()` to avoid deserialization overhead (verl Issues #6145).

### VLM Rollout Retokenization Drift

Do not pass raw prompts to rollout for VLMs. Retokenization drift between trainer tokenization and inference re-tokenization corrupts RL rewards (verl Issues #6168). Use TITO (token-in-token-out) via vLLM/sglang instead (verl Issues #6168).

### Rollout Logging Bug

A rollout logging bug writes `request_id` to the wrong dictionary, breaking traceability of generated samples (verl Issues #6250).

## Diagnostic Relevance

- **Zero rewards despite valid rollouts**: Check transformers 5 `apply_chat_template` return type; unwrap with `normalize_token_ids()` (verl Issues #6080).
- **Dataset load/format errors in RL**: Verify message-list format for `RLHFDataset`; tools must come from `tool_config_path`, not per-row fields (verl Issues #6096).
- **Tensor layout crashes in TransferQueue**: Upgrade to ≥0.1.7 and enable jagged tensors default (verl Issues #6407, #6261).
- **Multimodal filter taking hours**: Switch from `get()` to `pop()` for image/video fields in filter functions (verl Issues #6145).
- **VLM RL reward drift**: Compare trainer vs rollout token IDs; use TITO path to eliminate retokenization (verl Issues #6168).
- **Missing rollout traceability**: Check `request_id` mapping in rollout logging (verl Issues #6250).

## Related Pages

- [VLM Post-Training](vlm-post-training.md) — multimodal data handling and TITO requirements
- [SFT Cold Start](sft-cold-start.md) — SFT dataset preparation precedes RL data pipeline
- [Reward Design](reward-design.md) — reward computation depends on correct tokenization and dataset format

## Sources

- [verl Issues](../sources/verl-issues.md) — RLHFDataset format (#6096), transformers 5 tokenizer API (#6080), TransferQueue tensor layout (#6407, #6261), multimodal filter performance (#6145), VLM retokenization drift (#6168), rollout logging bug (#6250)
