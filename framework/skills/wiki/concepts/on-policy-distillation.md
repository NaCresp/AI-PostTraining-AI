# On-Policy Distillation (OPD)

On-Policy Distillation uses a teacher model to provide soft targets during on-policy training of a student model. In verl, OPD is configured with `algorithm.adv_estimator: distillation` and requires no critic.

## Mechanism

### verl Configuration

Set `algorithm.adv_estimator: distillation` to enable OPD. No critic model is needed — the teacher provides the distillation signal directly (verl-docs).

### GKD Recipe Engineering

The GKD (Generalized Knowledge Distillation) recipe implements one-step-off and two-step-off schedulers for overlapping rollout, teacher inference, and student training (verl-recipe). Batched NCCL weight sync provides approximately 12× speedup over naive streaming weight transfer (verl-recipe).

### Training vs Inference Engine Split

Design rationale: actor and reference models use the training engine for log-prob computation to match backward-pass precision; the teacher uses the inference engine for throughput on a large frozen model (verl Issues #5731).

### Padding-Mode Symmetry

Both `use_remove_padding=True` and `use_remove_padding=False` paths must handle top-k for `teacher_logprobs`. The `False` path was missing top-k handling, causing `KeyError` on `distillation_losses` (verl Issues #6293). Padding-mode symmetry is required for OPD to work in all configurations (verl Issues #6293).

### PR #6127 Regression

PR #6127 introduced `nested_tensor_from_tensor_list` assuming the ragged dimension is always last. This breaks OPD `teacher_logprobs` where the ragged dimension is at index 1, shape `(batch, [seq_len], topk)` (verl Issues #6152).

### Fully-Async Pipeline Limitation

The fully-async pipeline lacks GenRM support. Distillation and reward-model hybrid workflows require the synchronous training path today (verl Issues #5949).

## Diagnostic Relevance

- **`KeyError: distillation_losses`**: Check padding mode; ensure top-k handling exists for both `use_remove_padding=True` and `False` paths (verl Issues #6293).
- **`RuntimeError` in `nested_tensor_from_tensor_list`**: Suspect PR #6127 regression when ragged dim is not last; affects `teacher_logprobs` in OPD (verl Issues #6152).
- **Slow teacher sync**: Compare batched NCCL sync (GKD recipe) vs naive streaming; expect ~12× difference (verl-recipe).
- **GenRM + distillation in async mode**: Not supported; use sync path for hybrid distillation/reward-model workflows (verl Issues #5949).
- **Log-prob precision mismatch**: Verify actor/ref use training engine and teacher uses inference engine as designed (verl Issues #5731).

## Related Pages

- [Async Training](async-training.md) — GKD schedulers overlap rollout/teacher/training; fully-async lacks GenRM
- [verl](../entities/verl.md) — OPD listed as supported algorithm with `distillation` advantage estimator
- [FSDP Training](fsdp-training.md) — training backend used for actor/ref log-prob computation in OPD

## Sources

- [verl-recipe](../sources/verl-recipe.md) — GKD one-step-off/two-step-off schedulers, batched NCCL weight sync ~12× speedup
- [verl-docs](../sources/verl-docs.md) — OPD algorithm configuration (`algorithm.adv_estimator: distillation`)
- [verl Issues](../sources/verl-issues.md) — training/inference engine split rationale (#5731), padding-mode top-k symmetry (#6293), PR #6127 nested tensor regression (#6152), fully-async GenRM limitation (#5949)
