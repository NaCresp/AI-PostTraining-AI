# NeMo RL

NVIDIA's open-source post-training library for LLM and VLM reinforcement learning. Modular design with Ray orchestration, DTensor and Megatron Core backends, and HuggingFace model integration.

## Details

### Supported Algorithms

| Algorithm | Entry Point |
|-----------|-------------|
| GRPO | `examples/run_grpo.py` |
| DAPO | DAPO config + guide |
| SFT | `guides/sft.md` |
| DPO | `guides/dpo.md` |
| RM | Reward model training |
| On-policy distillation | `about/algorithms/on-policy-distillation.md` |

### Training Backends

- **DTensor**: PyTorch-native, good for prototyping and medium-scale
- **Megatron Core**: Large MoE, PP/TP/EP/CP for trillion-parameter models

### Custom Environments

Reward computation via `nemo_rl.environments`:
- `math_environment` — math verification
- `code_environment` — code execution
- `vlm_environment` — vision-language tasks
- `dapo_math_verifier` — DAPO-style math scoring
- Custom environments plug into the GRPO training loop

### Config System

YAML-driven with CLI overrides:

```bash
uv run python examples/run_grpo.py \
  policy.model_name="meta-llama/Llama-3.2-1B-Instruct" \
  cluster.gpus_per_node=8 \
  logger.wandb_enabled=True
```

### Advanced Features

- Async GRPO (`guides/async-grpo.md`)
- Eagle3 speculative decoding for faster rollouts
- FP8 quantization (`fp8.md`)
- YaRN long-context training
- Muon optimizer support
- Sequence packing and dynamic batching

## Relevance

NeMo RL is NVIDIA's production path for multimodal RL at scale. Strongest when already in the NeMo ecosystem or needing Megatron Core parallelism with custom reward environments.

## Related Pages

- [GRPO](grpo.md) — algorithm reference
- [DAPO](dapo.md) — DAPO recipe in NeMo RL
- [Batch Size Tuning](../concepts/batch-size-tuning.md) — sequence packing design
- [Async Training](../concepts/async-training.md) — async GRPO guide
- [verl](verl.md) — alternative RL framework

## Sources

- [NeMo RL Documentation](../sources/nemo-rl-docs.md) — algorithms, environments, backends, guides
