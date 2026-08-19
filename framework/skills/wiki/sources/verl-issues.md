# verl GitHub Closed Issues

- **Type**: issue tracker
- **Raw location**: `raw/issues/verl-issues.json`
- **Ingested**: 2026-05-27

## Key Takeaways

1. Silent correctness bugs dominate production failures: wrong attention masks (#6278), tokenizer API changes (#6080), fused kernel interactions (#6068), and dtype autocast hardcoding (#5932) produce bad training without crashes.
2. LoRA checkpoint export is fragile — `adapter_config.json` can save `lora_alpha=0` despite training with `lora_alpha=128`, making merged weights identical to base (#6380). Megatron-Bridge LoRA underperforms FSDP2 by ~10% or collapses (#5094).
3. FSDP2 has multiple infrastructure-level issues: CPUOffloadPolicy breaks update_weights (#5995), NCCL deadlock on new architectures from heterogeneous buffer ordering (#6365), dynamic batching without dp_group causes deadlock (#6176, #6178).
4. `transformers>=5.0` breaks multiple paths: tokenizer API returns BatchEncoding instead of list[int] (#6080), wrap policy fails on partial _no_split_modules (#6289), flash_attn varlen bugs (#6281).
5. VLM post-training requires special handling: RoPE/position_ids fragility (#4483, #6073), fused kernels + Ulysses SP>1 = silent quality regression (#6068), vision tensor CPU placement (#4906), TITO path needed to prevent retokenization drift (#6168).
6. Fully-async pipeline has known limitations: lacks GenRM support (#5949), checkpoint save failures on large MoE (#6026), rollout concurrency hardcoded to 16 (#6306).
7. On-policy distillation needs padding-mode symmetry for top-k handling (#6293) and ragged-dim-aware nested tensors (#6152).
8. Multi-turn tool-calling: ReMax fails with multiturn (#3027), MoE routing tensors must match truncation (#6027), per-row tools unsupported in RLHFDataset (#6096).
9. Multimodal data pipeline: pop() vs get() for image/video fields causes 1000× filter slowdown (#6145); TransferQueue needs jagged tensors ≥0.1.7 (#6407, #6261).
10. FP8 rollout quantization not plug-and-play — shape mismatch during weight sync (#6112); colocated vLLM IPC sockets collide across jobs (#6233).

## Pages Created or Updated

- [LoRA](../entities/lora.md) — new entity page: checkpoint fragility, Megatron-Bridge issues, working GRPO config
- [SFT Cold Start](../concepts/sft-cold-start.md) — new concept page: prerequisite SFT phase, VLM SFT considerations
- [FSDP Training](../concepts/fsdp-training.md) — new concept page: CPUOffloadPolicy, NCCL deadlock, wrap policy, dtype issues
- [On-Policy Distillation](../concepts/on-policy-distillation.md) — new concept page: padding-mode symmetry, nested tensor regression
- [VLM Post-Training](../concepts/vlm-post-training.md) — new concept page: RoPE fragility, fused kernels, TITO path
- [Data Pipeline](../concepts/data-pipeline.md) — new concept page: tokenizer API, TransferQueue, multimodal filtering
- [SFT Training Guide](../experience/sft-training-guide.md) — new experience page
- [VLM Post-Training Guide](../experience/vlm-post-training-guide.md) — new experience page
- [Distillation Guide](../experience/distillation-guide.md) — new experience page
- [Infrastructure Debugging](../experience/infrastructure-debugging.md) — new experience page
- [GRPO](../entities/grpo.md) — updated with production issues: attention mask bug, tokenizer API, LoRA config
- [DAPO](../entities/dapo.md) — updated with dynamic sampling hard masks, Gemma4 instability
- [verl](../entities/verl.md) — updated with cross-references to new pages
- [Multi-Turn Tool RL](../concepts/multi-turn-tool-rl.md) — updated with ReMax failure, MoE routing, per-row tools
- [Async Training](../concepts/async-training.md) — updated with concurrency cap, checkpoint failures, GenRM limitation
- [Training-Inference Mismatch](../concepts/training-inference-mismatch.md) — updated with FSDP2 sync, FP8, retokenization drift
- [Reward Design](../concepts/reward-design.md) — updated with tokenizer API, async GenRM limitation
- [Training Collapse Triage](../experience/training-collapse-triage.md) — silent correctness bug triage (via [RL Training Principles](../experience/rl-training-principles.md))
- [Stable GRPO Training](../experience/stable-grpo-training.md) — updated derived-from references

## Notes

- 82 closed issues fetched from verl-project/verl (pages 1-4; remaining pages rate-limited).
- verl-project/verl-recipe had only 4 closed issues (import error, NPU recipe RFC, license, missing script), none containing generalizable training knowledge.
- Skipped ~28 issues that were purely Ascend NPU hardware, Docker/environment setup, or documentation hygiene with no generalizable training knowledge.
- Issue comments were partially fetched (rate-limited after page 3); body and title were sufficient for most knowledge extraction.
