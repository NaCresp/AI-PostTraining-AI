"""Merge FSDP checkpoint shards into a single HuggingFace model checkpoint.
Usage: python3 merge_fsdp_checkpoint.py <fsdp_ckpt_dir> <output_dir> <base_model_path>
"""
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge(fsdp_dir, output_dir, base_model_path):
    actor_dir = os.path.join(fsdp_dir, "actor")

    print(f"Loading base model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    shard_files = sorted([
        f for f in os.listdir(actor_dir)
        if f.startswith("model_") and f.endswith(".pt")
    ])

    if not shard_files:
        shard_files = sorted([
            f for f in os.listdir(actor_dir)
            if f.endswith(".pt") or f.endswith(".safetensors")
        ])

    print(f"Found {len(shard_files)} shard files in {actor_dir}")

    merged_state = {}
    for sf in shard_files:
        path = os.path.join(actor_dir, sf)
        print(f"  Loading {sf}...")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(shard, dict):
            merged_state.update(shard)
        else:
            print(f"  Warning: {sf} is not a dict, skipping")

    print(f"Merged state dict has {len(merged_state)} keys")

    if merged_state:
        sample_key = list(merged_state.keys())[0]
        print(f"  Sample key: {sample_key}")

    clean_state = {}
    for k, v in merged_state.items():
        clean_key = k.replace("module.", "").replace("_fsdp_wrapped_module.", "")
        clean_state[clean_key] = v

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(clean_state.keys())

    missing = model_keys - ckpt_keys
    extra = ckpt_keys - model_keys

    if missing:
        print(f"  Warning: {len(missing)} keys missing from checkpoint (using base model weights)")
        for k in sorted(missing)[:5]:
            print(f"    {k}")
    if extra:
        print(f"  Warning: {len(extra)} extra keys in checkpoint (ignoring)")
        for k in sorted(extra)[:5]:
            print(f"    {k}")

    model.load_state_dict(clean_state, strict=False)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 merge_fsdp_checkpoint.py <fsdp_ckpt_dir> <output_dir> <base_model_path>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
