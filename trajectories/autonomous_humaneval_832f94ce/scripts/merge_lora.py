#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-model",
        default="/home/user/models/Qwen/Qwen3-1.7B-Base",
        help="Path to the base model (must be Qwen3-1.7B-Base).",
    )
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.realpath(args.base_model) != os.path.realpath("/home/user/models/Qwen/Qwen3-1.7B-Base"):
        raise SystemExit("Refusing to merge: --base-model must be /home/user/models/Qwen/Qwen3-1.7B-Base")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(args.output_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir, safe_serialization=True)


if __name__ == "__main__":
    main()

