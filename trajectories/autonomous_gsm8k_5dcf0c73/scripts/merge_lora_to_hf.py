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
        type=str,
        default="/home/user/models/Qwen/Qwen3-1.7B-Base",
    )
    p.add_argument("--lora-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="final_model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

