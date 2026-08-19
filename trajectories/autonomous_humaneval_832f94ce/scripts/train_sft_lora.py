#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-model",
        default="/home/user/models/Qwen/Qwen3-1.7B-Base",
        help="Path to the base model (must be Qwen3-1.7B-Base).",
    )
    p.add_argument("--dataset-jsonl", required=True, help="JSONL with prompt/completion chat messages.")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable example packing into max_length sequences.",
    )
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA injection.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.realpath(args.base_model) != os.path.realpath("/home/user/models/Qwen/Qwen3-1.7B-Base"):
        raise SystemExit("Refusing to train: --base-model must be /home/user/models/Qwen/Qwen3-1.7B-Base")

    ds = load_dataset("json", data_files=args.dataset_jsonl, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        bf16=True,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        packing=args.packing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        chat_template_path="templates/qwen3.jinja",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
