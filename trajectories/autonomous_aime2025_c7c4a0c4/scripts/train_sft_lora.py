#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@dataclass
class TokenizedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class ChatSFTDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path: Path, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        ds = load_dataset("json", data_files=str(jsonl_path), split="train")
        self.rows = ds

    def __len__(self) -> int:
        return self.rows.num_rows

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        messages = row["messages"]
        user_msgs = [m for m in messages if m["role"] != "assistant"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if len(assistant_msgs) != 1:
            raise ValueError(f"Expected 1 assistant message, got {len(assistant_msgs)}")
        assistant_text = assistant_msgs[0]["content"]

        prompt_text = self.tokenizer.apply_chat_template(
            user_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        response_ids = self.tokenizer(
            assistant_text + self.tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, self.max_length - len(prompt_ids)),
        )["input_ids"]

        input_ids = prompt_ids + response_ids
        input_ids = input_ids[: self.max_length]

        labels = [-100] * len(prompt_ids) + response_ids
        labels = labels[: self.max_length]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--init-lora", type=Path, default=None)
    p.add_argument("--train-jsonl", type=Path, default=Path("data/sft/train.jsonl"))
    p.add_argument("--val-jsonl", type=Path, default=Path("data/sft/val.jsonl"))
    p.add_argument("--chat-template", type=Path, default=Path("templates/qwen3.jinja"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/sft_lora"))
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume-from", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.chat_template = read_text(args.chat_template)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if args.init_lora is not None:
        model = PeftModel.from_pretrained(model, args.init_lora, is_trainable=True)
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = ChatSFTDataset(tokenizer, args.train_jsonl, max_length=args.max_length)
    val_ds = ChatSFTDataset(tokenizer, args.val_jsonl, max_length=args.max_length)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=DataCollator(tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    resume = str(args.resume_from) if args.resume_from is not None else None
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(str(args.output_dir / "lora"))
    tokenizer.save_pretrained(str(args.output_dir / "lora"))

    meta = {
        "base_model": args.base_model,
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "chat_template": str(args.chat_template),
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
