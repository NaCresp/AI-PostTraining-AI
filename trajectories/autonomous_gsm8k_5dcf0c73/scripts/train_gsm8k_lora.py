#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.gsm8k_sft_data import GSM8K_DATASET, gsm8k_record_to_example  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-model",
        type=str,
        default="/home/user/models/Qwen/Qwen3-1.7B-Base",
    )
    p.add_argument("--output-dir", type=str, default="runs/gsm8k_lora")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=-1)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()


class SftTokenizedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = gsm8k_record_to_example(self.records[idx])
        user_msgs = [{"role": "user", "content": ex.prompt}]
        full_msgs = [
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": ex.completion},
        ]

        prompt_ids: list[int] = self.tokenizer.apply_chat_template(
            user_msgs,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids: list[int] = self.tokenizer.apply_chat_template(
            full_msgs,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.max_seq_len,
        )

        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[: len(input_ids)]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class PadCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in batch]
        labels = [x["labels"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_jsonable(v) for v in obj])
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(*GSM8K_DATASET, split="train")
    if args.limit_train is not None and args.limit_train > 0:
        raw = raw.select(range(min(args.limit_train, len(raw))))
    records = [raw[i] for i in range(len(raw))]

    train_ds = SftTokenizedDataset(records=records, tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "lora": _jsonable(lora.to_dict()),
                "base_model": args.base_model,
            },
            f,
            indent=2,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
