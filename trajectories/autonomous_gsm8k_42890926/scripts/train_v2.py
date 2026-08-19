#!/usr/bin/env python3
"""
Fine-tune Qwen3-1.7B-Base on math data for GSM8K.
Uses HuggingFace Trainer with FSDP on 4 GPUs.
Masks user tokens in labels so we only train on assistant responses.
"""
import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

MODEL_PATH = "/home/user/models/Qwen/Qwen3-1.7B-Base"

@dataclass
class PaddingCollator:
    """Collator that pads to max length in batch and handles labels."""
    tokenizer: Any
    max_length: int = 768
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length
        )
        # Pad to multiple
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) //
                      self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            input_ids = f["input_ids"][:max_len]
            labels = f["labels"][:max_len]
            attn = [1] * len(input_ids)

            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attn = attn + [0] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attn)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class MathDataset(Dataset):
    """Dataset that formats math problems as chat conversations."""

    def __init__(self, data_file, tokenizer, max_length=768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Pre-compute the response template token IDs
        self.response_template_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

        with open(data_file, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} examples from {data_file}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        messages = example["messages"]

        # Format as Qwen3 chat
        text = ""
        for msg in messages:
            text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()

        # Find the assistant response start and mask everything before it
        resp_ids = self.response_template_ids
        resp_len = len(resp_ids)

        assistant_start = -1
        for i in range(len(input_ids) - resp_len + 1):
            if input_ids[i:i+resp_len] == resp_ids:
                assistant_start = i + resp_len
                break

        if assistant_start > 0:
            labels[:assistant_start] = [-100] * assistant_start

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="/home/ben/task/training_data/train.jsonl")
    parser.add_argument("--output-dir", type=str, default="/home/ben/task/checkpoints")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    print(f"Model parameters: {model.num_parameters():,}")

    print(f"Loading dataset from {args.data_file}...")
    dataset = MathDataset(args.data_file, tokenizer, max_length=args.max_length)

    collator = PaddingCollator(tokenizer=tokenizer, max_length=args.max_length)

    # Effective batch size = batch_size * grad_accum * num_gpus = 4*4*4 = 64
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        max_grad_norm=1.0,
        # FSDP
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"Model saved to {final_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
