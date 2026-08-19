#!/usr/bin/env python3
"""
Fine-tune Qwen3-1.7B-Base on math data for GSM8K.
Uses full fine-tuning with DeepSpeed ZeRO-2 or FSDP on 4 GPUs.
"""
import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

MODEL_PATH = "/home/user/models/Qwen/Qwen3-1.7B-Base"
OUTPUT_DIR = "/home/ben/task/checkpoints"

class MathChatDataset(Dataset):
    """Dataset that formats math problems as chat conversations."""

    def __init__(self, data_file, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_file, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} examples from {data_file}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        messages = example["messages"]

        # Format as Qwen3 chat format
        # <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>
        text = ""
        for msg in messages:
            text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # For causal LM, labels = input_ids, but we mask the user part
        labels = input_ids.copy()

        # Find where the assistant response starts to mask user tokens
        # We want to only compute loss on assistant tokens
        # Find the assistant start token position
        assistant_token = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

        # Find the position of assistant start
        assistant_start = -1
        for i in range(len(input_ids) - len(assistant_token) + 1):
            if input_ids[i:i+len(assistant_token)] == assistant_token:
                assistant_start = i + len(assistant_token)
                break

        # Mask everything before assistant response
        if assistant_start > 0:
            labels[:assistant_start] = [-100] * assistant_start

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="/home/ben/task/training_data/train.jsonl")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Ensure pad token
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

    print(f"Model loaded. Parameters: {model.num_parameters():,}")

    print(f"Loading dataset from {args.data_file}...")
    dataset = MathChatDataset(args.data_file, tokenizer, max_length=args.max_length)

    # Data collator for variable-length sequences
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # Training arguments
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
        save_total_limit=3,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        max_grad_norm=1.0,
        # FSDP settings for multi-GPU
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    print("Done!")


if __name__ == "__main__":
    main()
