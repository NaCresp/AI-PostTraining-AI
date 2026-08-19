#!/usr/bin/env python3
"""
SFT Training script for Qwen3-1.7B-Base on math competition data.
Uses TRL SFTTrainer with DeepSpeed ZeRO-2 for multi-GPU training.
"""
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_PATH = "/home/user/models/Qwen/Qwen3-1.7B-Base"
TRAIN_DATA = "/home/ben/task/train_data/train_focused.jsonl"
VAL_DATA = "/home/ben/task/train_data/val_focused.jsonl"
OUTPUT_DIR = "/home/ben/task/sft_output"
MAX_SEQ_LENGTH = 4096

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    dataset = load_dataset("json", data_files={
        "train": TRAIN_DATA,
        "validation": VAL_DATA,
    })

    print(f"Train size: {len(dataset['train'])}")
    print(f"Val size: {len(dataset['validation'])}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_SEQ_LENGTH,
        packing=True,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        report_to="none",
        seed=42,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print("Training complete!")

if __name__ == "__main__":
    main()
