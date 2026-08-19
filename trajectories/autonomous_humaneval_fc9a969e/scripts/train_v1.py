#!/usr/bin/env python3
"""Fine-tune Qwen3-1.7B-Base for code generation using SFT."""
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig

# Config
MODEL_PATH = "/home/user/models/Qwen/Qwen3-1.7B-Base"
DATA_PATH = "training_data.jsonl"
OUTPUT_DIR = "checkpoints_v1"
MAX_SEQ_LENGTH = 2048

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"Dataset size: {len(dataset)}")

    # Training config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # Effective batch size: 4 GPUs * 4 per_device * 4 grad_accum = 64
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        max_length=MAX_SEQ_LENGTH,
        packing=True,
        dataset_text_field="text",
        dataloader_num_workers=4,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=42,
    )

    print("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
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
