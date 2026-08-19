#!/usr/bin/env python3
"""Fine-tune Qwen3-1.7B-Base for code generation - optimized v2."""
import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Config
MODEL_PATH = "/home/user/models/Qwen/Qwen3-1.7B-Base"
DATA_PATH = "training_data.jsonl"
OUTPUT_DIR = "checkpoints_v2"
MAX_SEQ_LENGTH = 2048

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
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

    print("Loading and tokenizing dataset...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"Raw dataset size: {len(dataset)}")

    # Tokenize
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        # For causal LM, labels = input_ids
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    print(f"Tokenized dataset size: {len(tokenized_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        # Effective batch size: 4 GPUs * 8 * 2 = 64
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=42,
        max_grad_norm=1.0,
    )

    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    print("TRAINING COMPLETE!")

if __name__ == "__main__":
    main()
