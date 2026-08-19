#!/usr/bin/env python3
"""Fine-tune iteration 2: continue from v1 checkpoint with focused data."""
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

MODEL_PATH = "checkpoints_v3/final"
DATA_PATH = "tokenized_data_v2"
OUTPUT_DIR = "checkpoints_v4"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model from v1 checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print("Loading pre-tokenized dataset...")
    dataset = load_from_disk(DATA_PATH)
    print(f"Dataset size: {len(dataset)} packed chunks")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=42,
        max_grad_norm=1.0,
    )

    print("Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    print("Starting training iteration 2...")
    trainer.train()

    print("Saving final model...")
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("TRAINING V2 COMPLETE!")

if __name__ == "__main__":
    main()
