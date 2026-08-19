#!/usr/bin/env python3
"""SFT training v7: from base model with eval-aligned data.

Training data format exactly matches HumanEval evaluation pipeline:
- User prompt matches INSTRUCTION + function signature
- Assistant response is just the function body (no markdown, no signature)
- Think tags are included as the template prepends them

Uses SFTTrainer with packing=False to preserve EOS tokens.
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_PATH = "Qwen/Qwen3-1.7B"
DATA_PATH = "training_data_v7.jsonl"
OUTPUT_DIR = "checkpoints_v7"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"Dataset size: {len(dataset)}")

    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=42,
        max_grad_norm=1.0,
        max_length=2048,
        dataset_text_field="text",
        packing=False,
    )

    print("Creating SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training v7...")
    trainer.train()

    print("Saving final model...")
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("TRAINING V7 COMPLETE!")

if __name__ == "__main__":
    main()
