#!/usr/bin/env python3
"""
Fine-tune Qwen3-1.7B-Base on math data for GSM8K using TRL SFTTrainer.
Optimized for 4x A800 GPUs.
"""
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
import argparse

MODEL_PATH = "/home/user/models/Qwen/Qwen3-1.7B-Base"

def load_data(data_file):
    """Load training data and convert to HF dataset."""
    examples = []
    with open(data_file, "r") as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
    return Dataset.from_list(examples)

def formatting_func(example):
    """Format a single example as a chat string."""
    messages = example["messages"]
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return text

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
    parser.add_argument("--save-steps", type=int, default=1000)
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
    dataset = load_data(args.data_file)
    print(f"Dataset size: {len(dataset)}")

    # Use completion-only data collator to only train on assistant responses
    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    training_args = SFTConfig(
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
        report_to="none",
        seed=42,
        max_grad_norm=1.0,
        max_seq_length=args.max_length,
        dataset_text_field=None,  # We use formatting_func
        # FSDP
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
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
