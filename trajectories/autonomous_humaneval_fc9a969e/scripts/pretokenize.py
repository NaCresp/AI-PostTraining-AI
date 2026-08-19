#!/usr/bin/env python3
"""Pre-tokenize data for fast loading during training."""
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_PATH = "/home/user/models/Qwen/Qwen3-1.7B-Base"
DATA_PATH = "training_data.jsonl"
OUTPUT_PATH = "tokenized_data"
MAX_SEQ_LENGTH = 2048

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"Raw dataset size: {len(dataset)}")

    def tokenize_and_pack(examples):
        """Tokenize and concatenate texts into chunks of max_seq_length."""
        tokenized = tokenizer(examples["text"], truncation=False)

        # Concatenate all tokenized texts
        concatenated = {k: [] for k in tokenized.keys()}
        for k in tokenized.keys():
            for seq in tokenized[k]:
                concatenated[k].extend(seq)

        # Split into chunks
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH

        result = {
            k: [concatenated[k][i:i + MAX_SEQ_LENGTH] for i in range(0, total_length, MAX_SEQ_LENGTH)]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("Tokenizing and packing...")
    tokenized_dataset = dataset.map(
        tokenize_and_pack,
        batched=True,
        batch_size=1000,
        num_proc=16,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    print(f"Packed dataset size: {len(tokenized_dataset)} chunks of {MAX_SEQ_LENGTH} tokens")

    print("Saving...")
    tokenized_dataset.save_to_disk(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
