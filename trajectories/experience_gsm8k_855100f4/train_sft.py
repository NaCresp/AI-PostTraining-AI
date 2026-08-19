"""SFT training on GSM8K train split for Qwen3-1.7B-Base."""
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "/root/models/Qwen/Qwen3-1.7B-Base"
OUTPUT_DIR = "checkpoints/sft_v1"
MAX_SEQ_LEN = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = "<|im_end|>"

ds = load_dataset("openai/gsm8k", "main", split="train")

SYSTEM_PROMPT = "Solve the following math problem step by step. Put your final answer after ####."

def format_example(example):
    question = example["question"]
    answer = example["answer"]
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>\n"
    )
    return {"text": text}

ds = ds.map(format_example, remove_columns=ds.column_names)

print(f"Dataset size: {len(ds)}")
print(f"Example:\n{ds[0]['text'][:500]}")

lens = []
for ex in ds:
    toks = tokenizer(ex["text"], truncation=False)["input_ids"]
    lens.append(len(toks))
print(f"Token lengths: min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.0f}, median={sorted(lens)[len(lens)//2]}")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    max_length=MAX_SEQ_LEN,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    report_to="none",
    seed=42,
    packing=True,
    ddp_find_unused_parameters=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print(f"Training complete. Model saved to {OUTPUT_DIR}/final")
