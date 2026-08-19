"""GRPO v8: Very low KL penalty for maximum exploration from best model.
- beta=0.01 (much lower than v3's 0.04) - allows big policy shifts
- lr=1e-6 (very low to compensate for low KL penalty)
- 100 steps (short to catch peak before overshoot)
- Save every 20 steps for fine-grained peak detection
- temperature=1.0 (higher for more diversity)
"""
import os
import re
import sys
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

SFT_MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/grpo_v3/checkpoint-200"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/grpo_v8"

SYSTEM_PROMPT = "Solve the following math problem step by step. Put your final answer after ####."

print(f"Starting from: {SFT_MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = "<|im_end|>"

ds = load_dataset("openai/gsm8k", "main", split="train")

def extract_answer(text):
    matches = re.findall(r'####\s*(-?[\d,]+\.?\d*)', text)
    if matches:
        return matches[-1].replace(",", "").strip()
    return None

def format_prompt(example):
    question = example["question"]
    ground_truth = extract_answer(example["answer"])
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return {"prompt": prompt, "ground_truth": ground_truth}

ds = ds.map(format_prompt, remove_columns=ds.column_names)

def reward_fn(completions, ground_truth, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = completion
        pred = extract_answer(text)
        if pred is not None and gt is not None:
            try:
                if float(pred) == float(gt):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=100,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=6,
    max_prompt_length=512,
    max_completion_length=1024,
    num_generations=8,
    temperature=1.0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
    ddp_find_unused_parameters=False,
    beta=0.01,
)

model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=ds,
    reward_funcs=[reward_fn],
)

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print(f"GRPO v8 training complete. Model saved to {OUTPUT_DIR}/final")
