"""GRPO training on GSM8K for math reasoning improvement. Starting from SFT v2 (71.87% baseline)."""
import os
import re
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

SFT_MODEL_PATH = "checkpoints/sft_v2/final"
OUTPUT_DIR = "checkpoints/grpo_v1"

SYSTEM_PROMPT = "Solve the following math problem step by step. Put your final answer after ####."

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
print(f"Dataset size: {len(ds)}")
print(f"Example prompt: {ds[0]['prompt']}")
print(f"Example ground_truth: {ds[0]['ground_truth']}")

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

def format_reward(completions, ground_truth, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = completion
        has_format = bool(re.search(r'####\s*-?[\d,]+\.?\d*', text))
        rewards.append(0.2 if has_format else 0.0)
    return rewards

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=300,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    max_prompt_length=512,
    max_completion_length=512,
    num_generations=8,
    temperature=0.9,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
    ddp_find_unused_parameters=False,
    reward_weights=[1.0, 0.5],
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
    reward_funcs=[reward_fn, format_reward],
)

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print(f"GRPO training complete. Model saved to {OUTPUT_DIR}/final")
