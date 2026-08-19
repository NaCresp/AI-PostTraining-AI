"""Prepare augmented math dataset for SFT v2: GSM8K train + OpenMathInstruct-2 GSM8K-style problems."""
import json
import re
import random
from datasets import load_dataset

SYSTEM_PROMPT = "Solve the following math problem step by step. Put your final answer after ####."

def extract_final_answer(text):
    matches = re.findall(r'####\s*(-?[\d,]+\.?\d*)', text)
    if matches:
        return matches[-1].replace(",", "").strip()
    return None

def format_gsm8k(example):
    question = example["question"]
    answer = example["answer"]
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>\n"
    )
    return text

# 1. Load GSM8K train
gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
all_texts = []
for ex in gsm_ds:
    text = format_gsm8k(ex)
    all_texts.append(text)
print(f"GSM8K train: {len(all_texts)} examples")

# 2. Load OpenMathInstruct-2 GSM8K-style problems
omi_ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
omi_count = 0
omi_target = 15000
for ex in omi_ds:
    src = ex.get("problem_source", "")
    if src in ("gsm8k", "augmented_gsm8k"):
        problem = ex["problem"]
        solution = ex["generated_solution"]
        expected = ex.get("expected_answer", "")
        # Convert to #### format if not already present
        if "####" not in solution and expected:
            solution = solution.rstrip() + f"\n#### {expected}"
        # Verify it has #### format
        if extract_final_answer(solution) is not None:
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{problem}<|im_end|>\n"
                f"<|im_start|>assistant\n{solution}<|im_end|>\n"
            )
            all_texts.append(text)
            omi_count += 1
            if omi_count >= omi_target:
                break

print(f"OpenMathInstruct-2 GSM8K-style: {omi_count} examples")
print(f"Total dataset: {len(all_texts)} examples")

# Shuffle
random.seed(42)
random.shuffle(all_texts)

# Save as jsonl
with open("data/sft_augmented.jsonl", "w") as f:
    for text in all_texts:
        f.write(json.dumps({"text": text}) + "\n")

print(f"Saved to data/sft_augmented.jsonl")
