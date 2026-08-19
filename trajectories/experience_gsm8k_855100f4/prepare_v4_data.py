"""Prepare SFT v4 dataset: GSM8K + OpenMathInstruct-2 + NuminaMath-CoT (grade school level).
Filter for #### format, reasonable length, and grade-school difficulty."""
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

def format_text(question, answer):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>\n"
    )

# Start with existing v3 data
all_texts = []
with open("data/sft_v3_augmented.jsonl") as f:
    for line in f:
        all_texts.append(json.loads(line)["text"])
print(f"Starting with v3 data: {len(all_texts)} examples")

# Add NuminaMath-CoT examples (filtered for grade-school/basic level)
numi_ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
numi_count = 0
numi_target = 15000
grade_school_sources = {"synthetic_math", "gsm8k", "cn_k12", "orca_math", "synthetic_amc"}

for ex in numi_ds:
    src = ex.get("source", "")
    # Only take grade-school/basic level problems
    if src not in grade_school_sources:
        continue

    problem = ex.get("problem", "")
    solution = ex.get("solution", "")

    if not problem or not solution:
        continue

    # Skip very long problems/solutions
    if len(problem) > 500 or len(solution) > 1000:
        continue

    # Skip if no numeric answer can be extracted
    # Try to extract from "The answer is X" or similar patterns
    answer_patterns = [
        r'\\boxed\{([^}]+)\}',
        r'(?:answer is|= )\s*\$?(-?[\d,]+\.?\d*)',
        r'(-?[\d,]+\.?\d*)\s*$',
    ]

    final_answer = None
    for pat in answer_patterns:
        m = re.findall(pat, solution)
        if m:
            final_answer = m[-1].replace(",", "").strip()
            try:
                float(final_answer)
                break
            except ValueError:
                final_answer = None

    if final_answer is None:
        continue

    # Add #### format to solution if not present
    if "####" not in solution:
        solution = solution.rstrip() + f"\n#### {final_answer}"

    text = format_text(problem, solution)
    all_texts.append(text)
    numi_count += 1
    if numi_count >= numi_target:
        break

print(f"NuminaMath-CoT added: {numi_count}")
print(f"Total dataset: {len(all_texts)} examples")

random.seed(42)
random.shuffle(all_texts)

with open("data/sft_v4_augmented.jsonl", "w") as f:
    for text in all_texts:
        f.write(json.dumps({"text": text}) + "\n")

print(f"Saved to data/sft_v4_augmented.jsonl")
