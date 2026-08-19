"""Prepare larger augmented dataset for SFT v3: GSM8K + more OpenMathInstruct-2 data.
Filter for #### format and reasonable length."""
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

all_texts = []

# 1. GSM8K train (all examples - core data)
gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
for ex in gsm_ds:
    text = format_text(ex["question"], ex["answer"])
    all_texts.append(text)
print(f"GSM8K train: {len(all_texts)} examples")

# 2. OpenMathInstruct-2: Get GSM8K-style + augmented_math problems
omi_ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
omi_gsm = 0
omi_math = 0
omi_target_gsm = 20000
omi_target_math = 10000

for ex in omi_ds:
    src = ex.get("problem_source", "")
    problem = ex["problem"]
    solution = ex["generated_solution"]
    expected = ex.get("expected_answer", "")

    # Convert to #### format
    if "####" not in solution and expected:
        solution = solution.rstrip() + f"\n#### {expected}"

    answer = extract_final_answer(solution)
    if answer is None:
        continue

    # Filter: skip very long solutions (>800 chars) or very short (<50 chars)
    if len(solution) < 50 or len(solution) > 800:
        continue

    if src in ("gsm8k", "augmented_gsm8k") and omi_gsm < omi_target_gsm:
        all_texts.append(format_text(problem, solution))
        omi_gsm += 1
    elif src in ("augmented_math", "math") and omi_math < omi_target_math:
        all_texts.append(format_text(problem, solution))
        omi_math += 1

    if omi_gsm >= omi_target_gsm and omi_math >= omi_target_math:
        break

print(f"OpenMathInstruct-2 GSM8K-style: {omi_gsm}")
print(f"OpenMathInstruct-2 math: {omi_math}")
print(f"Total dataset: {len(all_texts)} examples")

# Shuffle
random.seed(42)
random.shuffle(all_texts)

with open("data/sft_v3_augmented.jsonl", "w") as f:
    for text in all_texts:
        f.write(json.dumps({"text": text}) + "\n")

print(f"Saved to data/sft_v3_augmented.jsonl")
