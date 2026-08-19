#!/usr/bin/env python3
"""
Prepare training data for AIME math competition fine-tuning.
Sources:
1. OpenR1-Math-220k - Has <think> reasoning traces from DeepSeek-R1
2. NuminaMath-CoT - Competition math with chain-of-thought solutions
3. AIME 2024 historical data (for AIME-specific formatting)

Format: Qwen3 chat format with <think> reasoning blocks.
The evaluation expects:
- <|im_start|>system\n...<|im_end|>
- <|im_start|>user\n...<|im_end|>
- <|im_start|>assistant\n<think>\n...\n</think>\n\nANSWER: ...<|im_end|>
"""

import json
import re
import os
import random
from datasets import load_dataset

OUTPUT_DIR = "/home/ben/task/train_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a mathematical reasoning assistant. Solve problems step by step, showing your detailed reasoning process. For competition math problems, provide your final answer as a specific number.

The last line of your response should be of the form "ANSWER: $ANSWER" where $ANSWER is the answer to the problem."""

def extract_numeric_answer(text):
    """Try to extract a numeric answer from solution text."""
    # Look for \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        ans = boxed[-1].strip()
        # Try to see if it's numeric
        try:
            val = float(ans.replace(',', ''))
            if val == int(val):
                return str(int(val))
            return ans
        except:
            return ans

    # Look for "ANSWER: ..."
    answer_match = re.search(r'ANSWER:\s*(.+)', text)
    if answer_match:
        return answer_match.group(1).strip()

    return None

def format_answer_line(answer):
    """Format the final answer line."""
    return f"ANSWER: {answer}"

def process_openr1_sample(sample):
    """Process a sample from OpenR1-Math-220k."""
    messages = sample.get('messages', [])
    if len(messages) < 2:
        return None

    problem = messages[0].get('content', '').strip()
    assistant_content = messages[1].get('content', '').strip()

    if not problem or not assistant_content:
        return None

    # Check correctness - prefer verified correct solutions
    correctness = sample.get('correctness_math_verify', [])
    if correctness and not any(correctness):
        return None

    # The assistant content already has <think>...</think> format
    # We need to ensure it ends with ANSWER: format
    answer = sample.get('answer', '')

    # Extract the thinking and response parts
    if '<think>' in assistant_content and '</think>' in assistant_content:
        think_match = re.search(r'<think>(.*?)</think>(.*)', assistant_content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            response = think_match.group(2).strip()
        else:
            thinking = assistant_content
            response = ""
    else:
        thinking = assistant_content
        response = ""

    # Build the response with explicit ANSWER: line
    if answer:
        # Clean up the answer
        clean_answer = str(answer).strip()
        # Check if the response already has ANSWER:
        if 'ANSWER:' not in response:
            if response:
                final_response = f"<think>\n{thinking}\n</think>\n\n{response}\n\nANSWER: {clean_answer}"
            else:
                final_response = f"<think>\n{thinking}\n</think>\n\nANSWER: {clean_answer}"
        else:
            final_response = f"<think>\n{thinking}\n</think>\n\n{response}"
    else:
        # Try to extract answer from the response
        extracted = extract_numeric_answer(response) or extract_numeric_answer(thinking)
        if extracted:
            if 'ANSWER:' not in response:
                if response:
                    final_response = f"<think>\n{thinking}\n</think>\n\n{response}\n\nANSWER: {extracted}"
                else:
                    final_response = f"<think>\n{thinking}\n</think>\n\nANSWER: {extracted}"
            else:
                final_response = f"<think>\n{thinking}\n</think>\n\n{response}"
        else:
            # Skip samples without clear answers
            return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": final_response}
        ]
    }

def process_numina_sample(sample):
    """Process a sample from NuminaMath-CoT, wrapping in <think> format."""
    problem = sample.get('problem', '').strip()
    solution = sample.get('solution', '').strip()

    if not problem or not solution:
        return None

    # Extract the answer
    answer = extract_numeric_answer(solution)
    if not answer:
        return None

    # Wrap solution in <think> format and add ANSWER: line
    final_response = f"<think>\n{solution}\n</think>\n\nANSWER: {answer}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": final_response}
        ]
    }

def main():
    all_samples = []

    # 1. Load OpenR1-Math-220k (priority: competition math sources)
    print("Loading OpenR1-Math-220k...")
    competition_sources = {'amc_aime', 'aops_forum', 'olympiads', 'number_theory', 'inequalities', 'olympiads_ref', 'cn_contest'}

    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    openr1_count = 0
    openr1_competition = 0

    for sample in ds:
        source = sample.get('source', '')

        # Keep all competition math
        if source in competition_sources:
            processed = process_openr1_sample(sample)
            if processed:
                # Check content length - skip very long ones (>12000 chars) and very short ones
                total_len = sum(len(m['content']) for m in processed['messages'])
                if 100 < total_len < 12000:
                    all_samples.append(processed)
                    openr1_competition += 1

        openr1_count += 1
        if openr1_count % 10000 == 0:
            print(f"  Processed {openr1_count} OpenR1 samples, kept {openr1_competition} competition samples")

    print(f"OpenR1-Math-220k: {openr1_competition} competition math samples from {openr1_count} total")

    # 2. Load NuminaMath-CoT (filter for competition-level sources)
    print("\nLoading NuminaMath-CoT...")
    competition_numina_sources = {'amc_aime', 'aops_forum', 'olympiads', 'synthetic_amc', 'math'}

    ds_numina = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    numina_count = 0
    numina_kept = 0

    for sample in ds_numina:
        source = sample.get('source', '')

        if source in competition_numina_sources:
            processed = process_numina_sample(sample)
            if processed:
                total_len = sum(len(m['content']) for m in processed['messages'])
                if 100 < total_len < 12000:
                    all_samples.append(processed)
                    numina_kept += 1

        numina_count += 1
        if numina_count % 50000 == 0:
            print(f"  Processed {numina_count} NuminaMath samples, kept {numina_kept}")

    print(f"NuminaMath-CoT: {numina_kept} competition math samples from {numina_count} total")

    # 3. Shuffle and save
    print(f"\nTotal samples: {len(all_samples)}")
    random.seed(42)
    random.shuffle(all_samples)

    # Save as JSONL
    output_path = os.path.join(OUTPUT_DIR, "math_train.jsonl")
    with open(output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Saved {len(all_samples)} samples to {output_path}")

    # Also save a small validation set
    val_size = min(500, len(all_samples) // 20)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    val_path = os.path.join(OUTPUT_DIR, "math_val.jsonl")
    train_path = os.path.join(OUTPUT_DIR, "math_train_final.jsonl")

    with open(val_path, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')

    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Train: {len(train_samples)} samples -> {train_path}")
    print(f"Val: {len(val_samples)} samples -> {val_path}")

if __name__ == "__main__":
    main()
