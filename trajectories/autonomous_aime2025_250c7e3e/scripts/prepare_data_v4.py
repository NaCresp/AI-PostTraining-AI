#!/usr/bin/env python3
"""
Prepare v4 training data:
- Only OpenR1 competition math (DeepSeek-R1 traces, verified correct)
- Strict integer answers 0-999
- Shorter, cleaner solutions (better for 1.7B model)
- Evaluation-matching format
"""
import json
import re
import random
from datasets import load_dataset

OUTPUT_TRAIN = "/home/ben/task/train_data/train_v4.jsonl"
OUTPUT_VAL = "/home/ben/task/train_data/val_v4.jsonl"

EVAL_PROMPT_PREFIX = """Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

"""

COMPETITION_SOURCES = {
    "amc_aime", "aops_forum", "olympiads", "number_theory",
    "inequalities", "olympiads_ref", "cn_contest"
}

def extract_think_and_answer(content):
    """Extract think content and answer from OpenR1 format."""
    # Find ANSWER line
    answer_match = re.search(r'ANSWER:\s*(.+?)$', content, re.MULTILINE)
    if not answer_match:
        # Try to find boxed answer
        boxed_match = re.findall(r'\\boxed\{([^}]*)\}', content)
        if boxed_match:
            answer_str = boxed_match[-1].strip()
        else:
            return None, None
    else:
        answer_str = answer_match.group(1).strip()

    # Clean answer
    answer_str = re.sub(r'\\boxed\{([^}]*)\}', r'\1', answer_str)
    answer_str = re.sub(r'\$([^$]*)\$', r'\1', answer_str)
    answer_str = answer_str.strip().rstrip('.')

    try:
        val = int(answer_str)
        if 0 <= val <= 999:
            return content, val
    except ValueError:
        try:
            val = int(float(answer_str))
            if float(answer_str) == val and 0 <= val <= 999:
                return content, val
        except:
            pass
    return None, None

def clean_response(content, answer_int):
    """Create clean <think>...</think>\n\nANSWER: N format."""
    # Extract reasoning
    if '<think>' in content and '</think>' in content:
        think_start = content.index('<think>') + len('<think>')
        think_end = content.index('</think>')
        reasoning = content[think_start:think_end].strip()
    elif '<think>' in content:
        think_start = content.index('<think>') + len('<think>')
        answer_match = re.search(r'ANSWER:', content)
        if answer_match:
            reasoning = content[think_start:answer_match.start()].strip()
        else:
            reasoning = content[think_start:].strip()
    else:
        answer_match = re.search(r'ANSWER:', content)
        if answer_match:
            reasoning = content[:answer_match.start()].strip()
        else:
            reasoning = content.strip()

    # Remove ANSWER lines from reasoning
    reasoning = re.sub(r'\nANSWER:.*$', '', reasoning, flags=re.MULTILINE).strip()

    if len(reasoning) < 50:
        return None

    return f"<think>\n{reasoning}\n</think>\n\nANSWER: {answer_int}"


def process_openr1():
    """Process OpenR1-Math-220k competition data."""
    print("Loading OpenR1-Math-220k...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")

    samples = []
    source_counts = {}

    for row in ds:
        source = row.get("source", "")
        if source not in COMPETITION_SOURCES:
            continue

        source_counts[source] = source_counts.get(source, 0) + 1

        # Check correctness
        if not row.get("correctness_math_verify", False):
            continue

        messages = row.get("messages", [])
        if len(messages) < 2:
            continue

        problem = ""
        solution = ""
        for msg in messages:
            if msg["role"] == "user":
                problem = msg["content"]
            elif msg["role"] == "assistant":
                solution = msg["content"]

        if not problem or not solution:
            continue

        # Extract and validate answer
        _, answer_int = extract_think_and_answer(solution)
        if answer_int is None:
            continue

        # Clean the response
        cleaned = clean_response(solution, answer_int)
        if cleaned is None:
            continue

        # Skip very long solutions (hard for 1.7B to learn)
        if len(cleaned) > 6000:
            continue

        sample = {
            "messages": [
                {"role": "user", "content": EVAL_PROMPT_PREFIX + problem},
                {"role": "assistant", "content": cleaned}
            ]
        }
        samples.append(sample)

    print(f"OpenR1 source distribution: {source_counts}")
    return samples


def process_numina():
    """Process NuminaMath-CoT for additional competition data."""
    print("Loading NuminaMath-CoT...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    competition_sources = {"amc_aime", "aops_forum", "olympiads", "synthetic_amc"}
    samples = []

    for row in ds:
        source = row.get("source", "")
        if source not in competition_sources:
            continue

        problem = row.get("problem", "")
        solution = row.get("solution", "")

        if not problem or not solution:
            continue

        # Try to extract integer answer from solution
        boxed = re.findall(r'\\boxed\{([^}]*)\}', solution)
        if not boxed:
            continue

        answer_str = boxed[-1].strip()
        try:
            val = int(answer_str)
            if not (0 <= val <= 999):
                continue
        except:
            continue

        # Format as think block
        cleaned = f"<think>\n{solution}\n</think>\n\nANSWER: {val}"

        # Skip very long solutions
        if len(cleaned) > 6000:
            continue

        sample = {
            "messages": [
                {"role": "user", "content": EVAL_PROMPT_PREFIX + problem},
                {"role": "assistant", "content": cleaned}
            ]
        }
        samples.append(sample)

    return samples


def main():
    openr1_samples = process_openr1()
    print(f"OpenR1 competition samples: {len(openr1_samples)}")

    numina_samples = process_numina()
    print(f"NuminaMath competition samples: {len(numina_samples)}")

    # Combine - prioritize OpenR1 (better reasoning traces)
    all_samples = openr1_samples + numina_samples
    print(f"Total: {len(all_samples)}")

    random.seed(42)
    random.shuffle(all_samples)

    val_size = 200
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    with open(OUTPUT_TRAIN, 'w') as f:
        for s in train_samples:
            f.write(json.dumps(s) + '\n')

    with open(OUTPUT_VAL, 'w') as f:
        for s in val_samples:
            f.write(json.dumps(s) + '\n')

    # Show samples
    print("\n=== Sample 0 ===")
    s = train_samples[0]
    print(f"User (first 200): {s['messages'][0]['content'][:200]}")
    print(f"Assistant (last 200): {s['messages'][1]['content'][-200:]}")

    # Stats
    total_len = sum(len(s['messages'][1]['content']) for s in train_samples)
    print(f"\nAverage solution length: {total_len / len(train_samples):.0f} chars")


if __name__ == '__main__':
    main()
