#!/usr/bin/env python3
"""
SFT v5 data preparation - focus on LONGER reasoning chains from high-quality sources.
Key changes from v4:
- Prefer solutions with longer reasoning (>3000 chars) to teach extended thinking
- Use OpenR1-Math-220k more aggressively (higher quality R1 reasoning traces)
- Include ALL competition-adjacent sources, not just strict competition
- Allow slightly longer solutions (up to 12000 chars) for max_seq_length=8192
- Focus on integer-answer problems matching AIME format
"""
import json
import random
import re
from datasets import load_dataset

EVAL_PROMPT = """Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem."""

def extract_answer(text):
    """Extract clean integer answer from solution text."""
    # Try to find \boxed{} answers
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        ans = boxed[-1].strip()
        # Clean up
        ans = ans.replace('\\', '').replace(' ', '').replace(',', '')
        try:
            val = int(float(ans))
            if 0 <= val <= 999:
                return str(val)
        except:
            pass

    # Try ANSWER: pattern
    answer_match = re.findall(r'ANSWER:\s*(\d+)', text)
    if answer_match:
        val = int(answer_match[-1])
        if 0 <= val <= 999:
            return str(val)

    return None

def format_solution_with_think(solution, answer):
    """Format solution in <think>...</think>\n\nANSWER: N format."""
    # Clean up the solution
    solution = solution.strip()

    # Remove any existing ANSWER: lines
    lines = solution.split('\n')
    clean_lines = [l for l in lines if not re.match(r'^\s*ANSWER:', l)]
    solution = '\n'.join(clean_lines).strip()

    # Remove trailing boxed answer if present
    solution = re.sub(r'\s*The\s+(?:final\s+)?answer\s+is\s*\\boxed\{[^}]*\}\s*\.?\s*$', '', solution)

    return f"<think>\n{solution}\n</think>\n\nANSWER: {answer}"

def process_openr1():
    """Load and process OpenR1-Math-220k - prefer longer reasoning chains."""
    print("Loading OpenR1-Math-220k...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")

    # Competition and math-focused sources
    GOOD_SOURCES = {
        "amc_aime", "aops_forum", "olympiads", "number_theory",
        "inequalities", "olympiads_ref", "cn_contest",
        "math", "algebra", "geometry", "combinatorics", "counting_and_probability",
    }

    samples = []
    for row in ds:
        source = row.get("source", "")

        # Check if verified correct
        if not row.get("correctness_math_verify", False):
            continue

        problem = row.get("problem", "")
        solution = row.get("solution", "")

        if not problem or not solution:
            continue

        # Extract answer
        answer = extract_answer(solution)
        if answer is None:
            continue

        sol_len = len(solution)

        # Accept solutions between 500 and 12000 chars
        if sol_len < 500 or sol_len > 12000:
            continue

        # Check source - be more permissive
        source_match = any(s in source.lower() for s in GOOD_SOURCES)

        # For non-competition sources, require longer solutions (better reasoning)
        if not source_match and sol_len < 2000:
            continue

        formatted = format_solution_with_think(solution, answer)

        samples.append({
            "messages": [
                {"role": "user", "content": f"{EVAL_PROMPT}\n\n{problem}"},
                {"role": "assistant", "content": formatted}
            ],
            "source": f"openr1_{source}",
            "solution_length": sol_len,
        })

    print(f"OpenR1 samples: {len(samples)}")
    return samples

def process_numina():
    """Load NuminaMath-CoT competition problems."""
    print("Loading NuminaMath-CoT...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    COMPETITION_SOURCES = {"amc_aime", "aops_forum", "olympiads", "synthetic_amc"}

    samples = []
    for row in ds:
        source = row.get("source", "")
        if source not in COMPETITION_SOURCES:
            continue

        problem = row.get("problem", "")
        solution = row.get("solution", "")

        if not problem or not solution:
            continue

        answer = extract_answer(solution)
        if answer is None:
            continue

        sol_len = len(solution)
        if sol_len < 300 or sol_len > 12000:
            continue

        formatted = format_solution_with_think(solution, answer)

        samples.append({
            "messages": [
                {"role": "user", "content": f"{EVAL_PROMPT}\n\n{problem}"},
                {"role": "assistant", "content": formatted}
            ],
            "source": f"numina_{source}",
            "solution_length": sol_len,
        })

    print(f"NuminaMath samples: {len(samples)}")
    return samples

def main():
    openr1_samples = process_openr1()
    numina_samples = process_numina()

    all_samples = openr1_samples + numina_samples
    random.seed(42)
    random.shuffle(all_samples)

    # Compute stats
    lengths = [s["solution_length"] for s in all_samples]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    # Prioritize longer solutions - sort by length and take top samples
    # But also keep a good mix
    long_samples = [s for s in all_samples if s["solution_length"] >= 3000]
    medium_samples = [s for s in all_samples if 1500 <= s["solution_length"] < 3000]
    short_samples = [s for s in all_samples if s["solution_length"] < 1500]

    print(f"\nBy length category:")
    print(f"  Long (>=3000): {len(long_samples)}")
    print(f"  Medium (1500-3000): {len(medium_samples)}")
    print(f"  Short (<1500): {len(short_samples)}")

    # Take all long and medium, plus some short for balance
    # Cap short samples to not dominate
    max_short = min(len(short_samples), len(long_samples) + len(medium_samples))
    selected = long_samples + medium_samples + short_samples[:max_short]
    random.shuffle(selected)

    print(f"\nTotal selected: {len(selected)}")

    # Split val
    val_size = 200
    val = selected[:val_size]
    train = selected[val_size:]

    # Compute final stats
    train_lengths = [s["solution_length"] for s in train]
    print(f"Train: {len(train)}, Val: {len(val)}")
    print(f"Avg solution length: {sum(train_lengths)/len(train_lengths):.0f} chars")

    # Source breakdown
    sources = {}
    for s in train:
        src = s["source"]
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"  {src}: {count}")

    # Save
    for split, data, path in [
        ("train", train, "/home/ben/task/train_data/train_v5.jsonl"),
        ("val", val, "/home/ben/task/train_data/val_v5.jsonl")
    ]:
        with open(path, 'w') as f:
            for s in data:
                row = {"messages": s["messages"]}
                f.write(json.dumps(row) + '\n')
        print(f"Saved {split}: {len(data)} samples to {path}")

if __name__ == "__main__":
    main()
