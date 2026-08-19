#!/usr/bin/env python3
"""
Create a high-quality focused dataset for AIME training.
Prioritize:
1. OpenR1 data (has <think> reasoning traces from strong models)
2. Higher difficulty NuminaMath problems (olympiads, amc_aime)
3. Problems with integer answers (AIME format)
"""
import json
import re
import random

INPUT_PATH = "/home/ben/task/train_data/math_train.jsonl"
OUTPUT_DIR = "/home/ben/task/train_data"

def has_integer_answer(sample):
    """Check if the answer is an integer (AIME format)."""
    content = sample['messages'][-1]['content']
    answer_match = re.search(r'ANSWER:\s*(.+)', content)
    if answer_match:
        ans = answer_match.group(1).strip()
        try:
            val = float(ans.replace(',', ''))
            return val == int(val)
        except:
            return False
    return False

def get_answer_value(sample):
    """Extract the numeric answer."""
    content = sample['messages'][-1]['content']
    answer_match = re.search(r'ANSWER:\s*(.+)', content)
    if answer_match:
        ans = answer_match.group(1).strip()
        try:
            return int(float(ans.replace(',', '')))
        except:
            return None
    return None

def main():
    with open(INPUT_PATH) as f:
        all_samples = [json.loads(line) for line in f]

    print(f"Total samples: {len(all_samples)}")

    # Separate by having integer answers (AIME gives integers 0-999)
    integer_answer_samples = [s for s in all_samples if has_integer_answer(s)]
    print(f"Samples with integer answers: {len(integer_answer_samples)}")

    # For training, use a mix:
    # - All integer-answer samples (most AIME-relevant)
    # - Some non-integer to maintain general math ability
    non_integer = [s for s in all_samples if not has_integer_answer(s)]

    # Priority: Use all integer-answer samples + subset of non-integer
    # But cap total at ~50k for reasonable training time
    random.seed(42)

    # Take all integer answer samples
    train_samples = list(integer_answer_samples)
    print(f"Integer answer samples: {len(train_samples)}")

    # Add some non-integer samples for diversity (up to 15k)
    if len(non_integer) > 15000:
        random.shuffle(non_integer)
        non_integer_subset = non_integer[:15000]
    else:
        non_integer_subset = non_integer

    train_samples.extend(non_integer_subset)
    print(f"Total after adding non-integer: {len(train_samples)}")

    # If still too many, subsample to ~50k
    if len(train_samples) > 50000:
        random.shuffle(train_samples)
        train_samples = train_samples[:50000]
        print(f"Subsampled to: {len(train_samples)}")

    random.shuffle(train_samples)

    # Split val
    val_size = 200
    val_samples = train_samples[:val_size]
    train_final = train_samples[val_size:]

    # Save
    with open(f"{OUTPUT_DIR}/train_focused.jsonl", 'w') as f:
        for s in train_final:
            f.write(json.dumps(s) + '\n')

    with open(f"{OUTPUT_DIR}/val_focused.jsonl", 'w') as f:
        for s in val_samples:
            f.write(json.dumps(s) + '\n')

    print(f"\nFocused train: {len(train_final)} samples")
    print(f"Focused val: {len(val_samples)} samples")

    # Stats on answer distribution
    answers = [get_answer_value(s) for s in train_final]
    answers = [a for a in answers if a is not None]
    aime_range = [a for a in answers if 0 <= a <= 999]
    print(f"\nAnswers in AIME range (0-999): {len(aime_range)}/{len(answers)}")

if __name__ == "__main__":
    main()
