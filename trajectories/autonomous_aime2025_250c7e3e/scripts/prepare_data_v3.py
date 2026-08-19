#!/usr/bin/env python3
"""
Prepare v3 training data with format matching the AIME evaluation exactly:
- No system message
- User message includes instructions + problem (matching eval format)
- Assistant response: <think>...\n</think>\n\nANSWER: N
- Only clean integer answers 0-999
"""
import json
import re
import random

TRAIN_FOCUSED = "/home/ben/task/train_data/train_focused.jsonl"
OUTPUT_TRAIN = "/home/ben/task/train_data/train_v3.jsonl"
OUTPUT_VAL = "/home/ben/task/train_data/val_v3.jsonl"

EVAL_PROMPT_PREFIX = """Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

"""

def clean_answer(answer_str):
    """Extract clean integer from answer string."""
    answer_str = answer_str.strip()
    # Remove LaTeX formatting
    answer_str = re.sub(r'\\boxed\{([^}]*)\}', r'\1', answer_str)
    answer_str = re.sub(r'\$([^$]*)\$', r'\1', answer_str)
    answer_str = answer_str.strip()
    try:
        val = int(answer_str)
        if 0 <= val <= 999:
            return val
        return None
    except ValueError:
        return None

def clean_assistant_response(content, clean_answer_int):
    """Ensure proper <think>...</think>\n\nANSWER: N format."""
    # Extract the reasoning part (inside <think>)
    if '<think>' in content and '</think>' in content:
        think_start = content.index('<think>') + len('<think>')
        think_end = content.index('</think>')
        reasoning = content[think_start:think_end].strip()
    elif '<think>' in content:
        # No closing tag - extract everything between <think> and ANSWER
        think_start = content.index('<think>') + len('<think>')
        answer_match = re.search(r'ANSWER:', content)
        if answer_match:
            reasoning = content[think_start:answer_match.start()].strip()
        else:
            reasoning = content[think_start:].strip()
    else:
        # No think tags at all - use everything before ANSWER as reasoning
        answer_match = re.search(r'ANSWER:', content)
        if answer_match:
            reasoning = content[:answer_match.start()].strip()
        else:
            reasoning = content.strip()

    # Remove any trailing ANSWER lines from reasoning
    reasoning = re.sub(r'\nANSWER:.*$', '', reasoning, flags=re.MULTILINE).strip()

    # Remove \boxed from the very end if present
    # Build clean response
    return f"<think>\n{reasoning}\n</think>\n\nANSWER: {clean_answer_int}"

def process_data():
    samples = []

    with open(TRAIN_FOCUSED) as f:
        for line in f:
            data = json.loads(line)
            msgs = data['messages']

            # Extract problem from user message
            user_msg = None
            assistant_msg = None
            for msg in msgs:
                if msg['role'] == 'user':
                    user_msg = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_msg = msg['content']

            if not user_msg or not assistant_msg:
                continue

            # Extract and validate answer
            answer_match = re.search(r'ANSWER:\s*(.+?)$', assistant_msg, re.MULTILINE)
            if not answer_match:
                continue

            clean_ans = clean_answer(answer_match.group(1))
            if clean_ans is None:
                continue

            # Clean up assistant response
            cleaned_response = clean_assistant_response(assistant_msg, clean_ans)

            # Skip if reasoning is too short (likely garbage)
            reasoning_part = cleaned_response.split('</think>')[0]
            if len(reasoning_part) < 100:
                continue

            # Create formatted sample (NO system message, matching eval format)
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": EVAL_PROMPT_PREFIX + user_msg
                    },
                    {
                        "role": "assistant",
                        "content": cleaned_response
                    }
                ]
            }
            samples.append(sample)

    print(f"Total clean samples: {len(samples)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)

    val_size = 200
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Write output
    with open(OUTPUT_TRAIN, 'w') as f:
        for s in train_samples:
            f.write(json.dumps(s) + '\n')

    with open(OUTPUT_VAL, 'w') as f:
        for s in val_samples:
            f.write(json.dumps(s) + '\n')

    # Show a sample
    print("\n=== Sample 0 ===")
    s = train_samples[0]
    print(f"User (first 300): {s['messages'][0]['content'][:300]}")
    print(f"Assistant (first 300): {s['messages'][1]['content'][:300]}")
    print(f"Assistant (last 200): {s['messages'][1]['content'][-200:]}")

if __name__ == '__main__':
    process_data()
