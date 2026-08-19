#!/usr/bin/env python3
"""
Prepare training data for GSM8K fine-tuning.
Uses GSM8K train set + MetaMathQA + other math datasets.
Formats everything to match the evaluation chat template.
"""
import json
import os
import random
from datasets import load_dataset

random.seed(42)

# The evaluation uses this exact prompt template
MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

def format_gsm8k_answer(reasoning, answer):
    """Format answer in the way the eval expects."""
    # Clean up reasoning
    reasoning = reasoning.strip()
    # Make sure answer is clean
    answer = answer.strip().replace(",", "")
    return f"{reasoning}\n\nANSWER: {answer}"


def process_gsm8k_train():
    """Process GSM8K training set."""
    print("Loading GSM8K train...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    examples = []
    for item in ds:
        question = item["question"]
        answer_text = item["answer"]

        # Parse the GSM8K format: reasoning #### answer
        parts = answer_text.split("####")
        final_answer = parts[-1].strip().replace(",", "")
        reasoning = "####".join(parts[:-1]).strip()

        user_msg = MATH_PROMPT_TEMPLATE.format(question=question)
        assistant_msg = format_gsm8k_answer(reasoning, final_answer)

        examples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ],
            "source": "gsm8k_train"
        })

    print(f"GSM8K train: {len(examples)} examples")
    return examples


def process_metamathqa():
    """Process MetaMathQA dataset - very high quality math augmentation."""
    print("Loading MetaMathQA...")
    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train")
    except Exception as e:
        print(f"Failed to load MetaMathQA: {e}")
        return []

    examples = []
    # Filter for GSM8K-related examples (they have "GSM" in the type field)
    gsm_examples = []
    other_examples = []

    for item in ds:
        q_type = item.get("type", "")
        original_question = item.get("original_question", "")
        query = item.get("query", "")
        response = item.get("response", "")

        if not query or not response:
            continue

        # Extract the answer from the response
        # MetaMathQA typically ends with "The answer is: X"
        answer = None
        if "The answer is:" in response:
            answer = response.split("The answer is:")[-1].strip().rstrip(".")
        elif "the answer is:" in response:
            answer = response.split("the answer is:")[-1].strip().rstrip(".")
        elif "\\boxed{" in response:
            import re
            boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
            if boxed:
                answer = boxed[-1]

        if answer is None:
            continue

        # Clean the answer
        answer = answer.strip().replace(",", "").replace("$", "")

        # Try to determine if numeric answer
        try:
            float(answer)
        except (ValueError, TypeError):
            # Skip non-numeric answers for GSM8K since eval uses numeric match
            if "GSM" in q_type:
                continue
            else:
                continue  # Skip non-numeric for all

        # Clean up the response to be just reasoning (remove the "The answer is" part)
        reasoning = response
        for end_phrase in ["The answer is:", "the answer is:", "Therefore, the answer is:", "So the answer is:", "Hence, the answer is:"]:
            if end_phrase in reasoning:
                reasoning = reasoning[:reasoning.rfind(end_phrase)].strip()
                break

        user_msg = MATH_PROMPT_TEMPLATE.format(question=query)
        assistant_msg = format_gsm8k_answer(reasoning, answer)

        entry = {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ],
            "source": f"metamathqa_{q_type}"
        }

        if "GSM" in q_type:
            gsm_examples.append(entry)
        else:
            other_examples.append(entry)

    # Take all GSM examples and a sample of others
    print(f"MetaMathQA GSM-related: {len(gsm_examples)}")
    print(f"MetaMathQA other: {len(other_examples)}")

    # Use all GSM examples and sample from others
    random.shuffle(other_examples)
    selected_other = other_examples[:min(30000, len(other_examples))]

    examples = gsm_examples + selected_other
    print(f"MetaMathQA total selected: {len(examples)}")
    return examples


def process_orca_math():
    """Process a subset of Orca-Math dataset."""
    print("Loading Orca-Math...")
    try:
        ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    except Exception as e:
        print(f"Failed to load Orca-Math: {e}")
        return []

    examples = []
    for item in ds:
        question = item.get("question", "")
        answer = item.get("answer", "")

        if not question or not answer:
            continue

        # Extract numeric answer from the response
        # Orca math typically has detailed solutions
        # Try to find the last number mentioned
        import re

        # Look for patterns like "the answer is X" or similar
        numeric_answer = None

        # Try common patterns
        for pattern in [
            r'(?:the answer is|= |equals )\s*\$?\s*([\d,]+\.?\d*)',
            r'(?:Therefore|So|Thus|Hence),?\s+.*?(\d[\d,]*\.?\d*)',
            r'####\s*([\d,]+\.?\d*)',
        ]:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            if matches:
                numeric_answer = matches[-1].replace(",", "")
                break

        if numeric_answer is None:
            # Try to get the last number
            numbers = re.findall(r'(?<![a-zA-Z])(\d+\.?\d*)(?![a-zA-Z])', answer)
            if numbers:
                numeric_answer = numbers[-1]
            else:
                continue

        user_msg = MATH_PROMPT_TEMPLATE.format(question=question)
        assistant_msg = format_gsm8k_answer(answer, numeric_answer)

        examples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ],
            "source": "orca_math"
        })

    random.shuffle(examples)
    examples = examples[:30000]  # Take a subset
    print(f"Orca-Math selected: {len(examples)}")
    return examples


def main():
    all_examples = []

    # GSM8K train is highest priority - closest to eval distribution
    gsm8k_examples = process_gsm8k_train()
    all_examples.extend(gsm8k_examples)

    # MetaMathQA - high quality augmentation including GSM8K variants
    metamath_examples = process_metamathqa()
    all_examples.extend(metamath_examples)

    # Orca-Math for additional math word problems
    orca_examples = process_orca_math()
    all_examples.extend(orca_examples)

    # Shuffle
    random.shuffle(all_examples)

    print(f"\nTotal training examples: {len(all_examples)}")

    # Save
    output_dir = "/home/ben/task/training_data"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "train.jsonl")
    with open(output_file, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved to {output_file}")

    # Also save just GSM8K data for potential focused training
    gsm8k_file = os.path.join(output_dir, "gsm8k_train.jsonl")
    with open(gsm8k_file, "w") as f:
        for ex in gsm8k_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved GSM8K-only to {gsm8k_file}")

    # Print some stats
    sources = {}
    for ex in all_examples:
        src = ex["source"].split("_")[0]
        sources[src] = sources.get(src, 0) + 1
    print("\nSource distribution:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
