#!/usr/bin/env python3
"""
Generate high-quality training data for code completion model evaluated on HumanEval.
Critical: Outputs ONLY function bodies (indented code) in assistant responses.
"""

import json
import ast
import re
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm


def extract_functions_from_code(code: str) -> List[Tuple[str, str, str]]:
    """
    Extract functions from code using AST parsing.
    Returns list of (function_name, signature_with_docstring, body_only) tuples.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            try:
                # Get the full source of the function
                func_lines = code.splitlines()

                # Get function signature
                func_name = node.name
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)

                # Handle return annotation
                return_annotation = ""
                if node.returns:
                    return_annotation = f" -> {ast.unparse(node.returns)}"

                signature = f"def {func_name}({', '.join(args)}){return_annotation}:"

                # Extract docstring if present
                docstring = ast.get_docstring(node)
                docstring_lines = 0
                if docstring:
                    # Count docstring lines in original code
                    if node.body and isinstance(node.body[0], ast.Expr):
                        if isinstance(node.body[0].value, ast.Constant):
                            # Determine quote style
                            start_line = node.body[0].lineno - 1
                            end_line = node.body[0].end_lineno
                            docstring_lines = end_line - start_line

                # Get function body (excluding docstring)
                func_start_line = node.lineno - 1  # 0-indexed
                func_end_line = node.end_lineno

                # Extract body lines
                body_start = func_start_line + 1  # Skip def line
                if docstring:
                    body_start += docstring_lines  # Skip docstring lines

                body_lines = func_lines[body_start:func_end_line]

                # Filter out empty body
                body_code = '\n'.join(body_lines).strip()
                if not body_code or body_code == 'pass':
                    continue

                # Count actual code lines (non-empty, non-comment)
                code_lines = [
                    line for line in body_lines
                    if line.strip() and not line.strip().startswith('#')
                ]
                if len(code_lines) < 2:
                    continue

                # Build signature with docstring for user prompt
                if docstring:
                    # Use triple quotes
                    signature_with_doc = f'{signature}\n    """{docstring}"""'
                else:
                    signature_with_doc = signature

                # Body should maintain indentation (4 spaces)
                # Ensure consistent indentation
                body_formatted = '\n'.join(body_lines)

                functions.append((func_name, signature_with_doc, body_formatted))

            except Exception as e:
                continue

    return functions


def generate_simple_docstring(func_name: str, code: str) -> str:
    """Generate a simple docstring based on function name and code."""
    # Convert snake_case to words
    words = func_name.replace('_', ' ').title()

    # Check for return statement
    has_return = 'return' in code

    if has_return:
        return f"{words}"
    else:
        return f"Perform {words.lower()}"


def create_training_sample(signature_with_doc: str, body: str) -> Dict[str, str]:
    """
    Create a training sample in ChatML format.
    User message: signature + docstring
    Assistant response: <think></think> + body only (indented code)
    """
    user_message = (
        "Read the following function signature and docstring, and fully implement "
        "the function described. Your response should only contain the code for this function.\n\n"
        f"{signature_with_doc}"
    )

    # Assistant response: think tags + body only (NO function signature, NO markdown)
    # The body should already have proper indentation
    assistant_message = f"<think>\n\n</think>\n\n{body}"

    # Create ChatML format
    chatml = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_message}<|im_end|>\n"
    )

    return {"text": chatml}


def is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def process_mbpp() -> List[Dict[str, str]]:
    """Process MBPP dataset."""
    print("Loading MBPP dataset...")
    samples = []

    try:
        # Load all splits
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", trust_remote_code=True)

        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                continue

            split = dataset[split_name]
            print(f"Processing MBPP {split_name} split: {len(split)} samples")

            for item in tqdm(split, desc=f"MBPP {split_name}"):
                code = item.get('code', '')

                if not code:
                    continue

                # Extract functions
                functions = extract_functions_from_code(code)

                for func_name, sig_with_doc, body in functions:
                    # Create full function for validation
                    full_func = f"{sig_with_doc}\n{body}"

                    if not is_valid_python(full_func):
                        continue

                    sample = create_training_sample(sig_with_doc, body)
                    samples.append(sample)

        print(f"MBPP: Generated {len(samples)} samples")
    except Exception as e:
        print(f"Error processing MBPP: {e}")

    return samples


def process_magicoder_oss() -> List[Dict[str, str]]:
    """Process Magicoder-OSS-Instruct dataset."""
    print("Loading Magicoder-OSS-Instruct...")
    samples = []

    try:
        dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", trust_remote_code=True)

        if 'train' in dataset:
            split = dataset['train']
            print(f"Processing Magicoder-OSS: {len(split)} samples")

            for item in tqdm(split, desc="Magicoder-OSS"):
                # Extract code from response
                response = item.get('response', '')

                # Try to extract code blocks
                code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)

                if code_blocks:
                    for code in code_blocks:
                        functions = extract_functions_from_code(code)

                        for func_name, sig_with_doc, body in functions:
                            full_func = f"{sig_with_doc}\n{body}"

                            if not is_valid_python(full_func):
                                continue

                            sample = create_training_sample(sig_with_doc, body)
                            samples.append(sample)
                else:
                    # Try treating whole response as code
                    functions = extract_functions_from_code(response)

                    for func_name, sig_with_doc, body in functions:
                        full_func = f"{sig_with_doc}\n{body}"

                        if not is_valid_python(full_func):
                            continue

                        sample = create_training_sample(sig_with_doc, body)
                        samples.append(sample)

        print(f"Magicoder-OSS: Generated {len(samples)} samples")
    except Exception as e:
        print(f"Error processing Magicoder-OSS: {e}")

    return samples


def process_magicoder_evol() -> List[Dict[str, str]]:
    """Process Magicoder-Evol-Instruct dataset."""
    print("Loading Magicoder-Evol-Instruct...")
    samples = []

    try:
        dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", trust_remote_code=True)

        if 'train' in dataset:
            split = dataset['train']
            print(f"Processing Magicoder-Evol: {len(split)} samples")

            for item in tqdm(split, desc="Magicoder-Evol"):
                response = item.get('response', '')

                # Try to extract code blocks
                code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)

                if code_blocks:
                    for code in code_blocks:
                        functions = extract_functions_from_code(code)

                        for func_name, sig_with_doc, body in functions:
                            full_func = f"{sig_with_doc}\n{body}"

                            if not is_valid_python(full_func):
                                continue

                            sample = create_training_sample(sig_with_doc, body)
                            samples.append(sample)
                else:
                    # Try treating whole response as code
                    functions = extract_functions_from_code(response)

                    for func_name, sig_with_doc, body in functions:
                        full_func = f"{sig_with_doc}\n{body}"

                        if not is_valid_python(full_func):
                            continue

                        sample = create_training_sample(sig_with_doc, body)
                        samples.append(sample)

        print(f"Magicoder-Evol: Generated {len(samples)} samples")
    except Exception as e:
        print(f"Error processing Magicoder-Evol: {e}")

    return samples


def main():
    """Main function to generate training data."""
    print("="*80)
    print("Generating Training Data v7 for HumanEval Code Completion")
    print("="*80)

    all_samples = []

    # Process all datasets
    all_samples.extend(process_mbpp())
    all_samples.extend(process_magicoder_oss())
    all_samples.extend(process_magicoder_evol())

    print(f"\nTotal samples before deduplication: {len(all_samples)}")

    # Deduplicate by text field
    unique_samples = []
    seen_texts = set()

    for sample in tqdm(all_samples, desc="Deduplicating"):
        text = sample['text']
        if text not in seen_texts:
            seen_texts.add(text)
            unique_samples.append(sample)

    print(f"Total samples after deduplication: {len(unique_samples)}")

    # Write to JSONL
    output_file = "/home/ben/task/training_data_v7.jsonl"
    print(f"\nWriting to {output_file}...")

    with open(output_file, 'w') as f:
        for sample in tqdm(unique_samples, desc="Writing"):
            f.write(json.dumps(sample) + '\n')

    print(f"\n{'='*80}")
    print(f"SUCCESS! Generated {len(unique_samples)} training samples")
    print(f"Output: {output_file}")
    print(f"{'='*80}")

    # Print a sample for verification
    if unique_samples:
        print("\nSample output (first example):")
        print("-" * 80)
        print(unique_samples[0]['text'][:800] + "..." if len(unique_samples[0]['text']) > 800 else unique_samples[0]['text'])
        print("-" * 80)


if __name__ == "__main__":
    main()
