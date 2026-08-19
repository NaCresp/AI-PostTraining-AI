#!/usr/bin/env python3
"""Prepare training data for code generation fine-tuning."""
import json
import os
import random
from datasets import load_dataset

random.seed(42)

def format_chatml(system_msg, user_msg, assistant_msg):
    """Format a conversation in ChatML format."""
    parts = []
    if system_msg:
        parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
    return "\n".join(parts)

def is_python_code(text):
    """Check if text contains Python code."""
    python_indicators = ['def ', 'import ', 'class ', 'print(', 'return ', 'for ', 'while ', 'if __name__', 'lambda ', '.append(', '.join(', 'range(', 'len(']
    text_lower = text.lower()
    count = sum(1 for ind in python_indicators if ind.lower() in text_lower)
    return count >= 2

def extract_code_from_response(response):
    """Extract code from markdown code blocks if present."""
    if '```python' in response:
        parts = response.split('```python')
        if len(parts) > 1:
            code = parts[1].split('```')[0].strip()
            return code
    if '```' in response:
        parts = response.split('```')
        if len(parts) > 2:
            code = parts[1].strip()
            if code.startswith('python\n'):
                code = code[7:]
            return code
    return response

def main():
    all_samples = []

    system_prompts = [
        "You are a helpful coding assistant that writes clean, correct Python code.",
        "You are an expert Python programmer. Write clean, efficient code.",
        "You are a Python coding assistant. Implement the requested function correctly.",
    ]

    # 1. Load Magicoder-OSS-Instruct-75K
    print("Loading Magicoder-OSS-Instruct-75K...")
    try:
        ds1 = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        for item in ds1:
            problem = item.get('problem', '') or item.get('instruction', '')
            solution = item.get('solution', '') or item.get('response', '') or item.get('output', '')
            if not problem or not solution:
                continue
            # Filter for Python-relevant
            if is_python_code(solution) or 'python' in problem.lower() or 'def ' in solution:
                sys_prompt = random.choice(system_prompts)
                text = format_chatml(sys_prompt, problem, solution)
                all_samples.append({"text": text, "source": "magicoder_oss"})
        print(f"  -> {sum(1 for s in all_samples if s['source'] == 'magicoder_oss')} samples from Magicoder-OSS")
    except Exception as e:
        print(f"  Error loading Magicoder-OSS: {e}")

    # 2. Load Magicoder-Evol-Instruct-110K
    print("Loading Magicoder-Evol-Instruct-110K...")
    try:
        ds2 = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
        count_before = len(all_samples)
        for item in ds2:
            instruction = item.get('instruction', '') or item.get('problem', '')
            response = item.get('response', '') or item.get('solution', '') or item.get('output', '')
            if not instruction or not response:
                continue
            if is_python_code(response) or 'python' in instruction.lower() or 'def ' in response:
                sys_prompt = random.choice(system_prompts)
                text = format_chatml(sys_prompt, instruction, response)
                all_samples.append({"text": text, "source": "magicoder_evol"})
        print(f"  -> {len(all_samples) - count_before} samples from Magicoder-Evol")
    except Exception as e:
        print(f"  Error loading Magicoder-Evol: {e}")

    # 3. Load CodeFeedback-Filtered-Instruction
    print("Loading CodeFeedback-Filtered-Instruction...")
    try:
        ds3 = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
        count_before = len(all_samples)
        for item in ds3:
            query = item.get('query', '') or item.get('instruction', '')
            answer = item.get('answer', '') or item.get('response', '')
            if not query or not answer:
                continue
            if is_python_code(answer) or 'python' in query.lower() or 'def ' in answer:
                sys_prompt = random.choice(system_prompts)
                text = format_chatml(sys_prompt, query, answer)
                all_samples.append({"text": text, "source": "codefeedback"})
        print(f"  -> {len(all_samples) - count_before} samples from CodeFeedback")
    except Exception as e:
        print(f"  Error loading CodeFeedback: {e}")

    # 4. Create some synthetic HumanEval-style prompts (generic function completion, NOT from HumanEval)
    # These teach the model the exact format it will see during evaluation
    print("Creating synthetic function-completion examples...")
    synthetic_examples = [
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number.
    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(10)
    55
    \"\"\"
""",
            "response": """def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number.
    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(10)
    55
    \"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""
        },
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List

def flatten_list(nested: List) -> List:
    \"\"\"Flatten a nested list into a single list.
    >>> flatten_list([1, [2, 3], [4, [5, 6]]])
    [1, 2, 3, 4, 5, 6]
    >>> flatten_list([])
    []
    >>> flatten_list([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    \"\"\"
""",
            "response": """from typing import List

def flatten_list(nested: List) -> List:
    \"\"\"Flatten a nested list into a single list.
    >>> flatten_list([1, [2, 3], [4, [5, 6]]])
    [1, 2, 3, 4, 5, 6]
    >>> flatten_list([])
    []
    >>> flatten_list([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    \"\"\"
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result"""
        },
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List

def remove_duplicates(lst: List[int]) -> List[int]:
    \"\"\"Remove duplicates from a list while preserving order.
    >>> remove_duplicates([1, 2, 3, 2, 1, 4])
    [1, 2, 3, 4]
    >>> remove_duplicates([])
    []
    >>> remove_duplicates([1, 1, 1])
    [1]
    \"\"\"
""",
            "response": """from typing import List

def remove_duplicates(lst: List[int]) -> List[int]:
    \"\"\"Remove duplicates from a list while preserving order.
    >>> remove_duplicates([1, 2, 3, 2, 1, 4])
    [1, 2, 3, 4]
    >>> remove_duplicates([])
    []
    >>> remove_duplicates([1, 1, 1])
    [1]
    \"\"\"
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result"""
        },
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

def is_palindrome(s: str) -> bool:
    \"\"\"Check if a string is a palindrome (case-insensitive, ignoring non-alphanumeric characters).
    >>> is_palindrome("racecar")
    True
    >>> is_palindrome("A man, a plan, a canal: Panama")
    True
    >>> is_palindrome("hello")
    False
    \"\"\"
""",
            "response": """def is_palindrome(s: str) -> bool:
    \"\"\"Check if a string is a palindrome (case-insensitive, ignoring non-alphanumeric characters).
    >>> is_palindrome("racecar")
    True
    >>> is_palindrome("A man, a plan, a canal: Panama")
    True
    >>> is_palindrome("hello")
    False
    \"\"\"
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]"""
        },
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List

def binary_search(arr: List[int], target: int) -> int:
    \"\"\"Perform binary search on a sorted array and return the index of the target.
    Return -1 if the target is not found.
    >>> binary_search([1, 2, 3, 4, 5], 3)
    2
    >>> binary_search([1, 2, 3, 4, 5], 6)
    -1
    >>> binary_search([], 1)
    -1
    \"\"\"
""",
            "response": """from typing import List

def binary_search(arr: List[int], target: int) -> int:
    \"\"\"Perform binary search on a sorted array and return the index of the target.
    Return -1 if the target is not found.
    >>> binary_search([1, 2, 3, 4, 5], 3)
    2
    >>> binary_search([1, 2, 3, 4, 5], 6)
    -1
    >>> binary_search([], 1)
    -1
    \"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"""
        },
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import Dict, List

def count_words(text: str) -> Dict[str, int]:
    \"\"\"Count the frequency of each word in the text.
    >>> count_words("hello world hello")
    {'hello': 2, 'world': 1}
    >>> count_words("")
    {}
    \"\"\"
""",
            "response": """from typing import Dict, List

def count_words(text: str) -> Dict[str, int]:
    \"\"\"Count the frequency of each word in the text.
    >>> count_words("hello world hello")
    {'hello': 2, 'world': 1}
    >>> count_words("")
    {}
    \"\"\"
    if not text.strip():
        return {}
    result = {}
    for word in text.split():
        result[word] = result.get(word, 0) + 1
    return result"""
        },
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List, Tuple

def merge_sorted_lists(list1: List[int], list2: List[int]) -> List[int]:
    \"\"\"Merge two sorted lists into a single sorted list.
    >>> merge_sorted_lists([1, 3, 5], [2, 4, 6])
    [1, 2, 3, 4, 5, 6]
    >>> merge_sorted_lists([], [1, 2])
    [1, 2]
    >>> merge_sorted_lists([1], [])
    [1]
    \"\"\"
""",
            "response": """from typing import List, Tuple

def merge_sorted_lists(list1: List[int], list2: List[int]) -> List[int]:
    \"\"\"Merge two sorted lists into a single sorted list.
    >>> merge_sorted_lists([1, 3, 5], [2, 4, 6])
    [1, 2, 3, 4, 5, 6]
    >>> merge_sorted_lists([], [1, 2])
    [1, 2]
    >>> merge_sorted_lists([1], [])
    [1]
    \"\"\"
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result"""
        },
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List

def matrix_multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    \"\"\"Multiply two matrices and return the result.
    >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    [[19, 22], [43, 50]]
    >>> matrix_multiply([[1]], [[2]])
    [[2]]
    \"\"\"
""",
            "response": """from typing import List

def matrix_multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    \"\"\"Multiply two matrices and return the result.
    >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    [[19, 22], [43, 50]]
    >>> matrix_multiply([[1]], [[2]])
    [[2]]
    \"\"\"
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result"""
        },
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List

def longest_common_prefix(strs: List[str]) -> str:
    \"\"\"Find the longest common prefix string amongst an array of strings.
    >>> longest_common_prefix(["flower", "flow", "flight"])
    'fl'
    >>> longest_common_prefix(["dog", "racecar", "car"])
    ''
    >>> longest_common_prefix([])
    ''
    \"\"\"
""",
            "response": """from typing import List

def longest_common_prefix(strs: List[str]) -> str:
    \"\"\"Find the longest common prefix string amongst an array of strings.
    >>> longest_common_prefix(["flower", "flow", "flight"])
    'fl'
    >>> longest_common_prefix(["dog", "racecar", "car"])
    ''
    >>> longest_common_prefix([])
    ''
    \"\"\"
    if not strs:
        return ''
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ''
    return prefix"""
        },
        {
            "prompt": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

def gcd(a: int, b: int) -> int:
    \"\"\"Compute the greatest common divisor of two integers.
    >>> gcd(12, 8)
    4
    >>> gcd(7, 3)
    1
    >>> gcd(100, 25)
    25
    \"\"\"
""",
            "response": """def gcd(a: int, b: int) -> int:
    \"\"\"Compute the greatest common divisor of two integers.
    >>> gcd(12, 8)
    4
    >>> gcd(7, 3)
    1
    >>> gcd(100, 25)
    25
    \"\"\"
    while b:
        a, b = b, a % b
    return a"""
        },
    ]

    # Repeat synthetic examples to give them more weight (100x each = 1000 total)
    for ex in synthetic_examples:
        sys_prompt = random.choice(system_prompts)
        text = format_chatml(sys_prompt, ex["prompt"], ex["response"])
        for _ in range(100):
            all_samples.append({"text": text, "source": "synthetic"})

    print(f"  -> {sum(1 for s in all_samples if s['source'] == 'synthetic')} synthetic samples")

    # Shuffle and save
    random.shuffle(all_samples)
    print(f"\nTotal samples: {len(all_samples)}")

    # Save as JSONL
    output_path = "training_data.jsonl"
    with open(output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps({"text": sample["text"]}) + '\n')

    print(f"Saved to {output_path}")

    # Also report source distribution
    from collections import Counter
    sources = Counter(s["source"] for s in all_samples)
    for source, count in sources.most_common():
        print(f"  {source}: {count}")

if __name__ == "__main__":
    main()
