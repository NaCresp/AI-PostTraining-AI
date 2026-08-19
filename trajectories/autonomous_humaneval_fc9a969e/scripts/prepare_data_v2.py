#!/usr/bin/env python3
"""Create a focused training dataset specifically for HumanEval-format code completion.
Uses MBPP (Mostly Basic Python Problems) and other function-completion style data."""
import json
import random
import re
from datasets import load_dataset

random.seed(42)

SYSTEM_PROMPT = "You are a helpful coding assistant that writes clean, correct Python code."

TASK_PREFIX = """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function."""

def format_chatml(system_msg, user_msg, assistant_msg):
    parts = []
    if system_msg:
        parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
    return "\n".join(parts)

def format_chatml_thinking(system_msg, user_msg, assistant_msg):
    """Format with the thinking tags the template generates."""
    parts = []
    if system_msg:
        parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_msg}<|im_end|>")
    return "\n".join(parts)

def clean_code(code):
    """Remove markdown code blocks and clean up."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code

def extract_function(code):
    """Extract function from code including signature, docstring, and body."""
    lines = code.split('\n')
    func_lines = []
    in_function = False
    indent_level = None

    for line in lines:
        if line.strip().startswith('def ') and not in_function:
            in_function = True
            indent_level = len(line) - len(line.lstrip())
            func_lines.append(line)
        elif in_function:
            if line.strip() == '':
                func_lines.append(line)
            elif len(line) - len(line.lstrip()) > indent_level:
                func_lines.append(line)
            else:
                break
    return '\n'.join(func_lines).rstrip()

def main():
    all_samples = []

    # 1. Load MBPP dataset - these are function completion problems similar to HumanEval
    print("Loading MBPP...")
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        for item in ds:
            prompt = item.get('prompt', '')
            code = item.get('code', '')
            if not prompt or not code:
                continue
            code = clean_code(code)
            # Create a function-completion style prompt
            user_msg = f"{TASK_PREFIX}\n\n{code.split(chr(10))[0]}"  # Just the first line (def ...)
            # But actually, let's format it like HumanEval - provide signature with docstring
            # Try to extract function signature and docstring
            lines = code.strip().split('\n')
            sig_lines = []
            body_start = 0
            in_docstring = False
            docstring_done = False
            for i, line in enumerate(lines):
                if not docstring_done:
                    sig_lines.append(line)
                    if '"""' in line or "'''" in line:
                        if in_docstring:
                            docstring_done = True
                            body_start = i + 1
                        else:
                            in_docstring = True
                            # Check if docstring starts and ends on same line
                            stripped = line.strip()
                            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                                docstring_done = True
                                body_start = i + 1
                else:
                    break

            if not docstring_done:
                # No docstring found, use full prompt as user message
                user_msg = f"{TASK_PREFIX}\n\n{prompt}\n\n{lines[0]}\n"
                assistant_msg = code
            else:
                sig = '\n'.join(sig_lines)
                user_msg = f"{TASK_PREFIX}\n\n{sig}\n"
                assistant_msg = code

            text = format_chatml(SYSTEM_PROMPT, user_msg, assistant_msg)
            all_samples.append({"text": text, "source": "mbpp"})
            # Also add thinking variant
            text2 = format_chatml_thinking(SYSTEM_PROMPT, user_msg, assistant_msg)
            all_samples.append({"text": text2, "source": "mbpp_think"})

        print(f"  -> {sum(1 for s in all_samples if 'mbpp' in s['source'])} samples from MBPP")
    except Exception as e:
        print(f"  Error loading MBPP: {e}")

    # 2. Load from Magicoder - but filter for Python function implementations
    print("Loading Magicoder-OSS-Instruct-75K (Python functions only)...")
    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        count = 0
        for item in ds:
            problem = item.get('problem', '') or item.get('instruction', '')
            solution = item.get('solution', '') or item.get('response', '') or item.get('output', '')
            if not problem or not solution:
                continue

            solution_clean = clean_code(solution)

            # Only include if it looks like a Python function
            if 'def ' not in solution_clean:
                continue
            if not any(kw in solution_clean for kw in ['return ', 'print(', 'yield ', 'raise ']):
                continue

            user_msg = problem
            assistant_msg = solution_clean

            text = format_chatml(SYSTEM_PROMPT, user_msg, assistant_msg)
            all_samples.append({"text": text, "source": "magicoder"})

            # Also add with thinking
            text2 = format_chatml_thinking(SYSTEM_PROMPT, user_msg, assistant_msg)
            all_samples.append({"text": text2, "source": "magicoder_think"})
            count += 1

        print(f"  -> {count * 2} samples from Magicoder (with thinking variants)")
    except Exception as e:
        print(f"  Error loading Magicoder: {e}")

    # 3. Load Magicoder-Evol-Instruct-110K
    print("Loading Magicoder-Evol-Instruct-110K (Python functions only)...")
    try:
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
        count = 0
        for item in ds:
            instruction = item.get('instruction', '')
            response = item.get('response', '')
            if not instruction or not response:
                continue

            response_clean = clean_code(response)
            if 'def ' not in response_clean:
                continue
            if not any(kw in response_clean for kw in ['return ', 'print(', 'yield ', 'raise ']):
                continue

            text = format_chatml(SYSTEM_PROMPT, instruction, response_clean)
            all_samples.append({"text": text, "source": "evol"})
            text2 = format_chatml_thinking(SYSTEM_PROMPT, instruction, response_clean)
            all_samples.append({"text": text2, "source": "evol_think"})
            count += 1

        print(f"  -> {count * 2} samples from Evol")
    except Exception as e:
        print(f"  Error loading Evol: {e}")

    # 4. Add some synthetic HumanEval-style examples with the exact format
    print("Creating synthetic examples in exact HumanEval format...")
    synthetic = create_synthetic_humaneval_format_examples()
    for ex in synthetic:
        # Each synthetic example is duplicated 50x for weight
        for _ in range(50):
            text = format_chatml(SYSTEM_PROMPT, ex["user"], ex["assistant"])
            all_samples.append({"text": text, "source": "synthetic"})
            text2 = format_chatml_thinking(SYSTEM_PROMPT, ex["user"], ex["assistant"])
            all_samples.append({"text": text2, "source": "synthetic_think"})

    print(f"  -> {sum(1 for s in all_samples if 'synthetic' in s['source'])} synthetic samples")

    random.shuffle(all_samples)
    print(f"\nTotal samples: {len(all_samples)}")

    output_path = "training_data_v2.jsonl"
    with open(output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps({"text": sample["text"]}) + '\n')

    print(f"Saved to {output_path}")

    from collections import Counter
    sources = Counter(s["source"] for s in all_samples)
    for source, count in sources.most_common():
        print(f"  {source}: {count}")


def create_synthetic_humaneval_format_examples():
    """Create examples that perfectly match HumanEval evaluation format."""
    examples = [
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number.
    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(10)
    55
    \"\"\"""",
            "assistant": """from typing import List


def fibonacci(n: int) -> int:
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
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def flatten(nested: List) -> List:
    \"\"\"Flatten a nested list into a single list.
    >>> flatten([1, [2, 3], [4, [5, 6]]])
    [1, 2, 3, 4, 5, 6]
    >>> flatten([])
    []
    \"\"\"""",
            "assistant": """from typing import List


def flatten(nested: List) -> List:
    \"\"\"Flatten a nested list into a single list.
    >>> flatten([1, [2, 3], [4, [5, 6]]])
    [1, 2, 3, 4, 5, 6]
    >>> flatten([])
    []
    \"\"\"
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result"""
        },
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def remove_duplicates(lst: List[int]) -> List[int]:
    \"\"\"Remove duplicates from a list while preserving order.
    >>> remove_duplicates([1, 2, 3, 2, 1, 4])
    [1, 2, 3, 4]
    >>> remove_duplicates([])
    []
    \"\"\"""",
            "assistant": """from typing import List


def remove_duplicates(lst: List[int]) -> List[int]:
    \"\"\"Remove duplicates from a list while preserving order.
    >>> remove_duplicates([1, 2, 3, 2, 1, 4])
    [1, 2, 3, 4]
    >>> remove_duplicates([])
    []
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
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.


def is_palindrome(s: str) -> bool:
    \"\"\"Check if a string is a palindrome.
    >>> is_palindrome("racecar")
    True
    >>> is_palindrome("hello")
    False
    \"\"\"""",
            "assistant": """def is_palindrome(s: str) -> bool:
    \"\"\"Check if a string is a palindrome.
    >>> is_palindrome("racecar")
    True
    >>> is_palindrome("hello")
    False
    \"\"\"
    return s == s[::-1]"""
        },
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def binary_search(arr: List[int], target: int) -> int:
    \"\"\"Perform binary search and return index of target. Return -1 if not found.
    >>> binary_search([1, 2, 3, 4, 5], 3)
    2
    >>> binary_search([1, 2, 3, 4, 5], 6)
    -1
    \"\"\"""",
            "assistant": """from typing import List


def binary_search(arr: List[int], target: int) -> int:
    \"\"\"Perform binary search and return index of target. Return -1 if not found.
    >>> binary_search([1, 2, 3, 4, 5], 3)
    2
    >>> binary_search([1, 2, 3, 4, 5], 6)
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
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def merge_sort(arr: List[int]) -> List[int]:
    \"\"\"Sort a list using merge sort algorithm.
    >>> merge_sort([3, 1, 4, 1, 5, 9])
    [1, 1, 3, 4, 5, 9]
    >>> merge_sort([])
    []
    \"\"\"""",
            "assistant": """from typing import List


def merge_sort(arr: List[int]) -> List[int]:
    \"\"\"Sort a list using merge sort algorithm.
    >>> merge_sort([3, 1, 4, 1, 5, 9])
    [1, 1, 3, 4, 5, 9]
    >>> merge_sort([])
    []
    \"\"\"
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result"""
        },
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import Dict


def count_words(text: str) -> Dict[str, int]:
    \"\"\"Count the frequency of each word in the text.
    >>> count_words("hello world hello")
    {'hello': 2, 'world': 1}
    >>> count_words("")
    {}
    \"\"\"""",
            "assistant": """from typing import Dict


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
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def matrix_transpose(matrix: List[List[int]]) -> List[List[int]]:
    \"\"\"Return the transpose of a matrix.
    >>> matrix_transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    \"\"\"""",
            "assistant": """from typing import List


def matrix_transpose(matrix: List[List[int]]) -> List[List[int]]:
    \"\"\"Return the transpose of a matrix.
    >>> matrix_transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    \"\"\"
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]"""
        },
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def longest_common_prefix(strs: List[str]) -> str:
    \"\"\"Find the longest common prefix string amongst an array of strings.
    >>> longest_common_prefix(["flower", "flow", "flight"])
    'fl'
    >>> longest_common_prefix(["dog", "racecar", "car"])
    ''
    \"\"\"""",
            "assistant": """from typing import List


def longest_common_prefix(strs: List[str]) -> str:
    \"\"\"Find the longest common prefix string amongst an array of strings.
    >>> longest_common_prefix(["flower", "flow", "flight"])
    'fl'
    >>> longest_common_prefix(["dog", "racecar", "car"])
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
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.


def gcd(a: int, b: int) -> int:
    \"\"\"Compute the greatest common divisor of two integers.
    >>> gcd(12, 8)
    4
    >>> gcd(7, 3)
    1
    \"\"\"""",
            "assistant": """def gcd(a: int, b: int) -> int:
    \"\"\"Compute the greatest common divisor of two integers.
    >>> gcd(12, 8)
    4
    >>> gcd(7, 3)
    1
    \"\"\"
    while b:
        a, b = b, a % b
    return a"""
        },
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List, Optional


def find_max(lst: List[int]) -> Optional[int]:
    \"\"\"Find the maximum element in a list. Return None if the list is empty.
    >>> find_max([1, 5, 3, 2])
    5
    >>> find_max([])
    None
    \"\"\"""",
            "assistant": """from typing import List, Optional


def find_max(lst: List[int]) -> Optional[int]:
    \"\"\"Find the maximum element in a list. Return None if the list is empty.
    >>> find_max([1, 5, 3, 2])
    5
    >>> find_max([])
    None
    \"\"\"
    if not lst:
        return None
    return max(lst)"""
        },
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def rotate_list(lst: List[int], k: int) -> List[int]:
    \"\"\"Rotate the list to the right by k positions.
    >>> rotate_list([1, 2, 3, 4, 5], 2)
    [4, 5, 1, 2, 3]
    >>> rotate_list([1, 2, 3], 0)
    [1, 2, 3]
    \"\"\"""",
            "assistant": """from typing import List


def rotate_list(lst: List[int], k: int) -> List[int]:
    \"\"\"Rotate the list to the right by k positions.
    >>> rotate_list([1, 2, 3, 4, 5], 2)
    [4, 5, 1, 2, 3]
    >>> rotate_list([1, 2, 3], 0)
    [1, 2, 3]
    \"\"\"
    if not lst or k == 0:
        return lst
    k = k % len(lst)
    return lst[-k:] + lst[:-k]"""
        },
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    \"\"\"Return indices of two numbers that add up to target.
    >>> two_sum([2, 7, 11, 15], 9)
    [0, 1]
    >>> two_sum([3, 2, 4], 6)
    [1, 2]
    \"\"\"""",
            "assistant": """from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    \"\"\"Return indices of two numbers that add up to target.
    >>> two_sum([2, 7, 11, 15], 9)
    [0, 1]
    >>> two_sum([3, 2, 4], 6)
    [1, 2]
    \"\"\"
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []"""
        },
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.


def reverse_string(s: str) -> str:
    \"\"\"Reverse a string.
    >>> reverse_string("hello")
    'olleh'
    >>> reverse_string("")
    ''
    \"\"\"""",
            "assistant": """def reverse_string(s: str) -> str:
    \"\"\"Reverse a string.
    >>> reverse_string("hello")
    'olleh'
    >>> reverse_string("")
    ''
    \"\"\"
    return s[::-1]"""
        },
        {
            "user": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

from typing import List


def is_sorted(lst: List[int]) -> bool:
    \"\"\"Check if a list is sorted in non-decreasing order.
    >>> is_sorted([1, 2, 3, 4])
    True
    >>> is_sorted([1, 3, 2])
    False
    >>> is_sorted([])
    True
    \"\"\"""",
            "assistant": """from typing import List


def is_sorted(lst: List[int]) -> bool:
    \"\"\"Check if a list is sorted in non-decreasing order.
    >>> is_sorted([1, 2, 3, 4])
    True
    >>> is_sorted([1, 3, 2])
    False
    >>> is_sorted([])
    True
    \"\"\"
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True"""
        },
    ]
    return examples


if __name__ == "__main__":
    main()
