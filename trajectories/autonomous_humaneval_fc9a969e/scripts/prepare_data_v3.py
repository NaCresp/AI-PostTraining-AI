#!/usr/bin/env python3
"""Prepare SFT data for HumanEval-style function completion.

Key design choices:
- Each sample is a complete chat conversation with proper EOS
- The model must learn to produce ONLY the function body and stop
- Training data closely matches HumanEval eval format
- Use thinking tags since eval template uses enable_thinking=false which prepends <think>\n\n</think>
"""
import json
import random
from datasets import load_dataset

random.seed(42)

INSTRUCTION = """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"""

def make_sample(prompt: str, solution: str, system_msg: str = "You are a helpful coding assistant that writes clean, correct Python code."):
    text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{INSTRUCTION}{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{solution}<|im_end|>\n"
    )
    return {"text": text}

samples = []

# 1. MBPP - sanitized (good Python function problems)
print("Loading MBPP...")
try:
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    for row in mbpp:
        prompt = row.get("prompt", "")
        code = row.get("code", "")
        if prompt and code:
            user_msg = f"{INSTRUCTION}{prompt}"
            sample = make_sample(prompt, code)
            samples.append(sample)
    print(f"  MBPP: {len(samples)} samples")
except Exception as e:
    print(f"  MBPP failed: {e}")

mbpp_count = len(samples)

# 2. Magicoder-OSS - Python function completion only
print("Loading Magicoder-OSS...")
try:
    magi = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    count = 0
    for row in magi:
        problem = row.get("problem", "")
        solution = row.get("solution", "")
        if not problem or not solution:
            continue
        if "def " not in solution:
            continue
        if "python" not in problem.lower() and "def " not in problem:
            continue
        if len(solution) > 3000:
            continue
        sample = make_sample(problem, solution)
        samples.append(sample)
        count += 1
    print(f"  Magicoder-OSS: {count} samples")
except Exception as e:
    print(f"  Magicoder-OSS failed: {e}")

# 3. Magicoder-Evol - Python function completion
print("Loading Magicoder-Evol...")
try:
    evol = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    count = 0
    for row in evol:
        instruction = row.get("instruction", "")
        response = row.get("response", "")
        if not instruction or not response:
            continue
        if "def " not in response:
            continue
        if "python" not in instruction.lower() and "def " not in instruction:
            continue
        if len(response) > 3000:
            continue
        sample = make_sample(instruction, response)
        samples.append(sample)
        count += 1
    print(f"  Magicoder-Evol: {count} samples")
except Exception as e:
    print(f"  Magicoder-Evol failed: {e}")

# 4. Synthetic HumanEval-style function completions (many reps to strengthen the pattern)
print("Creating synthetic samples...")
synthetic_problems = [
    {
        "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of two integers.\n    >>> add(2, 3)\n    5\n    >>> add(-1, 1)\n    0\n    """\n',
        "solution": "    return a + b"
    },
    {
        "prompt": 'from typing import List\n\ndef mean(numbers: List[float]) -> float:\n    """Return the mean of a list of numbers.\n    >>> mean([1.0, 2.0, 3.0])\n    2.0\n    """\n',
        "solution": "    return sum(numbers) / len(numbers)"
    },
    {
        "prompt": 'def factorial(n: int) -> int:\n    """Return the factorial of n.\n    >>> factorial(5)\n    120\n    >>> factorial(0)\n    1\n    """\n',
        "solution": "    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    },
    {
        "prompt": 'def is_palindrome(s: str) -> bool:\n    """Check if a string is a palindrome.\n    >>> is_palindrome("racecar")\n    True\n    >>> is_palindrome("hello")\n    False\n    """\n',
        "solution": '    return s == s[::-1]'
    },
    {
        "prompt": 'from typing import List\n\ndef flatten(lst: List) -> List:\n    """Flatten a nested list.\n    >>> flatten([1, [2, [3, 4], 5]])\n    [1, 2, 3, 4, 5]\n    """\n',
        "solution": "    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"
    },
    {
        "prompt": 'def fibonacci(n: int) -> int:\n    """Return the nth Fibonacci number.\n    >>> fibonacci(0)\n    0\n    >>> fibonacci(1)\n    1\n    >>> fibonacci(10)\n    55\n    """\n',
        "solution": "    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
    },
    {
        "prompt": 'from typing import List\n\ndef binary_search(arr: List[int], target: int) -> int:\n    """Return the index of target in sorted array, or -1 if not found.\n    >>> binary_search([1, 3, 5, 7, 9], 5)\n    2\n    >>> binary_search([1, 3, 5, 7, 9], 4)\n    -1\n    """\n',
        "solution": "    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
    },
    {
        "prompt": 'def gcd(a: int, b: int) -> int:\n    """Return the greatest common divisor of a and b.\n    >>> gcd(12, 8)\n    4\n    >>> gcd(15, 5)\n    5\n    """\n',
        "solution": "    while b:\n        a, b = b, a % b\n    return a"
    },
    {
        "prompt": 'from typing import List\n\ndef remove_duplicates(lst: List[int]) -> List[int]:\n    """Remove duplicates from a list while preserving order.\n    >>> remove_duplicates([1, 2, 2, 3, 1, 4])\n    [1, 2, 3, 4]\n    """\n',
        "solution": "    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"
    },
    {
        "prompt": 'def is_prime(n: int) -> bool:\n    """Check if a number is prime.\n    >>> is_prime(7)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    """\n',
        "solution": "    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
    },
    {
        "prompt": 'from typing import List\n\ndef max_subarray_sum(nums: List[int]) -> int:\n    """Find the maximum subarray sum using Kadane\'s algorithm.\n    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n    6\n    """\n',
        "solution": "    max_sum = nums[0]\n    current_sum = nums[0]\n    for num in nums[1:]:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum"
    },
    {
        "prompt": 'from typing import Dict, List\n\ndef count_chars(s: str) -> Dict[str, int]:\n    """Count the frequency of each character in a string.\n    >>> count_chars("hello")\n    {\'h\': 1, \'e\': 1, \'l\': 2, \'o\': 1}\n    """\n',
        "solution": "    freq = {}\n    for c in s:\n        freq[c] = freq.get(c, 0) + 1\n    return freq"
    },
    {
        "prompt": 'from typing import List\n\ndef merge_sorted(a: List[int], b: List[int]) -> List[int]:\n    """Merge two sorted lists into one sorted list.\n    >>> merge_sorted([1, 3, 5], [2, 4, 6])\n    [1, 2, 3, 4, 5, 6]\n    """\n',
        "solution": "    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result"
    },
    {
        "prompt": 'def reverse_words(s: str) -> str:\n    """Reverse the order of words in a string.\n    >>> reverse_words("hello world")\n    \'world hello\'\n    >>> reverse_words("  spaces  between  ")\n    \'between spaces\'\n    """\n',
        "solution": '    return " ".join(s.split()[::-1])'
    },
    {
        "prompt": 'from typing import List, Tuple\n\ndef two_sum(nums: List[int], target: int) -> Tuple[int, int]:\n    """Return indices of two numbers that add up to target.\n    >>> two_sum([2, 7, 11, 15], 9)\n    (0, 1)\n    """\n',
        "solution": "    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return (seen[complement], i)\n        seen[num] = i\n    return (-1, -1)"
    },
    {
        "prompt": 'from typing import List\n\ndef matrix_multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:\n    """Multiply two matrices.\n    >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])\n    [[19, 22], [43, 50]]\n    """\n',
        "solution": "    rows_a, cols_a = len(a), len(a[0])\n    cols_b = len(b[0])\n    result = [[0] * cols_b for _ in range(rows_a)]\n    for i in range(rows_a):\n        for j in range(cols_b):\n            for k in range(cols_a):\n                result[i][j] += a[i][k] * b[k][j]\n    return result"
    },
    {
        "prompt": 'def count_vowels(s: str) -> int:\n    """Count the number of vowels in a string.\n    >>> count_vowels("hello")\n    2\n    >>> count_vowels("AEIOU")\n    5\n    """\n',
        "solution": '    return sum(1 for c in s.lower() if c in "aeiou")'
    },
    {
        "prompt": 'from typing import Optional, List\n\ndef find_second_largest(nums: List[int]) -> Optional[int]:\n    """Find the second largest number in a list.\n    >>> find_second_largest([1, 3, 5, 2, 4])\n    4\n    >>> find_second_largest([1])\n    None\n    """\n',
        "solution": "    if len(nums) < 2:\n        return None\n    first = second = float('-inf')\n    for num in nums:\n        if num > first:\n            second = first\n            first = num\n        elif num > second and num != first:\n            second = num\n    return second if second != float('-inf') else None"
    },
    {
        "prompt": 'def caesar_cipher(text: str, shift: int) -> str:\n    """Apply Caesar cipher to text with given shift.\n    >>> caesar_cipher("abc", 1)\n    \'bcd\'\n    >>> caesar_cipher("xyz", 3)\n    \'abc\'\n    """\n',
        "solution": "    result = []\n    for c in text:\n        if c.isalpha():\n            base = ord('a') if c.islower() else ord('A')\n            result.append(chr((ord(c) - base + shift) % 26 + base))\n        else:\n            result.append(c)\n    return ''.join(result)"
    },
    {
        "prompt": 'from typing import List\n\ndef rotate_list(lst: List[int], k: int) -> List[int]:\n    """Rotate list to the right by k positions.\n    >>> rotate_list([1, 2, 3, 4, 5], 2)\n    [4, 5, 1, 2, 3]\n    """\n',
        "solution": "    if not lst:\n        return lst\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k]"
    },
]

synth_count = 0
for prob in synthetic_problems:
    for _ in range(30):
        samples.append(make_sample(prob["prompt"], prob["solution"]))
        synth_count += 1

print(f"  Synthetic: {synth_count} samples")

random.shuffle(samples)
print(f"\nTotal: {len(samples)} samples")

with open("training_data_v3.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

print("Saved to training_data_v3.jsonl")
