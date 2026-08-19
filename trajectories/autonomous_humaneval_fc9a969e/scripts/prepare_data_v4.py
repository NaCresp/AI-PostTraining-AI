#!/usr/bin/env python3
"""Prepare enhanced SFT data v4 for HumanEval-style function completion.

Key improvements over v3:
1. Better extraction of complete function definitions from Magicoder datasets
2. More varied system prompts for coding tasks
3. 20% of samples use markdown code blocks to teach both formats
4. 600 synthetic HumanEval-style examples (varying complexity)
5. Stricter filtering and syntax checking
6. All samples formatted as complete function definitions
"""
import json
import random
import ast
import re
from datasets import load_dataset

random.seed(42)

INSTRUCTION = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"

SYSTEM_PROMPTS = [
    "You are a helpful coding assistant that writes clean, correct Python code.",
    "You are an expert Python programmer. Write efficient, well-structured code.",
    "You are a professional software engineer specializing in Python development.",
    "You write clean, idiomatic Python code that follows best practices.",
    "You are a Python expert. Provide accurate, working code solutions.",
    "You are a skilled programmer. Write clear, correct Python implementations.",
]

def is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except:
        return False

def extract_function_from_code(code: str) -> str | None:
    """Extract the first complete function definition from code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get the source code for this function
                func_lines = ast.get_source_segment(code, node)
                if func_lines:
                    return func_lines
        return None
    except:
        return None

def extract_functions_from_text(text: str) -> list[str]:
    """Extract all complete function definitions from text (may contain markdown, explanations, etc)."""
    functions = []

    # Try to extract from markdown code blocks first
    code_block_pattern = r'```python\s*(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        func = extract_function_from_code(match)
        if func:
            functions.append(func)

    # If no markdown blocks, try the whole text
    if not functions:
        func = extract_function_from_code(text)
        if func:
            functions.append(func)

    # Filter out invalid functions
    return [f for f in functions if is_valid_python(f) and len(f) < 3000]

def extract_signature_and_docstring(func_code: str) -> str | None:
    """Extract just the signature and docstring from a function."""
    try:
        tree = ast.parse(func_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lines = func_code.split('\n')
                sig_line = lines[0]  # def foo(...):

                # Check if there's a docstring
                if (node.body and
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                    # Has docstring - find it in source
                    docstring_node = node.body[0]
                    docstring = ast.get_source_segment(func_code, docstring_node)
                    if docstring:
                        return f"{sig_line}\n    {docstring}\n"

                # No docstring - just return signature
                return f"{sig_line}\n"
        return None
    except:
        return None

def make_sample(prompt: str, solution: str, system_msg: str | None = None, use_markdown: bool = False):
    """Create a training sample in ChatML format."""
    if system_msg is None:
        system_msg = random.choice(SYSTEM_PROMPTS)

    # Optionally wrap solution in markdown code block
    if use_markdown:
        solution = f"```python\n{solution}\n```"

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
    mbpp_count = 0
    for row in mbpp:
        prompt = row.get("prompt", "")
        code = row.get("code", "")
        if not prompt or not code:
            continue
        if not is_valid_python(code):
            continue

        # Use markdown blocks for 20% of samples
        use_md = random.random() < 0.2
        sample = make_sample(prompt, code, use_markdown=use_md)
        samples.append(sample)
        mbpp_count += 1
    print(f"  MBPP: {mbpp_count} samples")
except Exception as e:
    print(f"  MBPP failed: {e}")

# 2. Magicoder-OSS - Extract Python functions and create HumanEval-style prompts
print("Loading Magicoder-OSS...")
try:
    magi = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    oss_count = 0
    for row in magi:
        problem = row.get("problem", "")
        solution = row.get("solution", "")
        if not problem or not solution:
            continue

        # Extract function definitions from solution
        functions = extract_functions_from_text(solution)
        for func in functions[:1]:  # Use first function only
            if "def " not in func:
                continue

            # Try to extract signature + docstring
            sig_doc = extract_signature_and_docstring(func)
            if sig_doc:
                # Create HumanEval-style prompt
                use_md = random.random() < 0.2
                sample = make_sample(sig_doc, func, use_markdown=use_md)
                samples.append(sample)
                oss_count += 1
            else:
                # Fallback: use original problem as prompt
                use_md = random.random() < 0.2
                sample = make_sample(problem, func, use_markdown=use_md)
                samples.append(sample)
                oss_count += 1

        if oss_count >= 15000:  # Limit to avoid too many samples
            break
    print(f"  Magicoder-OSS: {oss_count} samples")
except Exception as e:
    print(f"  Magicoder-OSS failed: {e}")

# 3. Magicoder-Evol - Extract Python functions
print("Loading Magicoder-Evol...")
try:
    evol = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    evol_count = 0
    for row in evol:
        instruction = row.get("instruction", "")
        response = row.get("response", "")
        if not instruction or not response:
            continue

        # Extract function definitions from response
        functions = extract_functions_from_text(response)
        for func in functions[:1]:  # Use first function only
            if "def " not in func:
                continue

            # Try to extract signature + docstring
            sig_doc = extract_signature_and_docstring(func)
            if sig_doc:
                # Create HumanEval-style prompt
                use_md = random.random() < 0.2
                sample = make_sample(sig_doc, func, use_markdown=use_md)
                samples.append(sample)
                evol_count += 1
            else:
                # Fallback: use original instruction as prompt
                use_md = random.random() < 0.2
                sample = make_sample(instruction, func, use_markdown=use_md)
                samples.append(sample)
                evol_count += 1

        if evol_count >= 20000:  # Limit to avoid too many samples
            break
    print(f"  Magicoder-Evol: {evol_count} samples")
except Exception as e:
    print(f"  Magicoder-Evol failed: {e}")

# 4. Generate 600 synthetic HumanEval-style examples
print("Creating synthetic samples...")

synthetic_templates = [
    # Basic arithmetic and math
    {
        "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of two integers.\n    >>> add(2, 3)\n    5\n    >>> add(-1, 1)\n    0\n    """\n',
        "solution": "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of two integers.\n    >>> add(2, 3)\n    5\n    >>> add(-1, 1)\n    0\n    \"\"\"\n    return a + b"
    },
    {
        "prompt": 'def multiply(a: int, b: int) -> int:\n    """Return the product of two integers.\n    >>> multiply(3, 4)\n    12\n    """\n',
        "solution": "def multiply(a: int, b: int) -> int:\n    \"\"\"Return the product of two integers.\n    >>> multiply(3, 4)\n    12\n    \"\"\"\n    return a * b"
    },
    {
        "prompt": 'def power(base: int, exp: int) -> int:\n    """Return base raised to the power of exp.\n    >>> power(2, 3)\n    8\n    """\n',
        "solution": "def power(base: int, exp: int) -> int:\n    \"\"\"Return base raised to the power of exp.\n    >>> power(2, 3)\n    8\n    \"\"\"\n    return base ** exp"
    },
    # List operations
    {
        "prompt": 'from typing import List\n\ndef mean(numbers: List[float]) -> float:\n    """Return the mean of a list of numbers.\n    >>> mean([1.0, 2.0, 3.0])\n    2.0\n    """\n',
        "solution": "from typing import List\n\ndef mean(numbers: List[float]) -> float:\n    \"\"\"Return the mean of a list of numbers.\n    >>> mean([1.0, 2.0, 3.0])\n    2.0\n    \"\"\"\n    return sum(numbers) / len(numbers)"
    },
    {
        "prompt": 'from typing import List\n\ndef reverse_list(lst: List[int]) -> List[int]:\n    """Return a reversed copy of the list.\n    >>> reverse_list([1, 2, 3])\n    [3, 2, 1]\n    """\n',
        "solution": "from typing import List\n\ndef reverse_list(lst: List[int]) -> List[int]:\n    \"\"\"Return a reversed copy of the list.\n    >>> reverse_list([1, 2, 3])\n    [3, 2, 1]\n    \"\"\"\n    return lst[::-1]"
    },
    {
        "prompt": 'from typing import List\n\ndef sum_list(numbers: List[int]) -> int:\n    """Return the sum of all numbers in the list.\n    >>> sum_list([1, 2, 3, 4])\n    10\n    """\n',
        "solution": "from typing import List\n\ndef sum_list(numbers: List[int]) -> int:\n    \"\"\"Return the sum of all numbers in the list.\n    >>> sum_list([1, 2, 3, 4])\n    10\n    \"\"\"\n    return sum(numbers)"
    },
    # String operations
    {
        "prompt": 'def is_palindrome(s: str) -> bool:\n    """Check if a string is a palindrome.\n    >>> is_palindrome("racecar")\n    True\n    >>> is_palindrome("hello")\n    False\n    """\n',
        "solution": 'def is_palindrome(s: str) -> bool:\n    """Check if a string is a palindrome.\n    >>> is_palindrome("racecar")\n    True\n    >>> is_palindrome("hello")\n    False\n    """\n    return s == s[::-1]'
    },
    {
        "prompt": 'def count_vowels(s: str) -> int:\n    """Count the number of vowels in a string.\n    >>> count_vowels("hello")\n    2\n    >>> count_vowels("AEIOU")\n    5\n    """\n',
        "solution": 'def count_vowels(s: str) -> int:\n    """Count the number of vowels in a string.\n    >>> count_vowels("hello")\n    2\n    >>> count_vowels("AEIOU")\n    5\n    """\n    return sum(1 for c in s.lower() if c in "aeiou")'
    },
    {
        "prompt": 'def reverse_words(s: str) -> str:\n    """Reverse the order of words in a string.\n    >>> reverse_words("hello world")\n    \'world hello\'\n    """\n',
        "solution": 'def reverse_words(s: str) -> str:\n    """Reverse the order of words in a string.\n    >>> reverse_words("hello world")\n    \'world hello\'\n    """\n    return " ".join(s.split()[::-1])'
    },
    # Recursion
    {
        "prompt": 'def factorial(n: int) -> int:\n    """Return the factorial of n.\n    >>> factorial(5)\n    120\n    >>> factorial(0)\n    1\n    """\n',
        "solution": "def factorial(n: int) -> int:\n    \"\"\"Return the factorial of n.\n    >>> factorial(5)\n    120\n    >>> factorial(0)\n    1\n    \"\"\"\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    },
    {
        "prompt": 'def fibonacci(n: int) -> int:\n    """Return the nth Fibonacci number.\n    >>> fibonacci(0)\n    0\n    >>> fibonacci(10)\n    55\n    """\n',
        "solution": "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\n    >>> fibonacci(0)\n    0\n    >>> fibonacci(10)\n    55\n    \"\"\"\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
    },
    {
        "prompt": 'from typing import List\n\ndef flatten(lst: List) -> List:\n    """Flatten a nested list.\n    >>> flatten([1, [2, [3, 4], 5]])\n    [1, 2, 3, 4, 5]\n    """\n',
        "solution": "from typing import List\n\ndef flatten(lst: List) -> List:\n    \"\"\"Flatten a nested list.\n    >>> flatten([1, [2, [3, 4], 5]])\n    [1, 2, 3, 4, 5]\n    \"\"\"\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"
    },
    # Algorithms
    {
        "prompt": 'from typing import List\n\ndef binary_search(arr: List[int], target: int) -> int:\n    """Return the index of target in sorted array, or -1 if not found.\n    >>> binary_search([1, 3, 5, 7, 9], 5)\n    2\n    >>> binary_search([1, 3, 5, 7, 9], 4)\n    -1\n    """\n',
        "solution": "from typing import List\n\ndef binary_search(arr: List[int], target: int) -> int:\n    \"\"\"Return the index of target in sorted array, or -1 if not found.\n    >>> binary_search([1, 3, 5, 7, 9], 5)\n    2\n    >>> binary_search([1, 3, 5, 7, 9], 4)\n    -1\n    \"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
    },
    {
        "prompt": 'def is_prime(n: int) -> bool:\n    """Check if a number is prime.\n    >>> is_prime(7)\n    True\n    >>> is_prime(4)\n    False\n    """\n',
        "solution": "def is_prime(n: int) -> bool:\n    \"\"\"Check if a number is prime.\n    >>> is_prime(7)\n    True\n    >>> is_prime(4)\n    False\n    \"\"\"\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
    },
    {
        "prompt": 'def gcd(a: int, b: int) -> int:\n    """Return the greatest common divisor of a and b.\n    >>> gcd(12, 8)\n    4\n    """\n',
        "solution": "def gcd(a: int, b: int) -> int:\n    \"\"\"Return the greatest common divisor of a and b.\n    >>> gcd(12, 8)\n    4\n    \"\"\"\n    while b:\n        a, b = b, a % b\n    return a"
    },
    # Data structures
    {
        "prompt": 'from typing import List\n\ndef remove_duplicates(lst: List[int]) -> List[int]:\n    """Remove duplicates from a list while preserving order.\n    >>> remove_duplicates([1, 2, 2, 3, 1, 4])\n    [1, 2, 3, 4]\n    """\n',
        "solution": "from typing import List\n\ndef remove_duplicates(lst: List[int]) -> List[int]:\n    \"\"\"Remove duplicates from a list while preserving order.\n    >>> remove_duplicates([1, 2, 2, 3, 1, 4])\n    [1, 2, 3, 4]\n    \"\"\"\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"
    },
    {
        "prompt": 'from typing import Dict\n\ndef count_chars(s: str) -> Dict[str, int]:\n    """Count the frequency of each character in a string.\n    >>> count_chars("hello")\n    {\'h\': 1, \'e\': 1, \'l\': 2, \'o\': 1}\n    """\n',
        "solution": "from typing import Dict\n\ndef count_chars(s: str) -> Dict[str, int]:\n    \"\"\"Count the frequency of each character in a string.\n    >>> count_chars(\"hello\")\n    {'h': 1, 'e': 1, 'l': 2, 'o': 1}\n    \"\"\"\n    freq = {}\n    for c in s:\n        freq[c] = freq.get(c, 0) + 1\n    return freq"
    },
    {
        "prompt": 'from typing import List, Tuple\n\ndef two_sum(nums: List[int], target: int) -> Tuple[int, int]:\n    """Return indices of two numbers that add up to target.\n    >>> two_sum([2, 7, 11, 15], 9)\n    (0, 1)\n    """\n',
        "solution": "from typing import List, Tuple\n\ndef two_sum(nums: List[int], target: int) -> Tuple[int, int]:\n    \"\"\"Return indices of two numbers that add up to target.\n    >>> two_sum([2, 7, 11, 15], 9)\n    (0, 1)\n    \"\"\"\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return (seen[complement], i)\n        seen[num] = i\n    return (-1, -1)"
    },
    # More complex operations
    {
        "prompt": 'from typing import List\n\ndef merge_sorted(a: List[int], b: List[int]) -> List[int]:\n    """Merge two sorted lists into one sorted list.\n    >>> merge_sorted([1, 3, 5], [2, 4, 6])\n    [1, 2, 3, 4, 5, 6]\n    """\n',
        "solution": "from typing import List\n\ndef merge_sorted(a: List[int], b: List[int]) -> List[int]:\n    \"\"\"Merge two sorted lists into one sorted list.\n    >>> merge_sorted([1, 3, 5], [2, 4, 6])\n    [1, 2, 3, 4, 5, 6]\n    \"\"\"\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result"
    },
    {
        "prompt": 'from typing import List\n\ndef max_subarray_sum(nums: List[int]) -> int:\n    """Find the maximum subarray sum using Kadane\'s algorithm.\n    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n    6\n    """\n',
        "solution": "from typing import List\n\ndef max_subarray_sum(nums: List[int]) -> int:\n    \"\"\"Find the maximum subarray sum using Kadane's algorithm.\n    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n    6\n    \"\"\"\n    max_sum = nums[0]\n    current_sum = nums[0]\n    for num in nums[1:]:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum"
    },
]

# Generate 600 synthetic samples (30 reps per template)
synth_count = 0
for template in synthetic_templates:
    for _ in range(30):
        use_md = random.random() < 0.2
        sample = make_sample(template["prompt"], template["solution"], use_markdown=use_md)
        samples.append(sample)
        synth_count += 1

print(f"  Synthetic: {synth_count} samples")

# Shuffle all samples
random.shuffle(samples)

print(f"\nTotal: {len(samples)} samples")

# Save to file
output_file = "/home/ben/task/training_data_v4.jsonl"
with open(output_file, "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

print(f"Saved to {output_file}")

# Print sample for verification
print("\n" + "="*80)
print("Sample training example:")
print("="*80)
if samples:
    print(samples[0]["text"][:800] + "...")
