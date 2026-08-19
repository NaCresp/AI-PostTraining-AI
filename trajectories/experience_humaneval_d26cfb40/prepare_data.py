"""
Prepare training data for SFT on Python function completion.
Combines MBPP, CodeAlpaca, and synthetic data formatted as function completions.
"""
import json
import re
import random
from datasets import load_dataset

random.seed(42)

def format_as_function_completion(signature_and_doc: str, body: str) -> dict:
    """Format a training example as prompt (signature+docstring) -> completion (body)."""
    return {
        "prompt": signature_and_doc.rstrip(),
        "completion": "\n" + body.rstrip() + "\n"
    }

def extract_function_parts(code: str):
    """Extract function signature+docstring and body from a Python function."""
    lines = code.split('\n')
    # Find the def line
    def_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            def_idx = i
            break
    if def_idx is None:
        return None, None

    # Find docstring end
    in_docstring = False
    docstring_end = def_idx
    for i in range(def_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                if stripped.count(quote) >= 2 and len(stripped) > 6:
                    # Single-line docstring
                    docstring_end = i
                    break
                else:
                    in_docstring = True
                    continue
            elif stripped == '':
                continue
            else:
                # No docstring - the body starts here
                docstring_end = def_idx
                break
        else:
            if quote in stripped:
                docstring_end = i
                in_docstring = False
                break

    if in_docstring:
        return None, None

    header = '\n'.join(lines[def_idx:docstring_end + 1])
    body = '\n'.join(lines[docstring_end + 1:])

    # Clean body - remove empty leading lines
    body_lines = body.split('\n')
    while body_lines and body_lines[0].strip() == '':
        body_lines.pop(0)
    body = '\n'.join(body_lines)

    if not body.strip():
        return None, None

    return header, body


def process_mbpp():
    """Process MBPP dataset into function completion format."""
    examples = []

    for split in ['train', 'test', 'validation']:
        ds = load_dataset('mbpp', 'sanitized', split=split)
        for item in ds:
            code = item['code']
            prompt_text = item['prompt']
            test_cases = item.get('test_list', [])

            # Try to extract function parts
            header, body = extract_function_parts(code)
            if header and body:
                # Add docstring if not present
                if '"""' not in header and "'''" not in header:
                    # Create a proper header with docstring
                    def_line = header.split('\n')[0]
                    indent = '    '
                    docstring = f'{indent}"""{prompt_text}'
                    if test_cases:
                        docstring += '\n'
                        for tc in test_cases[:3]:
                            docstring += f'{indent}>>> {tc}\n'
                    docstring += f'{indent}"""'
                    header = def_line + '\n' + docstring

                examples.append(format_as_function_completion(header, body))

    print(f"MBPP: {len(examples)} examples")
    return examples


def process_code_alpaca():
    """Process CodeAlpaca-20k for Python function completions."""
    ds = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
    examples = []

    for item in ds:
        instruction = item['instruction']
        output = item['output']

        # Filter for Python-related
        if not any(kw in instruction.lower() for kw in ['python', 'function', 'def ', 'write a', 'create a', 'implement']):
            continue
        if not any(kw in output.lower() for kw in ['def ', 'return', 'print(']):
            continue
        # Skip if it looks like non-Python
        if any(kw in output for kw in ['public static', 'System.out', '#include', 'console.log', 'func ', 'fn ']):
            continue

        # Try to extract function
        header, body = extract_function_parts(output)
        if header and body:
            if '"""' not in header and "'''" not in header:
                def_line = header.split('\n')[0]
                indent = '    '
                docstring = f'{indent}"""{instruction}"""'
                header = def_line + '\n' + docstring
            examples.append(format_as_function_completion(header, body))

    print(f"CodeAlpaca: {len(examples)} examples")
    return examples


def process_starcoder():
    """Process self-instruct-starcoder dataset."""
    examples = []
    for split_name in ['curated', 'compile']:
        try:
            ds = load_dataset('codeparrot/self-instruct-starcoder', split=split_name)
            for item in ds:
                output = item.get('output', '')
                instruction = item.get('instruction', '')

                # Filter for Python
                if 'def ' not in output:
                    continue
                if any(kw in output for kw in ['public static', 'System.out', '#include', 'console.log']):
                    continue

                header, body = extract_function_parts(output)
                if header and body:
                    if '"""' not in header and "'''" not in header:
                        def_line = header.split('\n')[0]
                        indent = '    '
                        desc = instruction[:200] if instruction else 'Complete this function.'
                        docstring = f'{indent}"""{desc}"""'
                        header = def_line + '\n' + docstring
                    examples.append(format_as_function_completion(header, body))
        except Exception as e:
            print(f"Starcoder {split_name} error: {e}")

    print(f"Starcoder: {len(examples)} examples")
    return examples


SYNTHETIC_FUNCTIONS = [
    {
        "header": '''def fibonacci(n: int) -> int:\n    """Return the nth Fibonacci number.\n    >>> fibonacci(0)\n    0\n    >>> fibonacci(1)\n    1\n    >>> fibonacci(10)\n    55\n    """''',
        "body": '''    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b'''
    },
    {
        "header": '''def is_palindrome(s: str) -> bool:\n    """Check if a string is a palindrome, ignoring case and non-alphanumeric characters.\n    >>> is_palindrome("racecar")\n    True\n    >>> is_palindrome("hello")\n    False\n    >>> is_palindrome("A man a plan a canal Panama")\n    True\n    """''',
        "body": '''    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]'''
    },
    {
        "header": '''def flatten_list(nested: list) -> list:\n    """Flatten a nested list into a single list.\n    >>> flatten_list([[1, 2], [3, [4, 5]]])\n    [1, 2, 3, 4, 5]\n    >>> flatten_list([1, [2, [3, [4]]]])\n    [1, 2, 3, 4]\n    """''',
        "body": '''    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten_list(item))\n        else:\n            result.append(item)\n    return result'''
    },
    {
        "header": '''def binary_search(arr: list, target: int) -> int:\n    """Perform binary search on a sorted list. Return index of target, or -1 if not found.\n    >>> binary_search([1, 3, 5, 7, 9], 5)\n    2\n    >>> binary_search([1, 3, 5, 7, 9], 4)\n    -1\n    """''',
        "body": '''    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1'''
    },
    {
        "header": '''def gcd(a: int, b: int) -> int:\n    """Compute the greatest common divisor of two integers.\n    >>> gcd(12, 8)\n    4\n    >>> gcd(17, 5)\n    1\n    """''',
        "body": '''    while b:\n        a, b = b, a % b\n    return a'''
    },
    {
        "header": '''def merge_sorted_lists(list1: list, list2: list) -> list:\n    """Merge two sorted lists into a single sorted list.\n    >>> merge_sorted_lists([1, 3, 5], [2, 4, 6])\n    [1, 2, 3, 4, 5, 6]\n    """''',
        "body": '''    result = []\n    i = j = 0\n    while i < len(list1) and j < len(list2):\n        if list1[i] <= list2[j]:\n            result.append(list1[i])\n            i += 1\n        else:\n            result.append(list2[j])\n            j += 1\n    result.extend(list1[i:])\n    result.extend(list2[j:])\n    return result'''
    },
    {
        "header": '''def count_vowels(s: str) -> int:\n    """Count the number of vowels in a string.\n    >>> count_vowels("hello")\n    2\n    >>> count_vowels("aeiou")\n    5\n    """''',
        "body": '''    return sum(1 for c in s.lower() if c in 'aeiou')'''
    },
    {
        "header": '''def matrix_multiply(A: list, B: list) -> list:\n    """Multiply two matrices represented as lists of lists.\n    >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])\n    [[19, 22], [43, 50]]\n    """''',
        "body": '''    rows_A, cols_A = len(A), len(A[0])\n    rows_B, cols_B = len(B), len(B[0])\n    result = [[0] * cols_B for _ in range(rows_A)]\n    for i in range(rows_A):\n        for j in range(cols_B):\n            for k in range(cols_A):\n                result[i][j] += A[i][k] * B[k][j]\n    return result'''
    },
    {
        "header": '''def is_prime(n: int) -> bool:\n    """Check if a number is prime.\n    >>> is_prime(2)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(17)\n    True\n    """''',
        "body": '''    if n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True'''
    },
    {
        "header": '''def reverse_string(s: str) -> str:\n    """Reverse a string.\n    >>> reverse_string("hello")\n    'olleh'\n    >>> reverse_string("abcde")\n    'edcba'\n    """''',
        "body": '''    return s[::-1]'''
    },
    {
        "header": '''def remove_duplicates(lst: list) -> list:\n    """Remove duplicates from a list while preserving order.\n    >>> remove_duplicates([1, 2, 2, 3, 1, 4])\n    [1, 2, 3, 4]\n    """''',
        "body": '''    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result'''
    },
    {
        "header": '''def two_sum(nums: list, target: int) -> list:\n    """Find two indices whose values sum to target.\n    >>> two_sum([2, 7, 11, 15], 9)\n    [0, 1]\n    """''',
        "body": '''    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []'''
    },
    {
        "header": '''def max_subarray_sum(nums: list) -> int:\n    """Find the maximum sum of a contiguous subarray (Kadane's algorithm).\n    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n    6\n    """''',
        "body": '''    max_sum = current_sum = nums[0]\n    for num in nums[1:]:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum'''
    },
    {
        "header": '''def power(base: float, exp: int) -> float:\n    """Compute base raised to the power of exp using fast exponentiation.\n    >>> power(2, 10)\n    1024\n    >>> power(3, 0)\n    1\n    """''',
        "body": '''    if exp == 0:\n        return 1\n    if exp < 0:\n        return 1 / power(base, -exp)\n    result = 1\n    while exp > 0:\n        if exp % 2 == 1:\n            result *= base\n        base *= base\n        exp //= 2\n    return result'''
    },
    {
        "header": '''def longest_common_prefix(strs: list) -> str:\n    """Find the longest common prefix string amongst a list of strings.\n    >>> longest_common_prefix(["flower","flow","flight"])\n    'fl'\n    >>> longest_common_prefix(["dog","racecar","car"])\n    ''\n    """''',
        "body": '''    if not strs:\n        return ""\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ""\n    return prefix'''
    },
    {
        "header": '''def valid_parentheses(s: str) -> bool:\n    """Check if a string of parentheses is valid.\n    >>> valid_parentheses("(())")\n    True\n    >>> valid_parentheses("(]")\n    False\n    >>> valid_parentheses("()[]{}")\n    True\n    """''',
        "body": '''    stack = []\n    mapping = {')': '(', ']': '[', '}': '{'}\n    for char in s:\n        if char in mapping:\n            if not stack or stack[-1] != mapping[char]:\n                return False\n            stack.pop()\n        else:\n            stack.append(char)\n    return len(stack) == 0'''
    },
    {
        "header": '''def rotate_list(lst: list, k: int) -> list:\n    """Rotate a list to the right by k positions.\n    >>> rotate_list([1, 2, 3, 4, 5], 2)\n    [4, 5, 1, 2, 3]\n    """''',
        "body": '''    if not lst:\n        return lst\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k]'''
    },
    {
        "header": '''def char_frequency(s: str) -> dict:\n    """Return a dictionary with the frequency of each character in the string.\n    >>> char_frequency("hello")\n    {'h': 1, 'e': 1, 'l': 2, 'o': 1}\n    """''',
        "body": '''    freq = {}\n    for char in s:\n        freq[char] = freq.get(char, 0) + 1\n    return freq'''
    },
    {
        "header": '''def insertion_sort(arr: list) -> list:\n    """Sort a list using insertion sort.\n    >>> insertion_sort([5, 2, 4, 6, 1, 3])\n    [1, 2, 3, 4, 5, 6]\n    """''',
        "body": '''    arr = arr.copy()\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = key\n    return arr'''
    },
    {
        "header": '''def string_to_int(s: str) -> int:\n    """Convert a string to an integer, handling leading whitespace and sign.\n    >>> string_to_int("42")\n    42\n    >>> string_to_int("   -42")\n    -42\n    >>> string_to_int("4193 with words")\n    4193\n    """''',
        "body": '''    s = s.strip()\n    if not s:\n        return 0\n    sign = 1\n    i = 0\n    if s[0] in ('+', '-'):\n        sign = -1 if s[0] == '-' else 1\n        i = 1\n    result = 0\n    while i < len(s) and s[i].isdigit():\n        result = result * 10 + int(s[i])\n        i += 1\n    return sign * result'''
    },
    {
        "header": '''def transpose_matrix(matrix: list) -> list:\n    """Transpose a matrix (list of lists).\n    >>> transpose_matrix([[1, 2, 3], [4, 5, 6]])\n    [[1, 4], [2, 5], [3, 6]]\n    """''',
        "body": '''    if not matrix:\n        return []\n    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]'''
    },
    {
        "header": '''def all_permutations(lst: list) -> list:\n    """Generate all permutations of a list.\n    >>> sorted(all_permutations([1, 2, 3]))\n    [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]\n    """''',
        "body": '''    if len(lst) <= 1:\n        return [lst]\n    result = []\n    for i, elem in enumerate(lst):\n        rest = lst[:i] + lst[i+1:]\n        for perm in all_permutations(rest):\n            result.append([elem] + perm)\n    return result'''
    },
    {
        "header": '''def depth_first_search(graph: dict, start: str) -> list:\n    """Perform DFS on a graph represented as adjacency list, return visited nodes.\n    >>> depth_first_search({'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}, 'A')\n    ['A', 'B', 'D', 'C']\n    """''',
        "body": '''    visited = []\n    stack = [start]\n    seen = set()\n    while stack:\n        node = stack.pop()\n        if node not in seen:\n            seen.add(node)\n            visited.append(node)\n            for neighbor in reversed(graph.get(node, [])):\n                if neighbor not in seen:\n                    stack.append(neighbor)\n    return visited'''
    },
    {
        "header": '''def caesar_cipher(text: str, shift: int) -> str:\n    """Encrypt text using Caesar cipher with the given shift.\n    >>> caesar_cipher("hello", 3)\n    'khoor'\n    >>> caesar_cipher("abc", 1)\n    'bcd'\n    """''',
        "body": '''    result = []\n    for char in text:\n        if char.isalpha():\n            base = ord('a') if char.islower() else ord('A')\n            result.append(chr((ord(char) - base + shift) % 26 + base))\n        else:\n            result.append(char)\n    return ''.join(result)'''
    },
    {
        "header": '''def chunk_list(lst: list, size: int) -> list:\n    """Split a list into chunks of the given size.\n    >>> chunk_list([1, 2, 3, 4, 5], 2)\n    [[1, 2], [3, 4], [5]]\n    """''',
        "body": '''    return [lst[i:i+size] for i in range(0, len(lst), size)]'''
    },
    {
        "header": '''def find_missing_number(nums: list) -> int:\n    """Find the missing number in a list containing 0 to n with one missing.\n    >>> find_missing_number([3, 0, 1])\n    2\n    >>> find_missing_number([0, 1])\n    2\n    """''',
        "body": '''    n = len(nums)\n    expected_sum = n * (n + 1) // 2\n    return expected_sum - sum(nums)'''
    },
    {
        "header": '''def intersection(lst1: list, lst2: list) -> list:\n    """Find the intersection of two lists.\n    >>> sorted(intersection([1, 2, 3, 4], [3, 4, 5, 6]))\n    [3, 4]\n    """''',
        "body": '''    set1 = set(lst1)\n    return [x for x in lst2 if x in set1]'''
    },
    {
        "header": '''def sum_digits(n: int) -> int:\n    """Return the sum of digits of a non-negative integer.\n    >>> sum_digits(12345)\n    15\n    >>> sum_digits(0)\n    0\n    """''',
        "body": '''    total = 0\n    n = abs(n)\n    while n > 0:\n        total += n % 10\n        n //= 10\n    return total'''
    },
    {
        "header": '''def nth_largest(nums: list, n: int) -> int:\n    """Find the nth largest element in a list.\n    >>> nth_largest([3, 1, 4, 1, 5, 9], 2)\n    5\n    """''',
        "body": '''    sorted_nums = sorted(nums, reverse=True)\n    return sorted_nums[n - 1]'''
    },
    {
        "header": '''def run_length_encoding(s: str) -> str:\n    """Encode a string using run-length encoding.\n    >>> run_length_encoding("aaabbbcc")\n    'a3b3c2'\n    >>> run_length_encoding("abc")\n    'a1b1c1'\n    """''',
        "body": '''    if not s:\n        return ""\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            result.append(s[i-1] + str(count))\n            count = 1\n    result.append(s[-1] + str(count))\n    return ''.join(result)'''
    },
]


def generate_synthetic():
    """Generate synthetic function completion examples."""
    examples = []
    for func in SYNTHETIC_FUNCTIONS:
        examples.append(format_as_function_completion(func["header"], func["body"]))
    print(f"Synthetic: {len(examples)} examples")
    return examples


def main():
    all_examples = []

    # Collect from all sources
    print("Processing MBPP...")
    all_examples.extend(process_mbpp())

    print("Processing CodeAlpaca...")
    all_examples.extend(process_code_alpaca())

    print("Processing StarCoder...")
    all_examples.extend(process_starcoder())

    print("Processing Synthetic...")
    all_examples.extend(generate_synthetic())

    # Deduplicate by prompt
    seen_prompts = set()
    unique_examples = []
    for ex in all_examples:
        prompt_key = ex['prompt'][:100]
        if prompt_key not in seen_prompts:
            seen_prompts.add(prompt_key)
            unique_examples.append(ex)

    # Filter out very short or very long examples
    filtered = []
    for ex in unique_examples:
        prompt_len = len(ex['prompt'])
        completion_len = len(ex['completion'])
        if 20 < prompt_len < 2000 and 5 < completion_len < 3000:
            filtered.append(ex)

    random.shuffle(filtered)

    print(f"\nTotal unique examples: {len(filtered)}")

    # Save
    output_path = '/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data.jsonl'
    with open(output_path, 'w') as f:
        for ex in filtered:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved to {output_path}")

    # Show a few examples
    print("\n--- Sample examples ---")
    for i in range(min(3, len(filtered))):
        print(f"\nExample {i+1}:")
        print(f"PROMPT: {filtered[i]['prompt'][:200]}...")
        print(f"COMPLETION: {filtered[i]['completion'][:200]}...")


if __name__ == '__main__':
    main()
