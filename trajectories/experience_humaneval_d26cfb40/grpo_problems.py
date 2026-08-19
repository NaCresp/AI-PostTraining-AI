"""
GRPO training with code execution reward for Qwen3-1.7B.
Uses verl or TRL GRPO with a reward function that actually executes generated code.
"""
import json
import re
import os
import signal
import traceback
import multiprocessing
from io import StringIO
import contextlib


def execute_code_with_tests(code: str, test_code: str, timeout: int = 5) -> dict:
    """Execute generated code with test cases in a subprocess. Returns pass/fail."""
    full_code = code + "\n" + test_code

    def _run(code_str, result_queue):
        try:
            exec_globals = {}
            exec(code_str, exec_globals)
            result_queue.put({"passed": True, "error": None})
        except Exception as e:
            result_queue.put({"passed": False, "error": str(e)[:200]})

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run, args=(full_code, result_queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(1)
        return {"passed": False, "error": "timeout"}

    if result_queue.empty():
        return {"passed": False, "error": "no result"}

    return result_queue.get()


def create_grpo_problems():
    """Create programming problems with test cases for GRPO training.
    These are NOT HumanEval problems - they're independent programming exercises.
    """
    problems = [
        {
            "prompt": 'def reverse_words(s: str) -> str:\n    """Reverse the order of words in a string.\n    >>> reverse_words("hello world")\n    \'world hello\'\n    >>> reverse_words("  hello  world  ")\n    \'world hello\'\n    """',
            "test": 'assert reverse_words("hello world") == "world hello"\nassert reverse_words("  hello  world  ") == "world hello"\nassert reverse_words("a") == "a"'
        },
        {
            "prompt": 'def count_vowels(s: str) -> int:\n    """Count vowels in a string (case insensitive).\n    >>> count_vowels("hello")\n    2\n    >>> count_vowels("AEIOU")\n    5\n    """',
            "test": 'assert count_vowels("hello") == 2\nassert count_vowels("AEIOU") == 5\nassert count_vowels("xyz") == 0'
        },
        {
            "prompt": 'def is_anagram(s1: str, s2: str) -> bool:\n    """Check if two strings are anagrams.\n    >>> is_anagram("listen", "silent")\n    True\n    >>> is_anagram("hello", "world")\n    False\n    """',
            "test": 'assert is_anagram("listen", "silent") == True\nassert is_anagram("hello", "world") == False\nassert is_anagram("abc", "cba") == True'
        },
        {
            "prompt": 'def flatten(lst: list) -> list:\n    """Flatten a nested list.\n    >>> flatten([1, [2, 3], [4, [5]]])\n    [1, 2, 3, 4, 5]\n    """',
            "test": 'assert flatten([1, [2, 3], [4, [5]]]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []\nassert flatten([1, 2, 3]) == [1, 2, 3]'
        },
        {
            "prompt": 'def find_duplicates(lst: list) -> list:\n    """Find duplicate elements in a list.\n    >>> sorted(find_duplicates([1, 2, 3, 2, 1]))\n    [1, 2]\n    """',
            "test": 'assert sorted(find_duplicates([1, 2, 3, 2, 1])) == [1, 2]\nassert find_duplicates([1, 2, 3]) == []\nassert sorted(find_duplicates([1, 1, 1])) == [1]'
        },
        {
            "prompt": 'def pascal_triangle_row(n: int) -> list:\n    """Return the nth row of Pascal\'s triangle (0-indexed).\n    >>> pascal_triangle_row(0)\n    [1]\n    >>> pascal_triangle_row(4)\n    [1, 4, 6, 4, 1]\n    """',
            "test": 'assert pascal_triangle_row(0) == [1]\nassert pascal_triangle_row(1) == [1, 1]\nassert pascal_triangle_row(4) == [1, 4, 6, 4, 1]'
        },
        {
            "prompt": 'def roman_to_int(s: str) -> int:\n    """Convert a roman numeral string to integer.\n    >>> roman_to_int("III")\n    3\n    >>> roman_to_int("IX")\n    9\n    >>> roman_to_int("MCMXCIV")\n    1994\n    """',
            "test": 'assert roman_to_int("III") == 3\nassert roman_to_int("IX") == 9\nassert roman_to_int("MCMXCIV") == 1994'
        },
        {
            "prompt": 'def spiral_order(matrix: list) -> list:\n    """Return elements of a matrix in spiral order.\n    >>> spiral_order([[1,2,3],[4,5,6],[7,8,9]])\n    [1, 2, 3, 6, 9, 8, 7, 4, 5]\n    """',
            "test": 'assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]\nassert spiral_order([[1]]) == [1]'
        },
        {
            "prompt": 'def longest_common_subsequence(s1: str, s2: str) -> int:\n    """Find length of longest common subsequence of two strings.\n    >>> longest_common_subsequence("abcde", "ace")\n    3\n    """',
            "test": 'assert longest_common_subsequence("abcde", "ace") == 3\nassert longest_common_subsequence("abc", "def") == 0\nassert longest_common_subsequence("abc", "abc") == 3'
        },
        {
            "prompt": 'def is_balanced_parens(s: str) -> bool:\n    """Check if parentheses in string are balanced. Supports (), [], {}.\n    >>> is_balanced_parens("()[]{}")\n    True\n    >>> is_balanced_parens("([)]")\n    False\n    """',
            "test": 'assert is_balanced_parens("()[]{}") == True\nassert is_balanced_parens("([)]") == False\nassert is_balanced_parens("{[]}") == True\nassert is_balanced_parens("(") == False'
        },
        {
            "prompt": 'def rotate_matrix(matrix: list) -> list:\n    """Rotate a matrix 90 degrees clockwise.\n    >>> rotate_matrix([[1,2],[3,4]])\n    [[3, 1], [4, 2]]\n    """',
            "test": 'assert rotate_matrix([[1,2],[3,4]]) == [[3, 1], [4, 2]]\nassert rotate_matrix([[1]]) == [[1]]'
        },
        {
            "prompt": 'def compress_string(s: str) -> str:\n    """Run-length encode a string. Return original if not shorter.\n    >>> compress_string("aabcccccaaa")\n    \'a2b1c5a3\'\n    """',
            "test": 'assert compress_string("aabcccccaaa") == "a2b1c5a3"\nassert compress_string("abc") == "abc"'
        },
        {
            "prompt": 'def power_set(s: list) -> list:\n    """Return all subsets of a list.\n    >>> sorted([sorted(x) for x in power_set([1,2])])\n    [[], [1], [1, 2], [2]]\n    """',
            "test": 'result = power_set([1,2])\nassert len(result) == 4\nassert [] in result\nassert [1] in result or (1,) in result'
        },
        {
            "prompt": 'def merge_intervals(intervals: list) -> list:\n    """Merge overlapping intervals.\n    >>> merge_intervals([[1,3],[2,6],[8,10],[15,18]])\n    [[1, 6], [8, 10], [15, 18]]\n    """',
            "test": 'assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1, 6], [8, 10], [15, 18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1, 5]]'
        },
        {
            "prompt": 'def valid_sudoku_row(row: list) -> bool:\n    """Check if a sudoku row (list of 9 ints) is valid (no duplicates except 0).\n    >>> valid_sudoku_row([1,2,3,4,5,6,7,8,9])\n    True\n    >>> valid_sudoku_row([1,2,3,4,5,6,7,8,1])\n    False\n    """',
            "test": 'assert valid_sudoku_row([1,2,3,4,5,6,7,8,9]) == True\nassert valid_sudoku_row([1,2,3,4,5,6,7,8,1]) == False\nassert valid_sudoku_row([0,0,1,2,3,4,5,6,7]) == True'
        },
        {
            "prompt": 'def nth_prime(n: int) -> int:\n    """Return the nth prime number (1-indexed).\n    >>> nth_prime(1)\n    2\n    >>> nth_prime(6)\n    13\n    """',
            "test": 'assert nth_prime(1) == 2\nassert nth_prime(2) == 3\nassert nth_prime(6) == 13'
        },
        {
            "prompt": 'def zigzag_convert(s: str, numRows: int) -> str:\n    """Convert string to zigzag pattern and read line by line.\n    >>> zigzag_convert("PAYPALISHIRING", 3)\n    \'PAHNAPLSIIGYIR\'\n    """',
            "test": 'assert zigzag_convert("PAYPALISHIRING", 3) == "PAHNAPLSIIGYIR"\nassert zigzag_convert("A", 1) == "A"'
        },
        {
            "prompt": 'def group_anagrams(strs: list) -> list:\n    """Group anagrams together.\n    >>> result = group_anagrams(["eat","tea","tan","ate","nat","bat"])\n    >>> sorted([sorted(g) for g in result])\n    [[\'ate\', \'eat\', \'tea\'], [\'bat\'], [\'nat\', \'tan\']]\n    """',
            "test": 'result = group_anagrams(["eat","tea","tan","ate","nat","bat"])\nassert len(result) == 3\nassert sorted([sorted(g) for g in result]) == [[\'ate\', \'eat\', \'tea\'], [\'bat\'], [\'nat\', \'tan\']]'
        },
    ]
    return problems


# Test the problems work
if __name__ == '__main__':
    problems = create_grpo_problems()
    print(f"Created {len(problems)} GRPO problems")
    for i, p in enumerate(problems):
        print(f"\nProblem {i}: {p['prompt'][:80]}...")
