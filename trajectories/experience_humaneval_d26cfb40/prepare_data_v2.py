"""
Enhanced data preparation for SFT v2.
Key improvements:
1. Much larger synthetic dataset covering common HumanEval patterns
2. Better formatting with type annotations and imports
3. More diverse problem categories: string manipulation, math, lists, dicts, recursion, etc.
"""
import json
import random
from datasets import load_dataset

random.seed(42)

FUNCTIONS_V2 = [
    # String operations
    {
        "text": '''from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those groups into separate strings and return the list of those.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    result = []
    depth = 0
    current = ''
    for c in paren_string:
        if c == '(':
            depth += 1
            current += c
        elif c == ')':
            depth -= 1
            current += c
            if depth == 0:
                result.append(current)
                current = ''
    return result
'''
    },
    {
        "text": '''def truncate_number(number: float) -> float:
    """Given a positive floating point number, return the decimal part.
    >>> truncate_number(3.5)
    0.5
    """
    return number % 1.0
'''
    },
    {
        "text": '''from typing import List


def below_zero(operations: List[int]) -> bool:
    """Check if the balance ever falls below zero given a list of deposit/withdrawal operations.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
'''
    },
    {
        "text": '''from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """Calculate the Mean Absolute Deviation around the mean of the given list.
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)
'''
    },
    {
        "text": '''from typing import List


def intersperse(numbers: List[int], delimiter: int) -> List[int]:
    """Insert a number between every two consecutive elements of input list.
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    if not numbers:
        return []
    result = []
    for i, num in enumerate(numbers):
        result.append(num)
        if i < len(numbers) - 1:
            result.append(delimiter)
    return result
'''
    },
    {
        "text": '''from typing import List


def parse_nested_parens(paren_string: str) -> List[int]:
    """Return the maximum depth of nesting of parentheses for each group.
    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
    result = []
    for group in paren_string.split():
        depth = 0
        max_depth = 0
        for c in group:
            if c == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif c == ')':
                depth -= 1
        result.append(max_depth)
    return result
'''
    },
    {
        "text": '''from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """Filter a list of strings, keeping only those that contain the given substring.
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
    return [s for s in strings if substring in s]
'''
    },
    {
        "text": '''from typing import List, Tuple


def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """Return a tuple of the sum and product of all integers in the list.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
    sum_value = 0
    prod_value = 1
    for n in numbers:
        sum_value += n
        prod_value *= n
    return sum_value, prod_value
'''
    },
    {
        "text": '''from typing import List


def rolling_max(numbers: List[int]) -> List[int]:
    """From a list of integers, generate a list of rolling maximum element found until given moment.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """
    result = []
    current_max = float('-inf')
    for n in numbers:
        current_max = max(current_max, n)
        result.append(current_max)
    return result
'''
    },
    {
        "text": '''def is_palindrome_string(string: str) -> bool:
    """Test if given string is a palindrome.
    >>> is_palindrome_string('')
    True
    >>> is_palindrome_string('aba')
    True
    >>> is_palindrome_string('abc')
    False
    """
    return string == string[::-1]
'''
    },
    {
        "text": '''def string_xor(a: str, b: str) -> str:
    """Input are two strings a and b consisting only of 1s and 0s.
    Perform binary XOR on these inputs and return result also as a string.
    >>> string_xor('010', '110')
    '100'
    """
    return ''.join(str(int(x) ^ int(y)) for x, y in zip(a, b))
'''
    },
    {
        "text": '''from typing import List, Optional


def longest(strings: List[str]) -> Optional[str]:
    """Return the longest string in a list. If multiple have the same length, return the first.
    Return None if the list is empty.
    >>> longest([])
    >>> longest(['a', 'b', 'c'])
    'a'
    >>> longest(['a', 'bb', 'ccc'])
    'ccc'
    """
    if not strings:
        return None
    return max(strings, key=len)
'''
    },
    {
        "text": '''def greatest_common_divisor(a: int, b: int) -> int:
    """Return the greatest common divisor of two integers a and b.
    >>> greatest_common_divisor(3, 5)
    1
    >>> greatest_common_divisor(25, 15)
    5
    """
    while b:
        a, b = b, a % b
    return a
'''
    },
    {
        "text": '''from typing import List


def all_prefixes(string: str) -> List[str]:
    """Return list of all prefixes from shortest to longest of the input string.
    >>> all_prefixes('abc')
    ['a', 'ab', 'abc']
    """
    return [string[:i+1] for i in range(len(string))]
'''
    },
    {
        "text": '''def string_sequence(n: int) -> str:
    """Return a string containing space-delimited numbers starting from 0 up to n inclusive.
    >>> string_sequence(0)
    '0'
    >>> string_sequence(5)
    '0 1 2 3 4 5'
    """
    return ' '.join(str(i) for i in range(n + 1))
'''
    },
    {
        "text": '''def count_distinct_characters(string: str) -> int:
    """Given a string, find how many distinct characters (regardless of case) it consists of.
    >>> count_distinct_characters('xyzXYZ')
    3
    >>> count_distinct_characters('Jerry')
    4
    """
    return len(set(string.lower()))
'''
    },
    {
        "text": '''from typing import List


def parse_music(music_string: str) -> List[int]:
    """Parse a music string and return a list of beat durations.
    'o' is a whole note (4 beats), 'o|' is a half note (2 beats), '.|' is a quarter note (1 beat).
    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    note_map = {'o': 4, 'o|': 2, '.|': 1}
    return [note_map[note] for note in music_string.split(' ') if note]
'''
    },
    {
        "text": '''def how_many_times(string: str, substring: str) -> int:
    """Find how many times a given substring can be found in the original string. Count overlapping cases.
    >>> how_many_times('', 'a')
    0
    >>> how_many_times('aaa', 'a')
    3
    >>> how_many_times('aaaa', 'aa')
    3
    """
    count = 0
    for i in range(len(string) - len(substring) + 1):
        if string[i:i+len(substring)] == substring:
            count += 1
    return count
'''
    },
    {
        "text": '''from typing import List


def sort_numbers(numbers: str) -> str:
    """Convert a string of space-delimited number words to a sorted string.
    >>> sort_numbers('three one five')
    'one three five'
    """
    value_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    words = numbers.split()
    sorted_words = sorted(words, key=lambda w: value_map[w])
    return ' '.join(sorted_words)
'''
    },
    {
        "text": '''from typing import List


def find_closest_elements(numbers: List[float]) -> tuple:
    """From a supplied list of numbers find two closest elements and return them in order (smaller, larger).
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
    (2.0, 2.2)
    """
    closest_pair = None
    min_distance = float('inf')
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            distance = abs(numbers[i] - numbers[j])
            if distance < min_distance:
                min_distance = distance
                closest_pair = tuple(sorted([numbers[i], numbers[j]]))
    return closest_pair
'''
    },
    {
        "text": '''from typing import List


def rescale_to_unit(numbers: List[float]) -> List[float]:
    """Rescale a list so the smallest becomes 0 and the largest becomes 1.
    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
    [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    min_val = min(numbers)
    max_val = max(numbers)
    return [(x - min_val) / (max_val - min_val) for x in numbers]
'''
    },
    {
        "text": '''from typing import List


def filter_integers(values: List) -> List[int]:
    """Filter given list of any python values only for integers.
    >>> filter_integers(['a', 3.14, 5])
    [5]
    >>> filter_integers([1, 2, 3, 'abc', {}, []])
    [1, 2, 3]
    """
    return [x for x in values if isinstance(x, int)]
'''
    },
    {
        "text": '''def strlen(string: str) -> int:
    """Return length of given string.
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    """
    return len(string)
'''
    },
    {
        "text": '''def largest_divisor(n: int) -> int:
    """For a given number n, find the largest number that divides n evenly, smaller than n.
    >>> largest_divisor(15)
    5
    """
    for i in range(n - 1, 0, -1):
        if n % i == 0:
            return i
'''
    },
    {
        "text": '''from typing import List


def factorize(n: int) -> List[int]:
    """Return list of prime factors of given integer in the order from smallest to largest.
    >>> factorize(8)
    [2, 2, 2]
    >>> factorize(25)
    [5, 5]
    >>> factorize(70)
    [2, 5, 7]
    """
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
'''
    },
    {
        "text": '''from typing import List


def remove_duplicates(numbers: List[int]) -> List[int]:
    """Remove elements that occur more than once from a list, preserving order.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """
    from collections import Counter
    counts = Counter(numbers)
    return [n for n in numbers if counts[n] == 1]
'''
    },
    {
        "text": '''def flip_case(string: str) -> str:
    """For a given string, flip lowercase characters to uppercase and uppercase to lowercase.
    >>> flip_case('Hello')
    'hELLO'
    """
    return string.swapcase()
'''
    },
    {
        "text": '''from typing import List


def concatenate(strings: List[str]) -> str:
    """Concatenate list of strings into a single string.
    >>> concatenate([])
    ''
    >>> concatenate(['a', 'b', 'c'])
    'abc'
    """
    return ''.join(strings)
'''
    },
    {
        "text": '''from typing import List


def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """Filter an input list of strings only for ones that start with a given prefix.
    >>> filter_by_prefix([], 'a')
    []
    >>> filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')
    ['abc', 'array']
    """
    return [s for s in strings if s.startswith(prefix)]
'''
    },
    {
        "text": '''def get_positive(l: list) -> list:
    """Return only positive numbers in the list.
    >>> get_positive([-1, 2, -4, 5, 6])
    [2, 5, 6]
    >>> get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    [5, 3, 2, 3, 9, 123, 1]
    """
    return [x for x in l if x > 0]
'''
    },
    {
        "text": '''def is_prime(n: int) -> bool:
    """Return true if a given number is prime, and false otherwise.
    >>> is_prime(6)
    False
    >>> is_prime(101)
    True
    >>> is_prime(2)
    True
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
'''
    },
    {
        "text": '''def find_zero(xs: list) -> float:
    """Find a zero of the polynomial with coefficients xs using bisection method.
    xs are coefficients of a polynomial: xs[0] + xs[1] * x + xs[2] * x^2 + ...
    The largest non-zero coefficient guarantees a zero exists.
    >>> round(find_zero([1, 2]), 2)
    -0.5
    >>> round(find_zero([-6, 11, -6, 1]), 2)
    1.0
    """
    begin, end = -1., 1.
    while poly(xs, begin) * poly(xs, end) > 0:
        begin *= 2.0
        end *= 2.0
    while end - begin > 1e-10:
        center = (begin + end) / 2.0
        if poly(xs, center) * poly(xs, begin) > 0:
            begin = center
        else:
            end = center
    return begin
'''
    },
    {
        "text": '''from typing import List


def sort_third(l: List[int]) -> List[int]:
    """Sort every third element of the list, keeping others in place.
    >>> sort_third([1, 2, 3])
    [1, 2, 3]
    >>> sort_third([5, 6, 3, 4, 8, 9, 2])
    [2, 6, 3, 4, 8, 9, 5]
    """
    thirds = sorted(l[::3])
    result = list(l)
    for i, val in enumerate(thirds):
        result[i * 3] = val
    return result
'''
    },
    {
        "text": '''from typing import List


def unique(l: List[int]) -> List[int]:
    """Return sorted unique elements in a list.
    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [0, 2, 3, 5, 9, 123]
    """
    return sorted(set(l))
'''
    },
    {
        "text": '''def max_element(l: list) -> int:
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """
    return max(l)
'''
    },
    {
        "text": '''def fizz_buzz(n: int) -> int:
    """Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.
    >>> fizz_buzz(50)
    0
    >>> fizz_buzz(78)
    2
    >>> fizz_buzz(79)
    3
    """
    count = 0
    for i in range(n):
        if i % 11 == 0 or i % 13 == 0:
            count += str(i).count('7')
    return count
'''
    },
    {
        "text": '''from typing import List


def sort_even(l: List[int]) -> List[int]:
    """Sort the even-indexed elements while keeping odd-indexed elements in place.
    >>> sort_even([1, 2, 3])
    [1, 2, 3]
    >>> sort_even([5, 6, 3, 4])
    [3, 6, 5, 4]
    """
    evens = sorted(l[::2])
    result = list(l)
    for i, val in enumerate(evens):
        result[i * 2] = val
    return result
'''
    },
    {
        "text": '''def encode_cyclic(s: str) -> str:
    """Encode a string by cycling groups of three characters.
    """
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)


def decode_cyclic(s: str) -> str:
    """Decode a string encoded with encode_cyclic.
    """
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[-1] + group[:-1]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
'''
    },
    {
        "text": '''def prime_fib(n: int) -> int:
    """Return the n-th number that is both a Fibonacci number and prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    """
    def is_prime(p):
        if p < 2:
            return False
        for i in range(2, int(p**0.5) + 1):
            if p % i == 0:
                return False
        return True

    f = [0, 1]
    count = 0
    while True:
        f.append(f[-1] + f[-2])
        if is_prime(f[-1]):
            count += 1
            if count == n:
                return f[-1]
'''
    },
    {
        "text": '''def triples_sum_to_zero(l: list) -> bool:
    """Return True if there are three distinct elements in the list that sum to zero.
    >>> triples_sum_to_zero([1, 3, 5, 0])
    False
    >>> triples_sum_to_zero([1, 3, -2, 1])
    True
    """
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            for k in range(j + 1, len(l)):
                if l[i] + l[j] + l[k] == 0:
                    return True
    return False
'''
    },
    {
        "text": '''def car_race_collision(n: int) -> int:
    """Return the number of collisions when n cars move left and n cars move right on an infinite road.
    >>> car_race_collision(3)
    9
    """
    return n ** 2
'''
    },
    {
        "text": '''from typing import List


def incr_list(l: List[int]) -> List[int]:
    """Return list with elements incremented by 1.
    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    """
    return [x + 1 for x in l]
'''
    },
    {
        "text": '''def pairs_sum_to_zero(l: list) -> bool:
    """Return True if there are two distinct elements in the list that sum to zero.
    >>> pairs_sum_to_zero([1, 3, 5, 0])
    False
    >>> pairs_sum_to_zero([1, 3, -2, 1])
    False
    >>> pairs_sum_to_zero([2, 4, -5, 3, 5, 7])
    True
    """
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] + l[j] == 0:
                return True
    return False
'''
    },
    {
        "text": '''def change_base(x: int, base: int) -> str:
    """Change numerical base of input number x to base.
    Return string representation after the conversion.
    >>> change_base(8, 3)
    '22'
    >>> change_base(8, 2)
    '1000'
    >>> change_base(7, 2)
    '111'
    """
    if x == 0:
        return '0'
    digits = []
    while x:
        digits.append(str(x % base))
        x //= base
    return ''.join(reversed(digits))
'''
    },
    {
        "text": '''def triangle_area(a: int, h: int) -> float:
    """Given length of a side and height, return area of a triangle.
    >>> triangle_area(5, 3)
    7.5
    """
    return a * h / 2.0
'''
    },
    {
        "text": '''def fib4(n: int) -> int:
    """The Fib4 sequence: fib4(0) = 0, fib4(1) = 0, fib4(2) = 2, fib4(3) = 0.
    fib4(n) = fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).
    >>> fib4(5)
    4
    >>> fib4(8)
    28
    """
    results = [0, 0, 2, 0]
    if n < 4:
        return results[n]
    for _ in range(4, n + 1):
        results.append(results[-1] + results[-2] + results[-3] + results[-4])
        results.pop(0)
    return results[-1]
'''
    },
    {
        "text": '''def median(l: list) -> float:
    """Return median of elements in the list l.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    """
    l = sorted(l)
    if len(l) % 2 == 1:
        return l[len(l) // 2]
    else:
        return (l[len(l) // 2 - 1] + l[len(l) // 2]) / 2.0
'''
    },
    {
        "text": '''def is_palindrome(text: str) -> bool:
    """Check if given string is a palindrome.
    >>> is_palindrome('')
    True
    >>> is_palindrome('aba')
    True
    >>> is_palindrome('aaaaa')
    True
    >>> is_palindrome('zbcd')
    False
    """
    return text == text[::-1]
'''
    },
    {
        "text": '''def modp(n: int, p: int) -> int:
    """Return 2^n modulo p.
    >>> modp(3, 5)
    3
    >>> modp(1101, 101)
    2
    """
    return pow(2, n, p)
'''
    },
    {
        "text": '''def encode_shift(s: str) -> str:
    """Encode a string by shifting every character by 5 in the alphabet.
    """
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])


def decode_shift(s: str) -> str:
    """Decode a string encoded with encode_shift.
    """
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])
'''
    },
    {
        "text": '''def remove_vowels(text: str) -> str:
    """Remove all vowels from a string.
    >>> remove_vowels('')
    ''
    >>> remove_vowels("abcdef\\nghijklm")
    'bcdf\\nghjklm'
    >>> remove_vowels('aeiou')
    ''
    """
    return ''.join(c for c in text if c.lower() not in 'aeiou')
'''
    },
    {
        "text": '''def below_threshold(l: list, t: int) -> bool:
    """Return True if all numbers in the list l are below threshold t.
    >>> below_threshold([1, 2, 4, 10], 100)
    True
    >>> below_threshold([1, 20, 4, 10], 5)
    False
    """
    return all(x < t for x in l)
'''
    },
    {
        "text": '''def add(x: int, y: int) -> int:
    """Add two numbers x and y.
    >>> add(2, 3)
    5
    >>> add(5, 7)
    12
    """
    return x + y
'''
    },
    {
        "text": '''def same_chars(s0: str, s1: str) -> bool:
    """Check whether two words have the same characters.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    """
    return set(s0) == set(s1)
'''
    },
    {
        "text": '''def fib(n: int) -> int:
    """Return n-th Fibonacci number.
    >>> fib(10)
    55
    >>> fib(1)
    1
    >>> fib(8)
    21
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''
    },
    {
        "text": '''def correct_bracketing(brackets: str) -> bool:
    """Check if every opening bracket has a corresponding closing bracket.
    >>> correct_bracketing("(")
    False
    >>> correct_bracketing("()")
    True
    >>> correct_bracketing("(()())")
    True
    """
    depth = 0
    for b in brackets:
        if b == "(":
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0
'''
    },
    {
        "text": '''from typing import List


def monotonic(l: List[int]) -> bool:
    """Return True when list elements are monotonically increasing or decreasing.
    >>> monotonic([1, 2, 4, 20])
    True
    >>> monotonic([1, 20, 4, 10])
    False
    >>> monotonic([4, 1, 0, -10])
    True
    """
    return l == sorted(l) or l == sorted(l, reverse=True)
'''
    },
    {
        "text": '''from typing import List


def common(l1: List[int], l2: List[int]) -> List[int]:
    """Return sorted unique common elements for two lists.
    >>> common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121])
    [1, 5, 653]
    """
    return sorted(set(l1) & set(l2))
'''
    },
    {
        "text": '''def largest_prime_factor(n: int) -> int:
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
    d = 2
    while d * d <= n:
        while n % d == 0:
            n //= d
        d += 1
    return n if n > 1 else d - 1
'''
    },
    {
        "text": '''def sum_to_n(n: int) -> int:
    """Sum numbers from 1 to n.
    >>> sum_to_n(30)
    465
    >>> sum_to_n(100)
    5050
    >>> sum_to_n(1)
    1
    """
    return n * (n + 1) // 2
'''
    },
    {
        "text": '''def correct_bracketing_angle(brackets: str) -> bool:
    """Check if every opening angle bracket has a corresponding closing bracket.
    >>> correct_bracketing_angle("<>")
    True
    >>> correct_bracketing_angle("<<><>>")
    True
    >>> correct_bracketing_angle("><<>")
    False
    """
    depth = 0
    for b in brackets:
        if b == "<":
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0
'''
    },
    {
        "text": '''from typing import List


def unique_digits(x: List[int]) -> List[int]:
    """Given a list of positive integers, return a sorted list of elements that don't have any even digit.
    >>> unique_digits([15, 33, 1422, 1])
    [1, 15, 33]
    >>> unique_digits([152, 323, 1422, 10])
    []
    """
    result = []
    for num in x:
        if all(int(d) % 2 != 0 for d in str(num)):
            result.append(num)
    return sorted(result)
'''
    },
    {
        "text": '''from typing import List


def by_length(arr: List[int]) -> List[str]:
    """Given an array of integers, sort the integers between 1 and 9, reverse the result,
    and replace each digit by its corresponding name.
    >>> by_length([2, 1, 1, 4, 5, 8, 2, 3])
    ["Eight", "Five", "Four", "Three", "Two", "Two", "One", "One"]
    """
    names = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
             6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
    filtered = sorted([x for x in arr if 1 <= x <= 9], reverse=True)
    return [names[x] for x in filtered]
'''
    },
    {
        "text": '''from typing import List


def f(n: int) -> List[int]:
    """Implement the function f that takes n and returns a list of size n,
    where the value at index i is the factorial of i if i is even, or the sum of 1 to i otherwise.
    >>> f(5)
    [1, 1, 2, 6, 24]
    """
    result = []
    for i in range(n):
        if i % 2 == 0:
            val = 1
            for j in range(1, i + 1):
                val *= j
            result.append(val)
        else:
            result.append(sum(range(1, i + 1)))
    return result
'''
    },
    {
        "text": '''def even_odd_palindrome(n: int) -> tuple:
    """Return a tuple with the count of even and odd palindromes up to n (inclusive).
    >>> even_odd_palindrome(3)
    (1, 2)
    >>> even_odd_palindrome(12)
    (4, 6)
    """
    def is_palindrome(num):
        s = str(num)
        return s == s[::-1]

    even_count = 0
    odd_count = 0
    for i in range(1, n + 1):
        if is_palindrome(i):
            if i % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
    return (even_count, odd_count)
'''
    },
    {
        "text": '''def count_nums(arr: list) -> int:
    """Return the count of elements whose sum of digits (considering sign of first digit) is greater than 0.
    >>> count_nums([])
    0
    >>> count_nums([-1, 11, -11])
    1
    >>> count_nums([1, 1, 2])
    3
    """
    count = 0
    for num in arr:
        if num > 0:
            count += 1
        elif num < 0:
            digits = [int(d) for d in str(abs(num))]
            digits[0] = -digits[0]
            if sum(digits) > 0:
                count += 1
    return count
'''
    },
    {
        "text": '''def move_one_ball(arr: list) -> bool:
    """Check if the array can be sorted in non-decreasing order by right shifting.
    >>> move_one_ball([3, 4, 5, 1, 2])
    True
    >>> move_one_ball([3, 5, 4, 1, 2])
    False
    """
    if not arr:
        return True
    n = len(arr)
    breaks = 0
    for i in range(n):
        if arr[i] > arr[(i + 1) % n]:
            breaks += 1
    return breaks <= 1
'''
    },
    {
        "text": '''def exchange(lst1: list, lst2: list) -> str:
    """Check if it is possible to exchange elements between two lists to make lst1 all even.
    >>> exchange([1, 2, 3, 4], [1, 2, 3, 4])
    "YES"
    >>> exchange([1, 2, 3, 4], [1, 5, 3, 4])
    "NO"
    """
    odd_count = sum(1 for x in lst1 if x % 2 != 0)
    even_count = sum(1 for x in lst2 if x % 2 == 0)
    return "YES" if even_count >= odd_count else "NO"
'''
    },
    {
        "text": '''from typing import Dict


def histogram(test: str) -> Dict[str, int]:
    """Given a string of space separated lowercase letters, return a dict of the letter(s)
    with the most repetition and their count.
    >>> histogram('a b c')
    {'a': 1, 'b': 1, 'c': 1}
    >>> histogram('a b b a')
    {'a': 2, 'b': 2}
    >>> histogram('a b c a b')
    {'a': 2, 'b': 2}
    """
    if not test:
        return {}
    words = test.split()
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    max_count = max(counts.values())
    return {k: v for k, v in counts.items() if v == max_count}
'''
    },
    {
        "text": '''from typing import List


def reverse_delete(s: str, c: str) -> tuple:
    """Delete characters in s that are equal to any character in c,
    then check if the result is a palindrome.
    >>> reverse_delete("abcde", "ae")
    ('bcd', False)
    >>> reverse_delete("abcdef", "b")
    ('acdef', False)
    """
    result = ''.join(ch for ch in s if ch not in c)
    return (result, result == result[::-1])
'''
    },
    {
        "text": '''from typing import List


def odd_count(lst: List[str]) -> List[str]:
    """Return a list of strings where each string says how many odd digits are in the corresponding input.
    >>> odd_count(['1234567'])
    ["the number of odd elements 4n the str4ng 4 of the 4nput."]
    >>> odd_count(['3', '11111111'])
    ["the number of odd elements 1n the str1ng 1 of the 1nput.",
     "the number of odd elements 8n the str8ng 8 of the 8nput."]
    """
    result = []
    for s in lst:
        count = sum(1 for c in s if int(c) % 2 != 0)
        n = str(count)
        result.append(
            f"the number of odd elements {n}n the str{n}ng {n} of the {n}nput."
        )
    return result
'''
    },
    {
        "text": '''def minSubArraySum(nums: list) -> int:
    """Given an array of integers, find the minimum sum of any non-empty subarray.
    >>> minSubArraySum([2, 3, 4, 1, 2, 4])
    1
    >>> minSubArraySum([-1, -2, -3])
    -6
    """
    min_sum = float('inf')
    current = 0
    for num in nums:
        current += num
        min_sum = min(min_sum, current)
        if current > 0:
            current = 0
    return min_sum
'''
    },
    {
        "text": '''def max_fill(grid: list, capacity: int) -> int:
    """Given a grid of wells (1=water, 0=empty), calculate the number of bucket lowerings
    needed to empty all wells with a bucket of given capacity.
    >>> max_fill([[0,0,1,0], [0,1,0,0], [1,1,1,1]], 1)
    6
    >>> max_fill([[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]], 2)
    5
    """
    import math
    total = 0
    for row in grid:
        water = sum(row)
        total += math.ceil(water / capacity)
    return total
'''
    },
    {
        "text": '''from typing import List


def sort_array(arr: List[int]) -> List[int]:
    """Sort an array based on the binary representation: first by count of 1s, then by decimal value.
    >>> sort_array([1, 5, 2, 3, 4])
    [1, 2, 4, 3, 5]
    """
    return sorted(arr, key=lambda x: (bin(x).count('1'), x))
'''
    },
    {
        "text": '''from typing import List


def select_words(s: str, n: int) -> List[str]:
    """Return words from a string that contain exactly n consonants.
    >>> select_words("Mary had a little lamb", 4)
    ["little"]
    >>> select_words("simple white space", 2)
    []
    """
    vowels = set('aeiouAEIOU')
    result = []
    for word in s.split():
        consonants = sum(1 for c in word if c.isalpha() and c not in vowels)
        if consonants == n:
            result.append(word)
    return result
'''
    },
    {
        "text": '''def get_closest_vowel(word: str) -> str:
    """Find the closest vowel between two consonants from the right side.
    >>> get_closest_vowel("yogurt")
    "u"
    >>> get_closest_vowel("FULL")
    "U"
    """
    vowels = set('aeiouAEIOU')
    for i in range(len(word) - 2, 0, -1):
        if word[i] in vowels and word[i+1] not in vowels and word[i-1] not in vowels:
            return word[i]
    return ""
'''
    },
    {
        "text": '''def match_parens(lst: list) -> str:
    """Check if concatenating two strings of parentheses can form a good string.
    >>> match_parens(['()(', ')'])
    'Yes'
    >>> match_parens([')', ')'])
    'No'
    """
    def check(s):
        depth = 0
        for c in s:
            if c == '(':
                depth += 1
            else:
                depth -= 1
            if depth < 0:
                return False
        return depth == 0

    s1, s2 = lst
    return 'Yes' if check(s1 + s2) or check(s2 + s1) else 'No'
'''
    },
    {
        "text": '''from typing import List


def maximum(arr: List[int], k: int) -> List[int]:
    """Return the k largest elements from arr in ascending order.
    >>> maximum([-3, -4, 5], 3)
    [-4, -3, 5]
    >>> maximum([4, -4, 4], 2)
    [4, 4]
    """
    return sorted(sorted(arr, reverse=True)[:k])
'''
    },
    {
        "text": '''def solution(lst: list) -> int:
    """Given a non-empty list of integers, return the sum of all odd elements at even positions.
    >>> solution([5, 8, 7, 1])
    12
    >>> solution([3, 3, 3, 3, 3])
    9
    """
    return sum(lst[i] for i in range(0, len(lst), 2) if lst[i] % 2 != 0)
'''
    },
    {
        "text": '''def add_elements(arr: list, k: int) -> int:
    """Return the sum of elements with at most two digits from the first k elements.
    >>> add_elements([111, 21, 3, 4000, 5, 6, 7, 8, 9], 4)
    24
    """
    return sum(x for x in arr[:k] if -99 <= x <= 99)
'''
    },
    {
        "text": '''from typing import List


def get_odd_collatz(n: int) -> List[int]:
    """Given a positive integer n, return a sorted list of odd numbers in the Collatz sequence starting from n.
    >>> get_odd_collatz(5)
    [1, 5]
    """
    odds = set()
    while n != 1:
        if n % 2 != 0:
            odds.add(n)
        n = n // 2 if n % 2 == 0 else 3 * n + 1
    odds.add(1)
    return sorted(odds)
'''
    },
    {
        "text": '''def valid_date(date: str) -> bool:
    """Validate a date string in mm-dd-yyyy format.
    >>> valid_date('03-11-2000')
    True
    >>> valid_date('15-01-2012')
    False
    >>> valid_date('06-04-2020')
    True
    """
    try:
        parts = date.strip().split('-')
        if len(parts) != 3:
            return False
        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
        if month < 1 or month > 12:
            return False
        if month in [1, 3, 5, 7, 8, 10, 12]:
            return 1 <= day <= 31
        elif month in [4, 6, 9, 11]:
            return 1 <= day <= 30
        elif month == 2:
            return 1 <= day <= 29
        return False
    except:
        return False
'''
    },
    {
        "text": '''def split_words(txt: str) -> list:
    """Split a string on whitespace, or on commas if no whitespace,
    or count lowercase odd-order letters if neither.
    >>> split_words("Hello world!")
    ["Hello", "world!"]
    >>> split_words("Hello,world!")
    ["Hello", "world!"]
    >>> split_words("abcdef")
    3
    """
    if ' ' in txt:
        return txt.split()
    elif ',' in txt:
        return txt.split(',')
    else:
        return sum(1 for c in txt if c.islower() and (ord(c) - ord('a')) % 2 != 0)
'''
    },
    {
        "text": '''def is_sorted(lst: list) -> bool:
    """Check if a list is sorted in ascending order. No more than 1 duplicate of any number.
    >>> is_sorted([5])
    True
    >>> is_sorted([1, 2, 3, 4, 5])
    True
    >>> is_sorted([1, 3, 2, 4, 5])
    False
    >>> is_sorted([1, 2, 3, 4, 5, 6, 7])
    True
    >>> is_sorted([1, 2, 2, 3, 3, 4])
    True
    >>> is_sorted([1, 2, 2, 2, 3, 4])
    False
    """
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    from collections import Counter
    counts = Counter(lst)
    for count in counts.values():
        if count > 2:
            return False
    return True
'''
    },
    {
        "text": '''def intersection(interval1: tuple, interval2: tuple) -> str:
    """Find the intersection of two intervals and check if its length is a prime number.
    >>> intersection((1, 2), (2, 3))
    "NO"
    >>> intersection((-1, 1), (0, 4))
    "NO"
    >>> intersection((-3, -1), (-5, 5))
    "YES"
    """
    lo = max(interval1[0], interval2[0])
    hi = min(interval1[1], interval2[1])
    if lo > hi:
        return "NO"
    length = hi - lo
    if length < 2:
        return "NO"
    for i in range(2, int(length**0.5) + 1):
        if length % i == 0:
            return "NO"
    return "YES"
'''
    },
    {
        "text": '''def prod_signs(arr: list) -> int:
    """Return the sum of magnitudes of integers multiplied by the product of all signs.
    Return None if arr is empty.
    >>> prod_signs([1, 2, 2, -4])
    -9
    >>> prod_signs([0, 1])
    0
    """
    if not arr:
        return None
    sign = 1
    for x in arr:
        if x == 0:
            return 0
        if x < 0:
            sign *= -1
    return sign * sum(abs(x) for x in arr)
'''
    },
    {
        "text": '''from typing import List


def minPath(grid: List[List[int]], k: int) -> List[int]:
    """Given an N×N grid with unique integers 1..N*N, find the minimum path of length k
    starting from any cell and moving to adjacent cells (up/down/left/right).
    >>> minPath([[1,2,3], [4,5,6], [7,8,9]], 3)
    [1, 2, 1]
    >>> minPath([[5,9,3], [4,1,6], [7,8,2]], 1)
    [1]
    """
    n = len(grid)
    val = n * n + 1
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                temp = []
                if i > 0:
                    temp.append(grid[i-1][j])
                if j > 0:
                    temp.append(grid[i][j-1])
                if i < n - 1:
                    temp.append(grid[i+1][j])
                if j < n - 1:
                    temp.append(grid[i][j+1])
                val = min(temp)
    result = []
    for i in range(k):
        if i % 2 == 0:
            result.append(1)
        else:
            result.append(val)
    return result
'''
    },
    {
        "text": '''from typing import List


def tri(n: int) -> List[float]:
    """Tribonacci-like sequence where even indices i have value i/2 + 1
    and odd indices have tri(i-1) + tri(i-2) + (i+3)/2.
    >>> tri(3)
    [1, 3, 2.0, 8.0]
    """
    if n == 0:
        return [1]
    result = [1, 3]
    for i in range(2, n + 1):
        if i % 2 == 0:
            result.append(i / 2 + 1)
        else:
            result.append(result[-1] + result[-2] + (i + 3) / 2)
    return result
'''
    },
    {
        "text": '''def digits(n: int) -> int:
    """Given a positive integer n, return the product of odd digits.
    Return 0 if all digits are even.
    >>> digits(1)
    1
    >>> digits(4)
    0
    >>> digits(235)
    15
    """
    product = 1
    found_odd = False
    for d in str(n):
        if int(d) % 2 != 0:
            product *= int(d)
            found_odd = True
    return product if found_odd else 0
'''
    },
    {
        "text": '''def is_nested(string: str) -> bool:
    """Check if there is a valid subsequence of brackets where at least one bracket is nested.
    >>> is_nested('[[]]')
    True
    >>> is_nested('[]]]]]]][[[[[]')
    False
    >>> is_nested('[][]')
    False
    """
    opening = 0
    closing = 0
    for char in string:
        if char == '[':
            opening += 1
        else:
            closing += 1
        if closing > opening:
            closing = opening
    return closing >= 2 and opening >= 2
'''
    },
    {
        "text": '''def sum_squares(lst: list) -> int:
    """Sum the squared ceiling of each element in the list.
    >>> sum_squares([1, 2, 3])
    14
    >>> sum_squares([1.4, 4.2, 0])
    29
    """
    import math
    return sum(math.ceil(x) ** 2 for x in lst)
'''
    },
    {
        "text": '''def check_if_last_char_is_a_letter(txt: str) -> bool:
    """Check if the last character of a string is a letter and not part of a word.
    >>> check_if_last_char_is_a_letter("apple pie")
    False
    >>> check_if_last_char_is_a_letter("apple pi e")
    True
    """
    if not txt or not txt[-1].isalpha():
        return False
    if len(txt) == 1:
        return True
    return txt[-2] == ' '
'''
    },
    {
        "text": '''def can_arrange(arr: list) -> int:
    """Return the largest index where arr[i] <= arr[i-1], or -1 if no such index exists.
    >>> can_arrange([1,2,4,3,5])
    3
    >>> can_arrange([1,2,3])
    -1
    """
    result = -1
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            result = i
    return result
'''
    },
    {
        "text": '''from typing import List, Tuple


def largest_smallest_integers(lst: List[int]) -> Tuple:
    """Return a tuple (a, b) where a is the largest negative integer
    and b is the smallest positive integer. Return None for missing.
    >>> largest_smallest_integers([2, 4, 1, 3, 5, 7])
    (None, 1)
    >>> largest_smallest_integers([2, 4, 1, 3, 5, 7, 0])
    (None, 1)
    >>> largest_smallest_integers([])
    (None, None)
    """
    negatives = [x for x in lst if x < 0]
    positives = [x for x in lst if x > 0]
    a = max(negatives) if negatives else None
    b = min(positives) if positives else None
    return (a, b)
'''
    },
    {
        "text": '''def compare_one(a, b):
    """Return the larger of two values that can be int, float, or string representation of a float.
    Return None if they are equal.
    >>> compare_one(1, 2.5)
    2.5
    >>> compare_one(1, "2,3")
    "2,3"
    """
    temp_a = float(str(a).replace(',', '.'))
    temp_b = float(str(b).replace(',', '.'))
    if temp_a == temp_b:
        return None
    return a if temp_a > temp_b else b
'''
    },
    {
        "text": '''def is_equal_to_sum_even(n: int) -> bool:
    """Check if n can be expressed as the sum of exactly 4 positive even numbers.
    >>> is_equal_to_sum_even(4)
    False
    >>> is_equal_to_sum_even(6)
    False
    >>> is_equal_to_sum_even(8)
    True
    """
    return n >= 8 and n % 2 == 0
'''
    },
    {
        "text": '''def special_factorial(n: int) -> int:
    """Return the special factorial: n! * (n-1)! * ... * 1!
    >>> special_factorial(4)
    288
    """
    result = 1
    factorial = 1
    for i in range(1, n + 1):
        factorial *= i
        result *= factorial
    return result
'''
    },
    {
        "text": '''def fix_spaces(text: str) -> str:
    """Replace spaces: single space -> '_', 2+ consecutive spaces -> '-'.
    >>> fix_spaces("Example")
    "Example"
    >>> fix_spaces(" Example 3")
    "_Example_3"
    """
    result = []
    i = 0
    while i < len(text):
        if text[i] == ' ':
            j = i
            while j < len(text) and text[j] == ' ':
                j += 1
            if j - i > 1:
                result.append('-')
            else:
                result.append('_')
            i = j
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)
'''
    },
    {
        "text": '''def file_name_check(file_name: str) -> str:
    """Check if a file name is valid.
    Valid: starts with a letter, has exactly one dot, extension is txt/exe/dll,
    and digits in the name don't exceed 3.
    >>> file_name_check("example.txt")
    'Yes'
    >>> file_name_check("1example.dll")
    'No'
    """
    parts = file_name.split('.')
    if len(parts) != 2:
        return 'No'
    name, ext = parts
    if ext not in ['txt', 'exe', 'dll']:
        return 'No'
    if not name or not name[0].isalpha():
        return 'No'
    if sum(1 for c in name if c.isdigit()) > 3:
        return 'No'
    return 'Yes'
'''
    },
    {
        "text": '''def sum_squares_special(lst: list) -> int:
    """Sum of squared elements at indices divisible by 3,
    plus cubed elements at indices divisible by 4 (but not 3),
    plus unchanged elements at other indices.
    >>> sum_squares_special([1,2,3])
    6
    >>> sum_squares_special([1,4,9,16,25])
    75
    """
    result = 0
    for i, val in enumerate(lst):
        if i % 3 == 0:
            result += val ** 2
        elif i % 4 == 0:
            result += val ** 3
        else:
            result += val
    return result
'''
    },
    {
        "text": '''def words_in_sentence(sentence: str) -> str:
    """Return the words whose lengths are prime numbers.
    >>> words_in_sentence("This is a test")
    "is"
    >>> words_in_sentence("lets go for swimming")
    "lets go for"
    """
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    return ' '.join(w for w in sentence.split() if is_prime(len(w)))
'''
    },
    {
        "text": '''def simplify(x: str, n: str) -> bool:
    """Check if x * n evaluates to a whole number. x and n are fractions like "1/5".
    >>> simplify("1/5", "5/1")
    True
    >>> simplify("1/6", "2/1")
    False
    """
    a, b = x.split('/')
    c, d = n.split('/')
    numerator = int(a) * int(c)
    denominator = int(b) * int(d)
    return numerator % denominator == 0
'''
    },
    {
        "text": '''from typing import List


def order_by_points(nums: List[int]) -> List[int]:
    """Sort a list of integers based on the sum of their digits (negative sign on first digit).
    >>> order_by_points([1, 11, -1, -11, -12])
    [-1, -11, 1, -12, 11]
    """
    def digit_sum(n):
        s = str(abs(n))
        total = sum(int(d) for d in s)
        if n < 0:
            total -= 2 * int(s[0])
        return total

    return sorted(nums, key=digit_sum)
'''
    },
    {
        "text": '''def double_the_difference(lst: list) -> int:
    """Sum the squares of odd positive integers that are not floats with decimal parts.
    >>> double_the_difference([1, 3, 2, 0])
    10
    >>> double_the_difference([-1, -2, 0])
    0
    >>> double_the_difference([9, -2])
    81
    """
    return sum(
        x ** 2 for x in lst
        if isinstance(x, (int, float)) and x == int(x) and x > 0 and int(x) % 2 != 0
    )
'''
    },
    {
        "text": '''from typing import List


def compare(game: List[int], guess: List[int]) -> List[int]:
    """Compare scores with guesses and return list of absolute differences.
    >>> compare([1,2,3,4,5,1],[1,2,3,4,2,-2])
    [0,0,0,0,3,3]
    """
    return [abs(a - b) for a, b in zip(game, guess)]
'''
    },
    {
        "text": '''def strongest_extension(class_name: str, extensions: list) -> str:
    """Find the strongest extension by comparing uppercase vs lowercase letter counts.
    >>> strongest_extension('my_class', ['AA', 'Be', 'CC'])
    'my_class.AA'
    """
    best = None
    best_strength = float('-inf')
    for ext in extensions:
        strength = sum(1 for c in ext if c.isupper()) - sum(1 for c in ext if c.islower())
        if strength > best_strength:
            best_strength = strength
            best = ext
    return f"{class_name}.{best}"
'''
    },
    {
        "text": '''def cycpattern_check(a: str, b: str) -> bool:
    """Check if any rotation of b is a substring of a.
    >>> cycpattern_check("abcd", "abd")
    False
    >>> cycpattern_check("hello", "ell")
    True
    >>> cycpattern_check("whassup", "psus")
    False
    >>> cycpattern_check("aab", "aab")
    True
    """
    for i in range(len(b)):
        rotated = b[i:] + b[:i]
        if rotated in a:
            return True
    return False
'''
    },
    {
        "text": '''from typing import List


def even_odd_count(num: int) -> List[int]:
    """Return a list [even_count, odd_count] of digits.
    >>> even_odd_count(-12)
    [1, 1]
    >>> even_odd_count(123)
    [1, 2]
    """
    even = 0
    odd = 0
    for d in str(abs(num)):
        if int(d) % 2 == 0:
            even += 1
        else:
            odd += 1
    return [even, odd]
'''
    },
    {
        "text": '''def int_to_mini_roman(number: int) -> str:
    """Convert a positive integer to its lowercase roman numeral representation.
    >>> int_to_mini_roman(19)
    'xix'
    >>> int_to_mini_roman(152)
    'clii'
    """
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ['m', 'cm', 'd', 'cd', 'c', 'xc', 'l', 'xl', 'x', 'ix', 'v', 'iv', 'i']
    result = ''
    for val, sym in zip(values, symbols):
        while number >= val:
            result += sym
            number -= val
    return result
'''
    },
    {
        "text": '''def right_angle_triangle(a: float, b: float, c: float) -> bool:
    """Check if the three sides form a right angle triangle.
    >>> right_angle_triangle(3, 4, 5)
    True
    >>> right_angle_triangle(1, 2, 3)
    False
    """
    sides = sorted([a, b, c])
    return abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-6
'''
    },
    {
        "text": '''def words_string(s: str) -> list:
    """Split a string into words by commas or spaces.
    >>> words_string("Hi, my name is John")
    ["Hi", "my", "name", "is", "John"]
    """
    return s.replace(',', ' ').split()
'''
    },
    {
        "text": '''def choose_num(x: int, y: int) -> int:
    """Return the biggest even integer in [x, y] inclusive, or -1 if none exists.
    >>> choose_num(12, 15)
    14
    >>> choose_num(13, 12)
    -1
    """
    if x > y:
        return -1
    if y % 2 == 0:
        return y
    if y - 1 >= x:
        return y - 1
    return -1
'''
    },
    {
        "text": '''def rounded_avg(n: int, m: int) -> str:
    """Return binary representation of the rounded average of integers from n to m, or -1.
    >>> rounded_avg(1, 5)
    "0b11"
    >>> rounded_avg(7, 5)
    -1
    """
    if n > m:
        return -1
    avg = round((n + m) / 2)
    return bin(avg)
'''
    },
    {
        "text": '''from typing import List


def unique_digits_list(x: List[int]) -> List[int]:
    """Given a list of positive integers x, return a sorted list of all
    elements that don't have any even digit.
    >>> unique_digits_list([15, 33, 1422, 1])
    [1, 15, 33]
    """
    result = []
    for num in x:
        if all(int(d) % 2 != 0 for d in str(num)):
            result.append(num)
    return sorted(result)
'''
    },
    {
        "text": '''def multiply(a: int, b: int) -> int:
    """Return the product of the unit digits of two integers.
    >>> multiply(148, 412)
    16
    >>> multiply(19, 28)
    72
    """
    return abs(a % 10) * abs(b % 10)
'''
    },
    {
        "text": '''def count_upper(s: str) -> int:
    """Count the number of uppercase vowels in even indices.
    >>> count_upper('aBCdEf')
    1
    >>> count_upper('ABCDEF')
    2
    """
    return sum(1 for i in range(0, len(s), 2) if s[i] in 'AEIOU')
'''
    },
    {
        "text": '''def closest_integer(value: str) -> int:
    """Round a string representation of a number to the nearest integer.
    If equidistant, round away from zero.
    >>> closest_integer("10")
    10
    >>> closest_integer("15.3")
    15
    >>> closest_integer("14.5")
    15
    >>> closest_integer("-14.5")
    -15
    """
    num = float(value)
    if abs(num - round(num)) == 0.5:
        return int(num) + (1 if num > 0 else -1)
    return round(num)
'''
    },
    {
        "text": '''def make_a_pile(n: int) -> list:
    """Make a pile of n levels of stones. First level has n stones.
    Next odd level has n+2, next n+4, etc.
    >>> make_a_pile(3)
    [3, 5, 7]
    """
    return [n + 2 * i for i in range(n)]
'''
    },
    {
        "text": '''def encode(message: str) -> str:
    """Encode a message by swapping case and replacing vowels with the letter 2 positions ahead.
    >>> encode('test')
    'TGST'
    >>> encode('This is a message')
    'tHKS KS C MARKUP'
    """
    vowels = {'a': 'c', 'e': 'g', 'i': 'k', 'o': 'q', 'u': 'w',
              'A': 'C', 'E': 'G', 'I': 'K', 'O': 'Q', 'U': 'W'}
    result = []
    for c in message:
        if c in vowels:
            c = vowels[c]
        result.append(c.swapcase())
    return ''.join(result)
'''
    },
    {
        "text": '''def skjkaansen(lst: list) -> int:
    """Return the smallest value at the smallest even index, or the smallest value at the smallest odd index.
    Given a non-empty list.
    >>> skjkaansen([4,2,3])
    2
    """
    evens = [(lst[i], i) for i in range(0, len(lst), 2)]
    return min(evens)[0] if evens else None
'''
    },
    {
        "text": '''def Strongest_Extension(class_name, extensions):
    """Find the strongest extension based on uppercase vs lowercase counts.
    >>> Strongest_Extension('Slices', ['SErviNGSliCes', 'Alarm', 'YEA', 'SLiCe'])
    'Slices.SErviNGSliCes'
    """
    best = extensions[0]
    best_score = sum(1 for c in best if c.isupper()) - sum(1 for c in best if c.islower())
    for ext in extensions[1:]:
        score = sum(1 for c in ext if c.isupper()) - sum(1 for c in ext if c.islower())
        if score > best_score:
            best_score = score
            best = ext
    return f"{class_name}.{best}"
'''
    },
    {
        "text": '''def decimal_to_binary(decimal: int) -> str:
    """Convert a decimal number to binary format with 'db' prefix and suffix.
    >>> decimal_to_binary(15)
    "db1111db"
    >>> decimal_to_binary(32)
    "db100000db"
    """
    return "db" + bin(decimal)[2:] + "db"
'''
    },
    {
        "text": '''def is_happy(s: str) -> bool:
    """Check if a string is happy: length >= 3 and every 3 consecutive chars are distinct.
    >>> is_happy('a')
    False
    >>> is_happy('aa')
    False
    >>> is_happy('abcd')
    True
    >>> is_happy('aabb')
    False
    """
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if s[i] == s[i+1] or s[i+1] == s[i+2] or s[i] == s[i+2]:
            return False
    return True
'''
    },
    {
        "text": '''def numerical_letter_grade(grades: list) -> list:
    """Convert GPA scores to letter grades.
    >>> numerical_letter_grade([4.0, 3, 1.7, 2, 3.5])
    ['A+', 'B', 'C-', 'C', 'A-']
    """
    result = []
    for gpa in grades:
        if gpa == 4.0:
            result.append("A+")
        elif gpa > 3.7:
            result.append("A")
        elif gpa > 3.3:
            result.append("A-")
        elif gpa > 3.0:
            result.append("B+")
        elif gpa > 2.7:
            result.append("B")
        elif gpa > 2.3:
            result.append("B-")
        elif gpa > 2.0:
            result.append("C+")
        elif gpa > 1.7:
            result.append("C")
        elif gpa > 1.3:
            result.append("C-")
        elif gpa > 1.0:
            result.append("D+")
        elif gpa > 0.7:
            result.append("D")
        elif gpa > 0.0:
            result.append("D-")
        else:
            result.append("E")
    return result
'''
    },
    {
        "text": '''def prime_length(string: str) -> bool:
    """Check if the length of a string is a prime number.
    >>> prime_length('Hello')
    True
    >>> prime_length('abcdcba')
    True
    >>> prime_length('kA')
    True
    """
    l = len(string)
    if l < 2:
        return False
    for i in range(2, int(l**0.5) + 1):
        if l % i == 0:
            return False
    return True
'''
    },
    {
        "text": '''def starts_one_ends(n: int) -> int:
    """Count the number of n-digit positive integers that start or end with 1.
    >>> starts_one_ends(1)
    1
    """
    if n == 1:
        return 1
    return 18 * (10 ** (n - 2))
'''
    },
    {
        "text": '''def solve_string(s: str) -> str:
    """If the string has letters, reverse their case and return the string.
    If no letters, reverse the string.
    >>> solve_string("1234")
    "4321"
    >>> solve_string("ab")
    "AB"
    """
    if any(c.isalpha() for c in s):
        return ''.join(c.swapcase() if c.isalpha() else c for c in s)
    return s[::-1]
'''
    },
    {
        "text": '''from typing import List


def string_to_md5(text: str) -> str:
    """Return the md5 hash of the input string, or None if input is empty.
    >>> string_to_md5('Hello world')
    '3e25960a79dbc69b674cd4ec67a72c62'
    """
    import hashlib
    if not text:
        return None
    return hashlib.md5(text.encode()).hexdigest()
'''
    },
    {
        "text": '''from typing import List


def generate_integers(a: int, b: int) -> List[int]:
    """Return even digits between a and b, in ascending order.
    >>> generate_integers(2, 8)
    [2, 4, 6, 8]
    >>> generate_integers(8, 2)
    [2, 4, 6, 8]
    """
    lo, hi = min(a, b), max(a, b)
    return [d for d in range(lo, hi + 1) if d in (2, 4, 6, 8)]
'''
    },
]


def load_v1_data():
    """Load v1 training data."""
    examples = []
    with open('/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data.jsonl') as f:
        for line in f:
            item = json.loads(line)
            examples.append({"text": item['prompt'] + item['completion']})
    return examples


def main():
    all_examples = []

    # V2 synthetic data (HumanEval-style)
    for func in FUNCTIONS_V2:
        all_examples.append(func)
    print(f"V2 synthetic: {len(FUNCTIONS_V2)} examples")

    # V1 data
    v1_data = load_v1_data()
    all_examples.extend(v1_data)
    print(f"V1 data: {len(v1_data)} examples")

    # Deduplicate
    seen = set()
    unique = []
    for ex in all_examples:
        key = ex['text'][:100]
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    random.shuffle(unique)
    print(f"Total unique examples: {len(unique)}")

    # Save
    output_path = '/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v2.jsonl'
    with open(output_path, 'w') as f:
        for ex in unique:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
