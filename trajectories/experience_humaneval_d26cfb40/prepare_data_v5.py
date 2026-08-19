"""
Prepare v5_extra training data targeting specific failure patterns from v7 eval.
Focus on: list manipulation, string processing, mathematical patterns, edge cases.
"""
import json

OUTPUT_PATH = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v5_extra.jsonl"

EXAMPLES = [
    # --- Intersperse / list insertion patterns ---
    {
        "prompt": 'def intersperse_values(lst: list, val) -> list:\n    """Insert val between every two consecutive elements of lst.\n    >>> intersperse_values([1, 2, 3], 0)\n    [1, 0, 2, 0, 3]\n    """',
        "body": '    if not lst:\n        return []\n    result = [lst[0]]\n    for item in lst[1:]:\n        result.append(val)\n        result.append(item)\n    return result',
    },
    {
        "prompt": 'def insert_between(items: list, separator) -> list:\n    """Place separator between each element.\n    >>> insert_between(["a", "b", "c"], "-")\n    [\'a\', \'-\', \'b\', \'-\', \'c\']\n    """',
        "body": '    if len(items) <= 1:\n        return list(items)\n    result = []\n    for i, item in enumerate(items):\n        result.append(item)\n        if i < len(items) - 1:\n            result.append(separator)\n    return result',
    },
    {
        "prompt": 'def join_with(elements: list, sep) -> list:\n    """Join list elements with separator between them.\n    >>> join_with([1, 2, 3, 4], 0)\n    [1, 0, 2, 0, 3, 0, 4]\n    """',
        "body": '    result = []\n    for i, e in enumerate(elements):\n        if i > 0:\n            result.append(sep)\n        result.append(e)\n    return result',
    },
    # --- Palindrome patterns ---
    {
        "prompt": 'def shortest_palindrome(s: str) -> str:\n    """Find the shortest palindrome by adding characters to the end.\n    >>> shortest_palindrome("cat")\n    \'catac\'\n    """',
        "body": '    if not s or s == s[::-1]:\n        return s\n    for i in range(len(s)):\n        suffix = s[i:]\n        if suffix == suffix[::-1]:\n            return s + s[:i][::-1]\n    return s + s[:-1][::-1]',
    },
    {
        "prompt": 'def make_palindrome_append(s: str) -> str:\n    """Make shortest palindrome by appending to end of s.\n    Find longest palindromic suffix, then append reverse of prefix.\n    >>> make_palindrome_append("")\n    \'\'\n    >>> make_palindrome_append("cat")\n    \'catac\'\n    >>> make_palindrome_append("cata")\n    \'catac\'\n    """',
        "body": '    if not s:\n        return s\n    for i in range(len(s)):\n        if s[i:] == s[i:][::-1]:\n            return s + s[:i][::-1]\n    return s + s[:-1][::-1]',
    },
    # --- Sort by word representation ---
    {
        "prompt": 'def sort_numbers_by_name(numbers_str: str) -> str:\n    """Sort space-separated number words.\n    >>> sort_numbers_by_name("three one five")\n    \'one three five\'\n    """',
        "body": '    if not numbers_str.strip():\n        return ""\n    value_map = {\n        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,\n        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9\n    }\n    words = numbers_str.split()\n    sorted_words = sorted(words, key=lambda w: value_map[w])\n    return " ".join(sorted_words)',
    },
    {
        "prompt": 'def sort_number_words(s: str) -> str:\n    """Given space separated number words, sort them by value.\n    >>> sort_number_words("nine two zero")\n    \'zero two nine\'\n    """',
        "body": '    if not s:\n        return s\n    word_to_num = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}\n    words = s.split()\n    return " ".join(sorted(words, key=lambda w: word_to_num.get(w, 0)))',
    },
    # --- Remove duplicates (keep unique) ---
    {
        "prompt": 'def remove_duplicates(numbers: list) -> list:\n    """Remove elements that appear more than once.\n    Keep only elements that appear exactly once, in original order.\n    >>> remove_duplicates([1, 2, 3, 2, 4])\n    [1, 3, 4]\n    """',
        "body": '    from collections import Counter\n    count = Counter(numbers)\n    return [x for x in numbers if count[x] == 1]',
    },
    {
        "prompt": 'def unique_only(lst: list) -> list:\n    """Return only elements that appear exactly once, preserving order.\n    >>> unique_only([1, 2, 3, 2, 1, 5])\n    [3, 5]\n    """',
        "body": '    from collections import Counter\n    counts = Counter(lst)\n    return [x for x in lst if counts[x] == 1]',
    },
    {
        "prompt": 'def filter_unique(items: list) -> list:\n    """Keep items that occur exactly once.\n    >>> filter_unique(["a", "b", "a", "c"])\n    [\'b\', \'c\']\n    """',
        "body": '    freq = {}\n    for item in items:\n        freq[item] = freq.get(item, 0) + 1\n    return [item for item in items if freq[item] == 1]',
    },
    # --- Fibonacci-like sequences ---
    {
        "prompt": 'def tribonacci(n: int) -> int:\n    """Return nth tribonacci number.\n    tribonacci(0) = 0, tribonacci(1) = 0, tribonacci(2) = 1\n    tribonacci(n) = tribonacci(n-1) + tribonacci(n-2) + tribonacci(n-3)\n    >>> tribonacci(4)\n    2\n    """',
        "body": '    if n == 0:\n        return 0\n    if n == 1:\n        return 0\n    if n == 2:\n        return 1\n    a, b, c = 0, 0, 1\n    for _ in range(3, n + 1):\n        a, b, c = b, c, a + b + c\n    return c',
    },
    {
        "prompt": 'def fib_variant(n: int) -> int:\n    """fibfib(0) = 0, fibfib(1) = 0, fibfib(2) = 1\n    fibfib(n) = fibfib(n-1) + fibfib(n-2) + fibfib(n-3)\n    >>> fib_variant(5)\n    4\n    >>> fib_variant(8)\n    24\n    """',
        "body": '    if n <= 1:\n        return 0\n    if n == 2:\n        return 1\n    a, b, c = 0, 0, 1\n    for _ in range(3, n + 1):\n        a, b, c = b, c, a + b + c\n    return c',
    },
    # --- Vowel counting (including y at end) ---
    {
        "prompt": 'def count_vowels_special(s: str) -> int:\n    """Count vowels. Y at the end of the word also counts as a vowel.\n    >>> count_vowels_special("abcde")\n    2\n    >>> count_vowels_special("ACEDY")\n    3\n    """',
        "body": '    count = 0\n    vowels = "aeiouAEIOU"\n    for c in s:\n        if c in vowels:\n            count += 1\n    if s and s[-1] in "yY":\n        count += 1\n    return count',
    },
    {
        "prompt": 'def vowel_count(word: str) -> int:\n    """Count vowels in word. y counts as vowel only if last character.\n    >>> vowel_count("happy")\n    2\n    >>> vowel_count("gym")\n    1\n    """',
        "body": '    vowels = "aeiouAEIOU"\n    count = sum(1 for c in word if c in vowels)\n    if word and word[-1] in "yY":\n        count += 1\n    return count',
    },
    # --- Circular shift ---
    {
        "prompt": 'def circular_shift_digits(x: int, shift: int) -> str:\n    """Shift digits of x right by shift. If shift > num digits, reverse.\n    >>> circular_shift_digits(12, 1)\n    \'21\'\n    >>> circular_shift_digits(12, 2)\n    \'12\'\n    """',
        "body": '    s = str(x)\n    if shift > len(s):\n        return s[::-1]\n    shift = shift % len(s)\n    return s[-shift:] + s[:-shift] if shift else s',
    },
    {
        "prompt": 'def rotate_right(s: str, n: int) -> str:\n    """Rotate string right by n positions.\n    >>> rotate_right("abcde", 2)\n    \'deabc\'\n    """',
        "body": '    if not s:\n        return s\n    n = n % len(s)\n    return s[-n:] + s[:-n] if n else s',
    },
    # --- Fruit distribution ---
    {
        "prompt": 'def remaining_fruits(description: str, total: int) -> int:\n    """Given description like \'5 apples and 6 oranges\', find remaining from total.\n    >>> remaining_fruits("5 apples and 6 oranges", 19)\n    8\n    """',
        "body": '    nums = [int(w) for w in description.split() if w.isdigit()]\n    return total - sum(nums)',
    },
    {
        "prompt": 'def count_remaining(text: str, total: int) -> int:\n    """Extract numbers from text, subtract their sum from total.\n    >>> count_remaining("3 cats and 4 dogs", 10)\n    3\n    """',
        "body": '    numbers = [int(word) for word in text.split() if word.isdigit()]\n    return total - sum(numbers)',
    },
    # --- Search for number that appears >= itself times ---
    {
        "prompt": 'def search_frequency(lst: list) -> int:\n    """Find greatest integer where frequency >= integer.\n    If no such value exists, return -1.\n    >>> search_frequency([4, 1, 2, 2, 3, 1])\n    2\n    >>> search_frequency([1, 2, 2, 3, 3, 3, 4, 4, 4])\n    3\n    """',
        "body": '    from collections import Counter\n    count = Counter(lst)\n    result = -1\n    for num, freq in count.items():\n        if num > 0 and freq >= num:\n            result = max(result, num)\n    return result',
    },
    # --- Strange sort list (smallest, largest, next smallest...) ---
    {
        "prompt": 'def strange_sort(lst: list) -> list:\n    """Sort by alternating min and max.\n    >>> strange_sort([1, 2, 3, 4])\n    [1, 4, 2, 3]\n    >>> strange_sort([5, 5, 5, 5])\n    [5, 5, 5, 5]\n    """',
        "body": '    result = []\n    remaining = sorted(lst)\n    toggle = True\n    while remaining:\n        if toggle:\n            result.append(remaining.pop(0))\n        else:\n            result.append(remaining.pop())\n        toggle = not toggle\n    return result',
    },
    {
        "prompt": 'def alternating_sort(items: list) -> list:\n    """Alternate picking smallest and largest.\n    >>> alternating_sort([3, 1, 4, 1, 5])\n    [1, 5, 1, 4, 3]\n    """',
        "body": '    result = []\n    s = sorted(items)\n    left, right = 0, len(s) - 1\n    pick_left = True\n    while left <= right:\n        if pick_left:\n            result.append(s[left])\n            left += 1\n        else:\n            result.append(s[right])\n            right -= 1\n        pick_left = not pick_left\n    return result',
    },
    # --- Histogram (max frequency chars) ---
    {
        "prompt": 'def char_histogram(s: str) -> dict:\n    """Return dict of chars with highest frequency.\n    >>> char_histogram("a]b c")\n    {\'a\': 1, \'b\': 1, \'c\': 1}\n    >>> char_histogram("a b b a")\n    {\'a\': 2, \'b\': 2}\n    """',
        "body": '    if not s.strip():\n        return {}\n    counts = {}\n    for c in s.split():\n        counts[c] = counts.get(c, 0) + 1\n    if not counts:\n        return {}\n    max_freq = max(counts.values())\n    return {k: v for k, v in counts.items() if v == max_freq}',
    },
    # --- Reverse delete ---
    {
        "prompt": 'def reverse_delete_check(s: str, chars: str) -> tuple:\n    """Delete chars from s, check if result is palindrome.\n    >>> reverse_delete_check("abcde", "ae")\n    (\'bcd\', False)\n    >>> reverse_delete_check("abcdedcba", "ab")\n    (\'cdedec\', False)\n    """',
        "body": '    result = "".join(c for c in s if c not in chars)\n    is_palindrome = result == result[::-1]\n    return (result, is_palindrome)',
    },
    # --- Exchange lists ---
    {
        "prompt": 'def can_exchange(lst1: list, lst2: list) -> str:\n    """Check if swapping elements can make lst1 all even.\n    >>> can_exchange([1, 2, 3, 4], [1, 2, 3, 4])\n    \'YES\'\n    >>> can_exchange([1, 2, 3, 4], [1, 5, 3, 4])\n    \'NO\'\n    """',
        "body": '    odd_count = sum(1 for x in lst1 if x % 2 != 0)\n    even_available = sum(1 for x in lst2 if x % 2 == 0)\n    return "YES" if even_available >= odd_count else "NO"',
    },
    # --- Count nums with digit sum > 0 ---
    {
        "prompt": 'def count_positive_digit_sum(arr: list) -> int:\n    """Count numbers whose digit sum is positive.\n    For negative numbers, first digit is negative.\n    >>> count_positive_digit_sum([-1, 11, -11])\n    1\n    >>> count_positive_digit_sum([1, 1, 2])\n    3\n    """',
        "body": '    count = 0\n    for n in arr:\n        if n > 0:\n            count += 1\n        elif n < 0:\n            s = str(abs(n))\n            digit_sum = -int(s[0]) + sum(int(d) for d in s[1:])\n            if digit_sum > 0:\n                count += 1\n    return count',
    },
    # --- Move one ball check sorted ---
    {
        "prompt": 'def can_sort_by_rotation(arr: list) -> bool:\n    """Check if array can be sorted by right-shifting.\n    >>> can_sort_by_rotation([3, 4, 5, 1, 2])\n    True\n    >>> can_sort_by_rotation([3, 5, 4, 1, 2])\n    False\n    """',
        "body": '    if not arr:\n        return True\n    n = len(arr)\n    breaks = 0\n    for i in range(n):\n        if arr[i] > arr[(i + 1) % n]:\n            breaks += 1\n    return breaks <= 1',
    },
    # --- Rounded average in binary ---
    {
        "prompt": 'def rounded_avg_binary(n: int, m: int) -> str:\n    """Return binary of rounded average of integers from n to m inclusive.\n    If n > m return -1.\n    >>> rounded_avg_binary(1, 5)\n    \'0b11\'\n    >>> rounded_avg_binary(7, 5)\n    -1\n    """',
        "body": '    if n > m:\n        return -1\n    avg = round((n + m) / 2)\n    return bin(avg)',
    },
    # --- By length (sort and replace digits with names) ---
    {
        "prompt": 'def sort_digits_to_names(arr: list) -> list:\n    """Filter 1-9, sort descending, replace with word names.\n    >>> sort_digits_to_names([2, 1, 1, 4, 5, 8, 2, 3])\n    [\'Eight\', \'Five\', \'Four\', \'Three\', \'Two\', \'Two\', \'One\', \'One\']\n    """',
        "body": '    names = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}\n    filtered = sorted([x for x in arr if 1 <= x <= 9], reverse=True)\n    return [names[x] for x in filtered]',
    },
    # --- Choose number (largest even <= b and >= a) ---
    {
        "prompt": 'def choose_largest_even(a: int, b: int) -> int:\n    """Return the largest even number in range [a, b] inclusive.\n    Return -1 if no such number exists.\n    >>> choose_largest_even(12, 15)\n    14\n    >>> choose_largest_even(13, 12)\n    -1\n    """',
        "body": '    if a > b:\n        return -1\n    if b % 2 == 0:\n        return b\n    if b - 1 >= a:\n        return b - 1\n    return -1',
    },
    # --- Make a pile (n layers, each +2 from last) ---
    {
        "prompt": 'def stone_pile(n: int) -> list:\n    """Make n layers. First layer has n stones, next has n+2, then n+4, etc.\n    >>> stone_pile(3)\n    [3, 5, 7]\n    >>> stone_pile(1)\n    [1]\n    """',
        "body": '    return [n + 2 * i for i in range(n)]',
    },
    # --- Additional string patterns ---
    {
        "prompt": 'def encode_shift(s: str) -> str:\n    """Shift every character by 5 in the alphabet.\n    >>> encode_shift("abc")\n    \'fgh\'\n    """',
        "body": '    return "".join(chr((ord(c) - ord("a") + 5) % 26 + ord("a")) for c in s)',
    },
    {
        "prompt": 'def decode_shift(s: str) -> str:\n    """Decode a string encoded with shift of 5.\n    >>> decode_shift("fgh")\n    \'abc\'\n    """',
        "body": '    return "".join(chr((ord(c) - ord("a") - 5) % 26 + ord("a")) for c in s)',
    },
    # --- Find closest pair ---
    {
        "prompt": 'def find_closest_pair(numbers: list) -> tuple:\n    """Find two closest numbers and return them sorted.\n    >>> find_closest_pair([1.0, 2.0, 3.9, 4.0, 5.0, 2.2])\n    (3.9, 4.0)\n    """',
        "body": '    closest = None\n    min_diff = float("inf")\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            diff = abs(numbers[i] - numbers[j])\n            if diff < min_diff:\n                min_diff = diff\n                closest = tuple(sorted([numbers[i], numbers[j]]))\n    return closest',
    },
    # --- Monotonic check ---
    {
        "prompt": 'def is_monotonic(lst: list) -> bool:\n    """Check if list is monotonically increasing or decreasing.\n    >>> is_monotonic([1, 2, 4, 20])\n    True\n    >>> is_monotonic([1, 20, 4, 10])\n    False\n    """',
        "body": '    increasing = all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\n    decreasing = all(lst[i] >= lst[i+1] for i in range(len(lst)-1))\n    return increasing or decreasing',
    },
    # --- Find zero of polynomial ---
    {
        "prompt": 'def find_root(coefficients: list, low: float, high: float) -> float:\n    """Find a root of polynomial using bisection.\n    >>> abs(find_root([1, 0], -1, 1)) < 0.01\n    True\n    """',
        "body": '    def poly(x):\n        result = 0\n        for i, c in enumerate(coefficients):\n            result += c * x**i\n        return result\n    for _ in range(1000):\n        mid = (low + high) / 2\n        if poly(mid) * poly(low) <= 0:\n            high = mid\n        else:\n            low = mid\n    return (low + high) / 2',
    },
    # --- Sum of digits ---
    {
        "prompt": 'def digit_sum(n: int) -> int:\n    """Sum of digits of absolute value of n.\n    >>> digit_sum(-123)\n    6\n    >>> digit_sum(0)\n    0\n    """',
        "body": '    return sum(int(d) for d in str(abs(n)))',
    },
    # --- Unique sorted list ---
    {
        "prompt": 'def unique_sorted(lst: list) -> list:\n    """Return sorted list of unique elements.\n    >>> unique_sorted([5, 3, 5, 2, 3, 3, 9, 0, 123])\n    [0, 2, 3, 5, 9, 123]\n    """',
        "body": '    return sorted(set(lst))',
    },
    # --- RLE encode ---
    {
        "prompt": 'def rle_encode(s: str) -> list:\n    """Run-length encode a string.\n    >>> rle_encode("aaabbc")\n    [(\'a\', 3), (\'b\', 2), (\'c\', 1)]\n    """',
        "body": '    if not s:\n        return []\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            result.append((s[i-1], count))\n            count = 1\n    result.append((s[-1], count))\n    return result',
    },
    # --- Max subarray sum ---
    {
        "prompt": 'def max_subarray(arr: list) -> int:\n    """Find maximum contiguous subarray sum (Kadane).\n    >>> max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n    6\n    """',
        "body": '    if not arr:\n        return 0\n    max_sum = curr_sum = arr[0]\n    for x in arr[1:]:\n        curr_sum = max(x, curr_sum + x)\n        max_sum = max(max_sum, curr_sum)\n    return max_sum',
    },
    # --- Flatten nested list ---
    {
        "prompt": 'def flatten(lst: list) -> list:\n    """Flatten a nested list.\n    >>> flatten([1, [2, 3], [4, [5, 6]]])\n    [1, 2, 3, 4, 5, 6]\n    """',
        "body": '    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result',
    },
    # --- Zip longest ---
    {
        "prompt": 'def interleave(a: list, b: list) -> list:\n    """Interleave two lists, longer one fills the rest.\n    >>> interleave([1, 2, 3], [\'a\', \'b\'])\n    [1, \'a\', 2, \'b\', 3]\n    """',
        "body": '    result = []\n    i = 0\n    while i < len(a) or i < len(b):\n        if i < len(a):\n            result.append(a[i])\n        if i < len(b):\n            result.append(b[i])\n        i += 1\n    return result',
    },
    # --- Matrix transpose ---
    {
        "prompt": 'def transpose(matrix: list) -> list:\n    """Transpose a matrix.\n    >>> transpose([[1, 2], [3, 4], [5, 6]])\n    [[1, 3, 5], [2, 4, 6]]\n    """',
        "body": '    if not matrix:\n        return []\n    return [list(row) for row in zip(*matrix)]',
    },
    # --- Chunk list ---
    {
        "prompt": 'def chunk(lst: list, size: int) -> list:\n    """Split list into chunks of given size.\n    >>> chunk([1, 2, 3, 4, 5], 2)\n    [[1, 2], [3, 4], [5]]\n    """',
        "body": '    return [lst[i:i+size] for i in range(0, len(lst), size)]',
    },
    # --- Decode Roman numerals ---
    {
        "prompt": 'def roman_to_int(s: str) -> int:\n    """Convert Roman numeral to integer.\n    >>> roman_to_int("III")\n    3\n    >>> roman_to_int("IX")\n    9\n    >>> roman_to_int("MCMXCIV")\n    1994\n    """',
        "body": '    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}\n    total = 0\n    for i in range(len(s)):\n        if i + 1 < len(s) and values[s[i]] < values[s[i+1]]:\n            total -= values[s[i]]\n        else:\n            total += values[s[i]]\n    return total',
    },
    # --- Compare values (handle string numbers and ints/floats) ---
    {
        "prompt": 'def compare_one(a, b):\n    """Compare two values (int, float, or string). Return larger or None if equal.\n    String numbers may use comma as decimal separator.\n    >>> compare_one(1, 2.5)\n    2.5\n    >>> compare_one("5,1", "6")\n    \'6\'\n    >>> compare_one(1, 1)\n    None\n    """',
        "body": '    def to_float(x):\n        if isinstance(x, str):\n            return float(x.replace(",", "."))\n        return float(x)\n    fa, fb = to_float(a), to_float(b)\n    if fa == fb:\n        return None\n    return a if fa > fb else b',
    },
    # --- Common subsequence ---
    {
        "prompt": 'def longest_common_subsequence(s1: str, s2: str) -> str:\n    """Find the longest common subsequence.\n    >>> longest_common_subsequence("abcde", "ace")\n    \'ace\'\n    """',
        "body": '    m, n = len(s1), len(s2)\n    dp = [[""] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + s1[i-1]\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=len)\n    return dp[m][n]',
    },
    # --- Sliding window maximum ---
    {
        "prompt": 'def sliding_max(arr: list, k: int) -> list:\n    """Maximum of each sliding window of size k.\n    >>> sliding_max([1, 3, -1, -3, 5, 3, 6, 7], 3)\n    [3, 3, 5, 5, 6, 7]\n    """',
        "body": '    from collections import deque\n    result = []\n    dq = deque()\n    for i in range(len(arr)):\n        while dq and dq[0] < i - k + 1:\n            dq.popleft()\n        while dq and arr[dq[-1]] <= arr[i]:\n            dq.pop()\n        dq.append(i)\n        if i >= k - 1:\n            result.append(arr[dq[0]])\n    return result',
    },
    # --- Check balanced parentheses ---
    {
        "prompt": 'def is_balanced(s: str) -> bool:\n    """Check if parentheses are balanced.\n    >>> is_balanced("(())()")\n    True\n    >>> is_balanced("(()")\n    False\n    """',
        "body": '    count = 0\n    for c in s:\n        if c == "(":\n            count += 1\n        elif c == ")":\n            count -= 1\n        if count < 0:\n            return False\n    return count == 0',
    },
    # --- Next permutation ---
    {
        "prompt": 'def next_permutation(nums: list) -> list:\n    """Find next lexicographic permutation in place.\n    >>> next_permutation([1, 2, 3])\n    [1, 3, 2]\n    >>> next_permutation([3, 2, 1])\n    [1, 2, 3]\n    """',
        "body": '    nums = list(nums)\n    n = len(nums)\n    i = n - 2\n    while i >= 0 and nums[i] >= nums[i + 1]:\n        i -= 1\n    if i >= 0:\n        j = n - 1\n        while nums[j] <= nums[i]:\n            j -= 1\n        nums[i], nums[j] = nums[j], nums[i]\n    nums[i + 1:] = reversed(nums[i + 1:])\n    return nums',
    },
    # --- Merge intervals ---
    {
        "prompt": 'def merge_intervals(intervals: list) -> list:\n    """Merge overlapping intervals.\n    >>> merge_intervals([[1,3],[2,6],[8,10],[15,18]])\n    [[1, 6], [8, 10], [15, 18]]\n    """',
        "body": '    if not intervals:\n        return []\n    intervals.sort()\n    merged = [intervals[0]]\n    for start, end in intervals[1:]:\n        if start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged',
    },
    # --- Spiral matrix ---
    {
        "prompt": 'def spiral_order(matrix: list) -> list:\n    """Return matrix elements in spiral order.\n    >>> spiral_order([[1,2,3],[4,5,6],[7,8,9]])\n    [1, 2, 3, 6, 9, 8, 7, 4, 5]\n    """',
        "body": '    result = []\n    while matrix:\n        result += matrix.pop(0)\n        if matrix and matrix[0]:\n            for row in matrix:\n                result.append(row.pop())\n        if matrix:\n            result += matrix.pop()[::-1]\n        if matrix and matrix[0]:\n            for row in reversed(matrix):\n                result.append(row.pop(0))\n    return result',
    },
    # --- String multiply ---
    {
        "prompt": 'def multiply_strings(num1: str, num2: str) -> str:\n    """Multiply two non-negative integers given as strings.\n    >>> multiply_strings("123", "456")\n    \'56088\'\n    """',
        "body": '    return str(int(num1) * int(num2))',
    },
    # --- Find max element in nested structure ---
    {
        "prompt": 'def deep_max(data) -> int:\n    """Find maximum in a potentially nested list.\n    >>> deep_max([1, [2, [3, 4]], 5])\n    5\n    """',
        "body": '    if isinstance(data, (int, float)):\n        return data\n    return max(deep_max(item) for item in data)',
    },
    # --- Group anagrams ---
    {
        "prompt": "def group_anagrams(words: list) -> list:\n    \"\"\"Group anagrams together.\n    >>> sorted([sorted(g) for g in group_anagrams([\"eat\",\"tea\",\"tan\",\"ate\",\"nat\",\"bat\"])])\n    [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]\n    \"\"\"",
        "body": '    from collections import defaultdict\n    groups = defaultdict(list)\n    for word in words:\n        key = "".join(sorted(word))\n        groups[key].append(word)\n    return list(groups.values())',
    },
    # --- Encode/decode with shift ---
    {
        "prompt": 'def caesar_encrypt(text: str, shift: int) -> str:\n    """Caesar cipher encryption.\n    >>> caesar_encrypt("abc", 3)\n    \'def\'\n    """',
        "body": '    result = []\n    for c in text:\n        if c.isalpha():\n            base = ord("a") if c.islower() else ord("A")\n            result.append(chr((ord(c) - base + shift) % 26 + base))\n        else:\n            result.append(c)\n    return "".join(result)',
    },
    # --- Check if all elements same ---
    {
        "prompt": 'def all_same(lst: list) -> bool:\n    """Check if all elements are the same.\n    >>> all_same([1, 1, 1])\n    True\n    >>> all_same([1, 2, 1])\n    False\n    """',
        "body": '    return len(set(lst)) <= 1',
    },
    # --- Frequency sort ---
    {
        "prompt": 'def frequency_sort(s: str) -> str:\n    """Sort characters by frequency (highest first).\n    >>> frequency_sort("tree")\n    \'eert\'\n    """',
        "body": '    from collections import Counter\n    counts = Counter(s)\n    return "".join(sorted(s, key=lambda c: (-counts[c], c)))',
    },
    # --- Power set ---
    {
        "prompt": 'def power_set(lst: list) -> list:\n    """Return all subsets.\n    >>> sorted([sorted(s) for s in power_set([1, 2])])\n    [[], [1], [1, 2], [2]]\n    """',
        "body": '    result = [[]]\n    for item in lst:\n        result += [subset + [item] for subset in result]\n    return result',
    },
    # --- Top k frequent ---
    {
        "prompt": 'def top_k_frequent(nums: list, k: int) -> list:\n    """Return k most frequent elements.\n    >>> sorted(top_k_frequent([1,1,1,2,2,3], 2))\n    [1, 2]\n    """',
        "body": '    from collections import Counter\n    return [x for x, _ in Counter(nums).most_common(k)]',
    },
    # --- String compression ---
    {
        "prompt": 'def compress(s: str) -> str:\n    """Compress consecutive characters.\n    >>> compress("aabcccccaaa")\n    \'a2b1c5a3\'\n    """',
        "body": '    if not s:\n        return ""\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            result.append(s[i-1] + str(count))\n            count = 1\n    result.append(s[-1] + str(count))\n    return "".join(result)',
    },
    # --- Check if string has all unique characters ---
    {
        "prompt": 'def all_unique(s: str) -> bool:\n    """Check if all characters in string are unique.\n    >>> all_unique("abcdef")\n    True\n    >>> all_unique("aab")\n    False\n    """',
        "body": '    return len(s) == len(set(s))',
    },
    # --- Product except self ---
    {
        "prompt": 'def product_except_self(nums: list) -> list:\n    """Return array where each element is product of all others.\n    >>> product_except_self([1, 2, 3, 4])\n    [24, 12, 8, 6]\n    """',
        "body": '    n = len(nums)\n    result = [1] * n\n    left = 1\n    for i in range(n):\n        result[i] = left\n        left *= nums[i]\n    right = 1\n    for i in range(n - 1, -1, -1):\n        result[i] *= right\n        right *= nums[i]\n    return result',
    },
    # --- Two sum ---
    {
        "prompt": 'def two_sum(nums: list, target: int) -> list:\n    """Return indices of two numbers that add up to target.\n    >>> two_sum([2, 7, 11, 15], 9)\n    [0, 1]\n    """',
        "body": '    seen = {}\n    for i, n in enumerate(nums):\n        complement = target - n\n        if complement in seen:\n            return [seen[complement], i]\n        seen[n] = i\n    return []',
    },
    # --- Rotate matrix ---
    {
        "prompt": 'def rotate_90(matrix: list) -> list:\n    """Rotate matrix 90 degrees clockwise.\n    >>> rotate_90([[1,2],[3,4]])\n    [[3, 1], [4, 2]]\n    """',
        "body": '    n = len(matrix)\n    return [[matrix[n-1-j][i] for j in range(n)] for i in range(n)]',
    },
    # --- Valid parentheses with multiple types ---
    {
        "prompt": 'def valid_brackets(s: str) -> bool:\n    """Check if brackets are valid.\n    >>> valid_brackets("()[]{}")\n    True\n    >>> valid_brackets("(]")\n    False\n    """',
        "body": '    stack = []\n    pairs = {")": "(", "]": "[", "}": "{"}\n    for c in s:\n        if c in "([{":\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack[-1] != pairs[c]:\n                return False\n            stack.pop()\n    return len(stack) == 0',
    },
    # --- Count set bits ---
    {
        "prompt": 'def count_bits(n: int) -> int:\n    """Count number of 1 bits.\n    >>> count_bits(11)\n    3\n    """',
        "body": '    return bin(n).count("1")',
    },
    # --- Missing number ---
    {
        "prompt": 'def missing_number(nums: list) -> int:\n    """Find missing number in range [0, n].\n    >>> missing_number([3, 0, 1])\n    2\n    """',
        "body": '    n = len(nums)\n    return n * (n + 1) // 2 - sum(nums)',
    },
    # --- Single number ---
    {
        "prompt": 'def single_number(nums: list) -> int:\n    """Find element that appears once (others appear twice).\n    >>> single_number([2, 2, 1])\n    1\n    """',
        "body": '    result = 0\n    for n in nums:\n        result ^= n\n    return result',
    },
    # --- Valid palindrome (alphanumeric only) ---
    {
        "prompt": 'def is_alpha_palindrome(s: str) -> bool:\n    """Check if string is palindrome considering only alphanumeric.\n    >>> is_alpha_palindrome("A man, a plan, a canal: Panama")\n    True\n    """',
        "body": '    cleaned = "".join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]',
    },
    # --- Longest common prefix ---
    {
        "prompt": 'def longest_common_prefix(strs: list) -> str:\n    """Find longest common prefix.\n    >>> longest_common_prefix(["flower","flow","flight"])\n    \'fl\'\n    """',
        "body": '    if not strs:\n        return ""\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ""\n    return prefix',
    },
    # --- Nth digit ---
    {
        "prompt": 'def sum_of_squares(n: int) -> int:\n    """Sum of squares from 1 to n.\n    >>> sum_of_squares(3)\n    14\n    """',
        "body": '    return sum(i * i for i in range(1, n + 1))',
    },
]


def main():
    with open(OUTPUT_PATH, 'w') as f:
        count = 0
        for ex in EXAMPLES:
            for rep in range(3):
                f.write(json.dumps({"prompt": ex["prompt"], "body": ex["body"]}) + "\n")
                count += 1
    print(f"Wrote {count} examples ({len(EXAMPLES)} unique × 3 reps) to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
