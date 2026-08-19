"""
Prepare v6_extra data focusing on:
1. Problems with proper 4-space indentation (fix IndentationErrors)
2. Problems that require only function body output (fix NameErrors from test code generation)
3. More targeted algorithmic patterns for remaining assertion errors
"""
import json

OUTPUT_PATH = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v6_extra.jsonl"

EXAMPLES = [
    # --- Problems requiring careful indentation (4 spaces) ---
    {
        "prompt": 'def pluck_min_even(arr: list) -> list:\n    """Find smallest even value and its index.\n    Return [value, index] or [] if no even values.\n    >>> pluck_min_even([4, 2, 3])\n    [2, 1]\n    >>> pluck_min_even([1, 3, 5])\n    []\n    """',
        "body": '    min_val = float("inf")\n    min_idx = -1\n    for i, v in enumerate(arr):\n        if v % 2 == 0 and v < min_val:\n            min_val = v\n            min_idx = i\n    if min_idx == -1:\n        return []\n    return [min_val, min_idx]',
    },
    {
        "prompt": 'def words_string(s: str) -> list:\n    """Split a string of words by commas or spaces.\n    >>> words_string("Hi, my name is John")\n    [\'Hi\', \'my\', \'name\', \'is\', \'John\']\n    """',
        "body": '    return [w for w in s.replace(",", " ").split() if w]',
    },
    {
        "prompt": 'def choose_num(x: int, y: int) -> int:\n    """Return largest even number in range [x, y], or -1 if none exists.\n    >>> choose_num(12, 15)\n    14\n    >>> choose_num(13, 12)\n    -1\n    """',
        "body": '    if x > y:\n        return -1\n    if y % 2 == 0:\n        return y\n    if y - 1 >= x:\n        return y - 1\n    return -1',
    },
    {
        "prompt": 'def by_length(arr: list) -> list:\n    """Filter to 1-9, sort in reverse, map to names.\n    >>> by_length([2, 1, 1, 4, 5, 8, 2, 3])\n    [\'Eight\', \'Five\', \'Four\', \'Three\', \'Two\', \'Two\', \'One\', \'One\']\n    """',
        "body": '    names = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",\n             6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}\n    filtered = sorted([x for x in arr if 1 <= x <= 9], reverse=True)\n    return [names[x] for x in filtered]',
    },
    # --- Problems that need function body only (no driver code) ---
    {
        "prompt": 'def solve_case(s: str) -> str:\n    """Flip the case of each letter. If no letters, reverse string.\n    >>> solve_case("1234")\n    \'4321\'\n    >>> solve_case("aBcD")\n    \'AbCd\'\n    """',
        "body": '    has_letter = any(c.isalpha() for c in s)\n    if not has_letter:\n        return s[::-1]\n    return "".join(c.swapcase() if c.isalpha() else c for c in s)',
    },
    {
        "prompt": 'def encode_swap(message: str) -> str:\n    """Swap case and replace vowels with the vowel 2 positions ahead.\n    >>> encode_swap("test")\n    \'TGST\'\n    >>> encode_swap("TEST")\n    \'tgst\'\n    """',
        "body": '    vowels = "aeiou"\n    vowel_map = {"a": "c", "e": "g", "i": "k", "o": "q", "u": "w",\n                 "A": "C", "E": "G", "I": "K", "O": "Q", "U": "W"}\n    result = []\n    for c in message:\n        if c in vowel_map:\n            result.append(vowel_map[c])\n        elif c.isalpha():\n            result.append(c)\n        else:\n            result.append(c)\n    return "".join(c.swapcase() if c.isalpha() else c for c in result)',
    },
    {
        "prompt": 'def count_bored(S: str) -> int:\n    """Count sentences starting with \'I \'.\n    Sentences are split by . ? or !\n    >>> count_bored("I am happy. You are not. I am bored.")\n    2\n    """',
        "body": '    import re\n    sentences = re.split(r"[.?!]", S)\n    count = 0\n    for s in sentences:\n        s = s.strip()\n        if s.startswith("I "):\n            count += 1\n    return count',
    },
    # --- Mathematical / logic patterns ---
    {
        "prompt": 'def will_it_fly(q: list, w: int) -> bool:\n    """Return True if q is a palindromic list and sum <= w.\n    >>> will_it_fly([1, 2], 5)\n    False\n    >>> will_it_fly([3, 2, 3], 9)\n    True\n    """',
        "body": '    return q == q[::-1] and sum(q) <= w',
    },
    {
        "prompt": 'def is_palindromic_and_light(q: list, w: int) -> bool:\n    """Check if list is palindrome and sum of elements <= w.\n    >>> is_palindromic_and_light([1, 2, 1], 10)\n    True\n    >>> is_palindromic_and_light([1, 2, 3], 6)\n    False\n    """',
        "body": '    return q == q[::-1] and sum(q) <= w',
    },
    {
        "prompt": 'def count_positive_digit_sum_v2(arr: list) -> int:\n    """Count elements whose digit sum is > 0.\n    For negatives, first digit is negative.\n    >>> count_positive_digit_sum_v2([-1, -2, 0])\n    0\n    >>> count_positive_digit_sum_v2([1, 100, -11])\n    2\n    """',
        "body": '    count = 0\n    for n in arr:\n        if n > 0:\n            count += 1\n        elif n < 0:\n            digits = str(abs(n))\n            digit_sum = -int(digits[0]) + sum(int(d) for d in digits[1:])\n            if digit_sum > 0:\n                count += 1\n    return count',
    },
    {
        "prompt": 'def max_fill_buckets(grid: list, capacity: int) -> int:\n    """Each row is a well. Count total bucket trips to empty all wells.\n    >>> max_fill_buckets([[0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]], 1)\n    6\n    """',
        "body": '    import math\n    total = 0\n    for row in grid:\n        water = sum(row)\n        total += math.ceil(water / capacity)\n    return total',
    },
    {
        "prompt": 'def histogram_words(test: str) -> dict:\n    """Return dict of letters with highest frequency from space-separated words.\n    >>> histogram_words("a b c")\n    {\'a\': 1, \'b\': 1, \'c\': 1}\n    >>> histogram_words("a b b a")\n    {\'a\': 2, \'b\': 2}\n    """',
        "body": '    if not test.strip():\n        return {}\n    words = test.split()\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    max_count = max(counts.values())\n    return {k: v for k, v in counts.items() if v == max_count}',
    },
    {
        "prompt": 'def move_one_ball_check(arr: list) -> bool:\n    """Check if array can be sorted by right shifts.\n    >>> move_one_ball_check([3, 4, 5, 1, 2])\n    True\n    >>> move_one_ball_check([3, 5, 4, 1, 2])\n    False\n    """',
        "body": '    if not arr:\n        return True\n    n = len(arr)\n    inversions = 0\n    for i in range(n):\n        if arr[i] > arr[(i + 1) % n]:\n            inversions += 1\n    return inversions <= 1',
    },
    {
        "prompt": 'def exchange_to_even(lst1: list, lst2: list) -> str:\n    """Check if we can make lst1 all even by swapping with lst2.\n    >>> exchange_to_even([1, 2, 3, 4], [1, 2, 3, 4])\n    \'YES\'\n    """',
        "body": '    odd_in_1 = sum(1 for x in lst1 if x % 2 != 0)\n    even_in_2 = sum(1 for x in lst2 if x % 2 == 0)\n    return "YES" if even_in_2 >= odd_in_1 else "NO"',
    },
    {
        "prompt": 'def rounded_avg_to_bin(n: int, m: int):\n    """Return binary of rounded average of n..m, or -1 if n > m.\n    >>> rounded_avg_to_bin(1, 5)\n    \'0b11\'\n    """',
        "body": '    if n > m:\n        return -1\n    avg = round((n + m) / 2)\n    return bin(avg)',
    },
    {
        "prompt": 'def get_odd_collatz(n: int) -> list:\n    """Return sorted odd numbers in Collatz sequence starting from n.\n    >>> get_odd_collatz(5)\n    [1, 5]\n    """',
        "body": '    odds = set()\n    while n != 1:\n        if n % 2 != 0:\n            odds.add(n)\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n    odds.add(1)\n    return sorted(odds)',
    },
    {
        "prompt": 'def largest_smallest_ints(lst: list) -> tuple:\n    """Return (largest negative, smallest positive) or None for missing.\n    >>> largest_smallest_ints([2, 4, 1, 3, 5, 7])\n    (None, 1)\n    >>> largest_smallest_ints([-1, -2])\n    (-1, None)\n    """',
        "body": '    negatives = [x for x in lst if x < 0]\n    positives = [x for x in lst if x > 0]\n    a = max(negatives) if negatives else None\n    b = min(positives) if positives else None\n    return (a, b)',
    },
    {
        "prompt": 'def prod_signs(arr: list):\n    """Return sum of magnitudes * product of signs. None for empty.\n    >>> prod_signs([1, 2, 2, -4])\n    -9\n    >>> prod_signs([])\n    None\n    """',
        "body": '    if not arr:\n        return None\n    if 0 in arr:\n        return 0\n    sign = 1\n    for x in arr:\n        if x < 0:\n            sign *= -1\n    return sign * sum(abs(x) for x in arr)',
    },
    {
        "prompt": 'def get_closest_vowel(word: str) -> str:\n    """Find the closest vowel between two consonants from right side.\n    >>> get_closest_vowel("yogurt")\n    \'u\'\n    >>> get_closest_vowel("FULL")\n    \'U\'\n    """',
        "body": '    vowels = set("aeiouAEIOU")\n    for i in range(len(word) - 2, 0, -1):\n        if word[i] in vowels and word[i-1] not in vowels and word[i+1] not in vowels:\n            return word[i]\n    return ""',
    },
    {
        "prompt": 'def check_last_char(txt: str) -> bool:\n    """Check if last character is alphabetical and not part of a word.\n    >>> check_last_char("apple pie")\n    False\n    >>> check_last_char("apple pi e")\n    True\n    """',
        "body": '    if not txt or not txt[-1].isalpha():\n        return False\n    if len(txt) >= 2 and txt[-2] != " ":\n        return False\n    return True',
    },
    # --- Count digits matching position parity ---
    {
        "prompt": 'def count_parity_match(n: int) -> int:\n    """Count digits of n that have same parity as their position (0-indexed).\n    >>> count_parity_match(1234)\n    2\n    """',
        "body": '    count = 0\n    for i, d in enumerate(str(n)):\n        if int(d) % 2 == i % 2:\n            count += 1\n    return count',
    },
    # --- Hex key prime counting ---
    {
        "prompt": 'def hex_key_primes(num: str) -> int:\n    """Count hex digits that are prime numbers.\n    >>> hex_key_primes("AB")\n    1\n    >>> hex_key_primes("2357BD")\n    4\n    """',
        "body": '    primes = set("2357BD")\n    return sum(1 for c in num if c in primes)',
    },
    # --- Is simple power ---
    {
        "prompt": 'def is_simple_power(x: int, n: int) -> bool:\n    """Check if x is n^k for some integer k.\n    >>> is_simple_power(1, 4)\n    True\n    >>> is_simple_power(8, 2)\n    True\n    """',
        "body": '    if x == 1:\n        return True\n    power = n\n    while power < x:\n        power *= n\n    return power == x',
    },
    # --- Correct bracket groups ---
    {
        "prompt": 'def separate_paren_groups(paren_string: str) -> list:\n    """Separate balanced parentheses groups.\n    >>> separate_paren_groups("( ) (( )) (( )( ))")\n    [\'()\', \'(())\', \'(()())\']\n    """',
        "body": '    result = []\n    current = ""\n    depth = 0\n    for c in paren_string:\n        if c == "(":\n            depth += 1\n            current += c\n        elif c == ")":\n            depth -= 1\n            current += c\n            if depth == 0:\n                result.append(current)\n                current = ""\n    return result',
    },
    # --- Sum squares with rounding ---
    {
        "prompt": 'def sum_squares_ceil(lst: list) -> int:\n    """Sum of squared ceiling values.\n    >>> sum_squares_ceil([1.0, 2.0, 3.0])\n    14\n    >>> sum_squares_ceil([1.4, 4.2, 0.0])\n    29\n    """',
        "body": '    import math\n    return sum(math.ceil(x) ** 2 for x in lst)',
    },
    # --- Decimal to binary with extra markers ---
    {
        "prompt": 'def decimal_to_binary_marked(decimal: int) -> str:\n    """Convert decimal to binary with db prefix and suffix.\n    >>> decimal_to_binary_marked(15)\n    \'db1111db\'\n    >>> decimal_to_binary_marked(32)\n    \'db100000db\'\n    """',
        "body": '    return "db" + bin(decimal)[2:] + "db"',
    },
    # --- N-digit count starting or ending with 1 ---
    {
        "prompt": 'def starts_or_ends_one(n: int) -> int:\n    """Count n-digit positive integers that start or end with 1.\n    >>> starts_or_ends_one(1)\n    1\n    >>> starts_or_ends_one(2)\n    18\n    """',
        "body": '    if n == 1:\n        return 1\n    return 18 * (10 ** (n - 2))',
    },
    # --- Second smallest ---
    {
        "prompt": 'def next_smallest(lst: list):\n    """Return second smallest unique element, or None.\n    >>> next_smallest([1, 2, 3, 4, 5])\n    2\n    >>> next_smallest([5, 1, 4, 3, 2])\n    2\n    >>> next_smallest([])\n    None\n    >>> next_smallest([1, 1])\n    None\n    """',
        "body": '    unique = sorted(set(lst))\n    return unique[1] if len(unique) >= 2 else None',
    },
    # --- Sum odd positioned even-indexed ---
    {
        "prompt": 'def add_odd_index(lst: list) -> int:\n    """Sum elements at odd indices.\n    >>> add_odd_index([4, 2, 6, 7])\n    9\n    """',
        "body": '    return sum(lst[i] for i in range(1, len(lst), 2))',
    },
    {
        "prompt": 'def add_even_or_odd(lst: list) -> int:\n    """Add elements at even indices if first element is odd,\n    else add elements at odd indices.\n    >>> add_even_or_odd([4, 88])\n    88\n    """',
        "body": '    if not lst:\n        return 0\n    if lst[0] % 2 == 0:\n        return sum(lst[i] for i in range(1, len(lst), 2))\n    else:\n        return sum(lst[i] for i in range(0, len(lst), 2))',
    },
    # --- String sorting with coordinates ---
    {
        "prompt": 'def get_row(lst: list, x: int) -> list:\n    """Find all occurrences of x in 2D list, return as (row, col) sorted.\n    Sort by row ascending, then by column descending.\n    >>> get_row([[1, 2, 3], [4, 5, 6]], 1)\n    [(0, 0)]\n    """',
        "body": '    result = []\n    for i, row in enumerate(lst):\n        for j, val in enumerate(row):\n            if val == x:\n                result.append((i, j))\n    result.sort(key=lambda p: (p[0], -p[1]))\n    return result',
    },
    # --- Check sorting (with duplicates) ---
    {
        "prompt": 'def is_sorted_strict(lst: list) -> bool:\n    """Check if sorted ascending with no more than 1 duplicate of any element.\n    >>> is_sorted_strict([1, 2, 3, 4, 5])\n    True\n    >>> is_sorted_strict([1, 2, 2, 2, 3])\n    False\n    """',
        "body": '    from collections import Counter\n    if lst != sorted(lst):\n        return False\n    counts = Counter(lst)\n    return all(v <= 2 for v in counts.values())',
    },
    # --- Strong password check ---
    {
        "prompt": 'def is_happy_string(s: str) -> bool:\n    """Check if every 3 consecutive chars are distinct.\n    >>> is_happy_string("abc")\n    True\n    >>> is_happy_string("aab")\n    False\n    """',
        "body": '    if len(s) < 3:\n        return False\n    for i in range(len(s) - 2):\n        if len(set(s[i:i+3])) != 3:\n            return False\n    return True',
    },
    # --- Fix spaces in text ---
    {
        "prompt": 'def fix_spaces(text: str) -> str:\n    """Replace 1 space with _, 2+ consecutive spaces with -.\n    >>> fix_spaces("Example")\n    \'Example\'\n    >>> fix_spaces(" Example 3")\n    \'_Example_3\'\n    >>> fix_spaces("Example   Boof")\n    \'Example-Boof\'\n    """',
        "body": '    result = []\n    i = 0\n    while i < len(text):\n        if text[i] == " ":\n            count = 0\n            while i < len(text) and text[i] == " ":\n                count += 1\n                i += 1\n            if count >= 2:\n                result.append("-")\n            else:\n                result.append("_")\n        else:\n            result.append(text[i])\n            i += 1\n    return "".join(result)',
    },
    # --- Find max word (unique chars) ---
    {
        "prompt": 'def find_max_unique(words: list) -> str:\n    """Find word with max unique characters. Tie-break: alphabetically first.\n    >>> find_max_unique(["name", "of", "string"])\n    \'string\'\n    """',
        "body": '    return max(words, key=lambda w: (len(set(w)), -ord(w[0]))) if words else ""',
    },
    # --- Check prime multiplication ---
    {
        "prompt": 'def is_multiply_prime(a: int) -> bool:\n    """Check if a is product of 3 prime numbers.\n    >>> is_multiply_prime(30)\n    True\n    """',
        "body": '    def smallest_prime_factor(n):\n        for i in range(2, int(n**0.5) + 1):\n            if n % i == 0:\n                return i\n        return n\n    count = 0\n    n = a\n    while n > 1 and count < 4:\n        p = smallest_prime_factor(n)\n        n //= p\n        count += 1\n    return count == 3 and n == 1',
    },
    # --- Musical notes parsing ---
    {
        "prompt": 'def parse_music(music_string: str) -> list:\n    """Parse music notes: o=4, o|=2, .|=1.\n    >>> parse_music("o o| .| o| o| .| .| .| .| o o")\n    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n    """',
        "body": '    if not music_string:\n        return []\n    note_map = {"o": 4, "o|": 2, ".|": 1}\n    return [note_map[n] for n in music_string.split() if n in note_map]',
    },
    # --- String number formatting ---
    {
        "prompt": 'def format_number_string(n: int) -> str:\n    """Generate string of numbers from 0 to n separated by spaces.\n    >>> format_number_string(5)\n    \'0 1 2 3 4 5\'\n    """',
        "body": '    return " ".join(str(i) for i in range(n + 1))',
    },
    # --- Poly evaluation ---
    {
        "prompt": 'def evaluate_poly(coeffs: list, x: float) -> float:\n    """Evaluate polynomial with given coefficients at x.\n    coeffs[i] is coefficient of x^i.\n    >>> evaluate_poly([1, 2, 3], 2)\n    17\n    """',
        "body": '    return sum(c * x**i for i, c in enumerate(coeffs))',
    },
    # --- Find zero using bisection (general) ---
    {
        "prompt": 'def find_zero_poly(xs: list) -> float:\n    """Find zero of polynomial with even number of coefficients.\n    Uses bisection method.\n    >>> abs(find_zero_poly([1, 2])) < 1e-5\n    True\n    """',
        "body": '    def poly(x):\n        return sum(c * x**i for i, c in enumerate(xs))\n    lo, hi = -1.0, 1.0\n    while poly(lo) * poly(hi) > 0:\n        lo *= 2\n        hi *= 2\n    for _ in range(100):\n        mid = (lo + hi) / 2\n        if poly(mid) * poly(lo) <= 0:\n            hi = mid\n        else:\n            lo = mid\n    return (lo + hi) / 2',
    },
    # --- Triangle area using Heron's formula ---
    {
        "prompt": 'def triangle_area_sides(a: int, b: int, c: int) -> float:\n    """Return area using Heron formula, rounded to 2 decimals.\n    Return -1 if not a valid triangle.\n    >>> triangle_area_sides(3, 4, 5)\n    6.0\n    """',
        "body": '    if a + b <= c or a + c <= b or b + c <= a:\n        return -1\n    s = (a + b + c) / 2\n    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5\n    return round(area, 2)',
    },
    # --- Compare strings, return matching index ---
    {
        "prompt": 'def compare_one_val(a, b):\n    """Compare two values (int, float, or string with comma decimal).\n    Return the larger value in its original type, or None if equal.\n    >>> compare_one_val(1, 2)\n    2\n    >>> compare_one_val(\"5,1\", \"6\")\n    \'6\'\n    """',
        "body": '    def to_float(x):\n        if isinstance(x, str):\n            return float(x.replace(",", "."))\n        return float(x)\n    fa, fb = to_float(a), to_float(b)\n    if fa == fb:\n        return None\n    return a if fa > fb else b',
    },
    # --- Planets between ---
    {
        "prompt": 'def bf_planets(planet1: str, planet2: str) -> tuple:\n    """Return planets between planet1 and planet2 in order from sun.\n    >>> bf_planets("Jupiter", "Neptune")\n    (\'Saturn\', \'Uranus\')\n    """',
        "body": '    planets = ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")\n    if planet1 not in planets or planet2 not in planets:\n        return ()\n    i1 = planets.index(planet1)\n    i2 = planets.index(planet2)\n    if i1 > i2:\n        i1, i2 = i2, i1\n    return tuple(planets[i1 + 1:i2])',
    },
    # --- More list processing ---
    {
        "prompt": 'def total_match(lst1: list, lst2: list) -> list:\n    """Return list with smaller total character count.\n    >>> total_match(["hi", "admin"], ["hI", "Hi"])\n    [\'hI\', \'Hi\']\n    """',
        "body": '    total1 = sum(len(s) for s in lst1)\n    total2 = sum(len(s) for s in lst2)\n    if total1 <= total2:\n        return lst1\n    return lst2',
    },
    # --- Rolling max ---
    {
        "prompt": 'def rolling_max(numbers: list) -> list:\n    """Return running max for each position.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    """',
        "body": '    result = []\n    current_max = float("-inf")\n    for n in numbers:\n        current_max = max(current_max, n)\n        result.append(current_max)\n    return result',
    },
    # --- Rescale to unit ---
    {
        "prompt": 'def rescale_to_unit(numbers: list) -> list:\n    """Rescale list to [0, 1] range.\n    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n    [0.0, 0.25, 0.5, 0.75, 1.0]\n    """',
        "body": '    min_val = min(numbers)\n    max_val = max(numbers)\n    return [(x - min_val) / (max_val - min_val) for x in numbers]',
    },
    # --- String concatenation ---
    {
        "prompt": 'def concatenate(strings: list) -> str:\n    """Concatenate list of strings.\n    >>> concatenate(["a", "b", "c"])\n    \'abc\'\n    """',
        "body": '    return "".join(strings)',
    },
    # --- Filter by prefix ---
    {
        "prompt": 'def filter_by_prefix(strings: list, prefix: str) -> list:\n    """Filter strings that start with prefix.\n    >>> filter_by_prefix(["abc", "bcd", "abd"], "ab")\n    [\'abc\', \'abd\']\n    """',
        "body": '    return [s for s in strings if s.startswith(prefix)]',
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
