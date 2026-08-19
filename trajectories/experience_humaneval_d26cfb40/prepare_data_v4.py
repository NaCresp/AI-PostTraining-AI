"""
Prepare additional training data v4: 200+ unique synthetic examples targeting
harder HumanEval-style problems. Focus on:
1. Math / number theory (gcd, primes, factorials, modular arithmetic)
2. String processing (regex patterns, encoding, parsing)
3. List/array operations (sorting, searching, filtering, transforming)
4. Recursive algorithms (trees, sequences, divide-and-conquer)
5. Dictionary/set operations
6. Edge cases (empty inputs, negative numbers, special values)

Bodies that need external modules use inline imports.
Output: JSONL with 5x repetitions -> 1000+ lines.
"""

import json

OUTPUT_PATH = (
    "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/"
    "train_data_v4_extra.jsonl"
)

# ---------------------------------------------------------------------------
# 1. Math / number theory
# ---------------------------------------------------------------------------
MATH_EXAMPLES = [
    {
        "prompt": (
            'def is_prime(n: int) -> bool:\n'
            '    """Return True if n is a prime number.\n'
            '    >>> is_prime(7)\n'
            '    True\n'
            '    >>> is_prime(4)\n'
            '    False\n'
            '    >>> is_prime(1)\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    if n < 2:\n'
            '        return False\n'
            '    if n == 2:\n'
            '        return True\n'
            '    if n % 2 == 0:\n'
            '        return False\n'
            '    for i in range(3, int(n**0.5) + 1, 2):\n'
            '        if n % i == 0:\n'
            '            return False\n'
            '    return True'
        ),
    },
    {
        "prompt": (
            'def prime_factors(n: int) -> list:\n'
            '    """Return sorted list of prime factors of n.\n'
            '    >>> prime_factors(12)\n'
            '    [2, 2, 3]\n'
            '    >>> prime_factors(7)\n'
            '    [7]\n'
            '    """'
        ),
        "body": (
            '    factors = []\n'
            '    d = 2\n'
            '    while d * d <= n:\n'
            '        while n % d == 0:\n'
            '            factors.append(d)\n'
            '            n //= d\n'
            '        d += 1\n'
            '    if n > 1:\n'
            '        factors.append(n)\n'
            '    return factors'
        ),
    },
    {
        "prompt": (
            'def sieve_of_eratosthenes(limit: int) -> list:\n'
            '    """Return all primes up to limit (inclusive).\n'
            '    >>> sieve_of_eratosthenes(20)\n'
            '    [2, 3, 5, 7, 11, 13, 17, 19]\n'
            '    """'
        ),
        "body": (
            '    if limit < 2:\n'
            '        return []\n'
            '    sieve = [True] * (limit + 1)\n'
            '    sieve[0] = sieve[1] = False\n'
            '    for i in range(2, int(limit**0.5) + 1):\n'
            '        if sieve[i]:\n'
            '            for j in range(i*i, limit + 1, i):\n'
            '                sieve[j] = False\n'
            '    return [i for i in range(2, limit + 1) if sieve[i]]'
        ),
    },
    {
        "prompt": (
            'def gcd(a: int, b: int) -> int:\n'
            '    """Compute GCD using Euclidean algorithm.\n'
            '    >>> gcd(48, 18)\n'
            '    6\n'
            '    >>> gcd(0, 5)\n'
            '    5\n'
            '    """'
        ),
        "body": (
            '    while b:\n'
            '        a, b = b, a % b\n'
            '    return a'
        ),
    },
    {
        "prompt": (
            'def lcm(a: int, b: int) -> int:\n'
            '    """Compute the least common multiple of a and b.\n'
            '    >>> lcm(4, 6)\n'
            '    12\n'
            '    >>> lcm(7, 3)\n'
            '    21\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    return abs(a * b) // math.gcd(a, b) if a and b else 0'
        ),
    },
    {
        "prompt": (
            'def factorial_iter(n: int) -> int:\n'
            '    """Return n! iteratively.\n'
            '    >>> factorial_iter(5)\n'
            '    120\n'
            '    >>> factorial_iter(0)\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    result = 1\n'
            '    for i in range(2, n + 1):\n'
            '        result *= i\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def factorial_recursive(n: int) -> int:\n'
            '    """Return n! recursively.\n'
            '    >>> factorial_recursive(6)\n'
            '    720\n'
            '    >>> factorial_recursive(0)\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    if n <= 1:\n'
            '        return 1\n'
            '    return n * factorial_recursive(n - 1)'
        ),
    },
    {
        "prompt": (
            'def fibonacci(n: int) -> int:\n'
            '    """Return the n-th Fibonacci number (0-indexed, fib(0)=0).\n'
            '    >>> fibonacci(0)\n'
            '    0\n'
            '    >>> fibonacci(7)\n'
            '    13\n'
            '    """'
        ),
        "body": (
            '    if n <= 0:\n'
            '        return 0\n'
            '    a, b = 0, 1\n'
            '    for _ in range(n - 1):\n'
            '        a, b = b, a + b\n'
            '    return b'
        ),
    },
    {
        "prompt": (
            'def fibonacci_sequence(n: int) -> list:\n'
            '    """Return first n Fibonacci numbers.\n'
            '    >>> fibonacci_sequence(7)\n'
            '    [0, 1, 1, 2, 3, 5, 8]\n'
            '    """'
        ),
        "body": (
            '    if n <= 0:\n'
            '        return []\n'
            '    if n == 1:\n'
            '        return [0]\n'
            '    seq = [0, 1]\n'
            '    for _ in range(2, n):\n'
            '        seq.append(seq[-1] + seq[-2])\n'
            '    return seq'
        ),
    },
    {
        "prompt": (
            'def is_perfect_number(n: int) -> bool:\n'
            '    """Return True if n equals the sum of its proper divisors.\n'
            '    >>> is_perfect_number(28)\n'
            '    True\n'
            '    >>> is_perfect_number(12)\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    if n < 2:\n'
            '        return False\n'
            '    divisors_sum = 1\n'
            '    for i in range(2, int(n**0.5) + 1):\n'
            '        if n % i == 0:\n'
            '            divisors_sum += i\n'
            '            if i != n // i:\n'
            '                divisors_sum += n // i\n'
            '    return divisors_sum == n'
        ),
    },
    {
        "prompt": (
            'def power_mod(base: int, exp: int, mod: int) -> int:\n'
            '    """Return (base**exp) % mod efficiently.\n'
            '    >>> power_mod(2, 10, 1000)\n'
            '    24\n'
            '    """'
        ),
        "body": '    return pow(base, exp, mod)',
    },
    {
        "prompt": (
            'def collatz_length(n: int) -> int:\n'
            '    """Return the number of steps to reach 1 in the Collatz sequence.\n'
            '    >>> collatz_length(6)\n'
            '    8\n'
            '    >>> collatz_length(1)\n'
            '    0\n'
            '    """'
        ),
        "body": (
            '    steps = 0\n'
            '    while n != 1:\n'
            '        if n % 2 == 0:\n'
            '            n //= 2\n'
            '        else:\n'
            '            n = 3 * n + 1\n'
            '        steps += 1\n'
            '    return steps'
        ),
    },
    {
        "prompt": (
            'def digits_sum(n: int) -> int:\n'
            '    """Return the sum of digits of an integer (handle negatives).\n'
            '    >>> digits_sum(123)\n'
            '    6\n'
            '    >>> digits_sum(-45)\n'
            '    9\n'
            '    """'
        ),
        "body": '    return sum(int(d) for d in str(abs(n)))',
    },
    {
        "prompt": (
            'def is_armstrong(n: int) -> bool:\n'
            '    """Return True if n is an Armstrong (narcissistic) number.\n'
            '    >>> is_armstrong(153)\n'
            '    True\n'
            '    >>> is_armstrong(100)\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    digits = str(n)\n'
            '    power = len(digits)\n'
            '    return sum(int(d) ** power for d in digits) == n'
        ),
    },
    {
        "prompt": (
            'def num_divisors(n: int) -> int:\n'
            '    """Return the count of divisors of n.\n'
            '    >>> num_divisors(12)\n'
            '    6\n'
            '    >>> num_divisors(1)\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    count = 0\n'
            '    for i in range(1, int(n**0.5) + 1):\n'
            '        if n % i == 0:\n'
            '            count += 2 if i != n // i else 1\n'
            '    return count'
        ),
    },
    {
        "prompt": (
            'def int_to_roman(num: int) -> str:\n'
            '    """Convert an integer to a Roman numeral string.\n'
            '    >>> int_to_roman(58)\n'
            '    \'LVIII\'\n'
            '    >>> int_to_roman(1994)\n'
            '    \'MCMXCIV\'\n'
            '    """'
        ),
        "body": (
            '    val = [1000,900,500,400,100,90,50,40,10,9,5,4,1]\n'
            '    syms = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]\n'
            '    result = ""\n'
            '    for i, v in enumerate(val):\n'
            '        while num >= v:\n'
            '            result += syms[i]\n'
            '            num -= v\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def roman_to_int(s: str) -> int:\n'
            '    """Convert a Roman numeral string to an integer.\n'
            '    >>> roman_to_int("MCMXCIV")\n'
            '    1994\n'
            '    >>> roman_to_int("III")\n'
            '    3\n'
            '    """'
        ),
        "body": (
            '    mapping = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}\n'
            '    total = 0\n'
            '    prev = 0\n'
            '    for ch in reversed(s):\n'
            '        curr = mapping[ch]\n'
            '        if curr < prev:\n'
            '            total -= curr\n'
            '        else:\n'
            '            total += curr\n'
            '        prev = curr\n'
            '    return total'
        ),
    },
    {
        "prompt": (
            'def is_palindrome_num(n: int) -> bool:\n'
            '    """Return True if integer reads the same forwards and backwards.\n'
            '    >>> is_palindrome_num(121)\n'
            '    True\n'
            '    >>> is_palindrome_num(-121)\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    if n < 0:\n'
            '        return False\n'
            '    s = str(n)\n'
            '    return s == s[::-1]'
        ),
    },
    {
        "prompt": (
            'def catalan(n: int) -> int:\n'
            '    """Return the n-th Catalan number.\n'
            '    >>> catalan(0)\n'
            '    1\n'
            '    >>> catalan(5)\n'
            '    42\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    return math.comb(2 * n, n) // (n + 1)'
        ),
    },
    {
        "prompt": (
            'def count_primes_up_to(n: int) -> int:\n'
            '    """Return count of primes less than or equal to n.\n'
            '    >>> count_primes_up_to(10)\n'
            '    4\n'
            '    >>> count_primes_up_to(1)\n'
            '    0\n'
            '    """'
        ),
        "body": (
            '    if n < 2:\n'
            '        return 0\n'
            '    sieve = [True] * (n + 1)\n'
            '    sieve[0] = sieve[1] = False\n'
            '    for i in range(2, int(n**0.5) + 1):\n'
            '        if sieve[i]:\n'
            '            for j in range(i*i, n + 1, i):\n'
            '                sieve[j] = False\n'
            '    return sum(sieve)'
        ),
    },
    {
        "prompt": (
            'def phi(n: int) -> int:\n'
            "    \"\"\"Euler's totient function: count integers in [1,n] coprime to n.\n"
            '    >>> phi(9)\n'
            '    6\n'
            '    >>> phi(1)\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    result = n\n'
            '    p = 2\n'
            '    temp = n\n'
            '    while p * p <= temp:\n'
            '        if temp % p == 0:\n'
            '            while temp % p == 0:\n'
            '                temp //= p\n'
            '            result -= result // p\n'
            '        p += 1\n'
            '    if temp > 1:\n'
            '        result -= result // temp\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def binomial(n: int, k: int) -> int:\n'
            '    """Compute the binomial coefficient C(n, k).\n'
            '    >>> binomial(5, 2)\n'
            '    10\n'
            '    >>> binomial(10, 0)\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    return math.comb(n, k)'
        ),
    },
    {
        "prompt": (
            'def integer_sqrt(n: int) -> int:\n'
            '    """Return the floor of the square root of n.\n'
            '    >>> integer_sqrt(16)\n'
            '    4\n'
            '    >>> integer_sqrt(17)\n'
            '    4\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    return int(math.isqrt(n))'
        ),
    },
    {
        "prompt": (
            'def sum_of_squares(n: int) -> int:\n'
            '    """Return the sum of squares of 1 to n.\n'
            '    >>> sum_of_squares(3)\n'
            '    14\n'
            '    >>> sum_of_squares(0)\n'
            '    0\n'
            '    """'
        ),
        "body": '    return n * (n + 1) * (2 * n + 1) // 6',
    },
    {
        "prompt": (
            'def triangular_number(n: int) -> int:\n'
            '    """Return the n-th triangular number.\n'
            '    >>> triangular_number(5)\n'
            '    15\n'
            '    >>> triangular_number(1)\n'
            '    1\n'
            '    """'
        ),
        "body": '    return n * (n + 1) // 2',
    },
    {
        "prompt": (
            'def extended_gcd(a: int, b: int) -> tuple:\n'
            '    """Return (gcd, x, y) such that a*x + b*y == gcd.\n'
            '    >>> g, x, y = extended_gcd(30, 20)\n'
            '    >>> g\n'
            '    10\n'
            '    """'
        ),
        "body": (
            '    if b == 0:\n'
            '        return a, 1, 0\n'
            '    g, x1, y1 = extended_gcd(b, a % b)\n'
            '    return g, y1, x1 - (a // b) * y1'
        ),
    },
    {
        "prompt": (
            'def nth_prime(n: int) -> int:\n'
            '    """Return the n-th prime number (1-indexed).\n'
            '    >>> nth_prime(1)\n'
            '    2\n'
            '    >>> nth_prime(6)\n'
            '    13\n'
            '    """'
        ),
        "body": (
            '    count = 0\n'
            '    candidate = 2\n'
            '    while True:\n'
            '        is_p = all(candidate % i != 0 for i in range(2, int(candidate**0.5) + 1))\n'
            '        if is_p:\n'
            '            count += 1\n'
            '            if count == n:\n'
            '                return candidate\n'
            '        candidate += 1'
        ),
    },
    {
        "prompt": (
            'def count_set_bits(n: int) -> int:\n'
            '    """Return the number of 1-bits in the binary representation.\n'
            '    >>> count_set_bits(13)\n'
            '    3\n'
            '    >>> count_set_bits(0)\n'
            '    0\n'
            '    """'
        ),
        "body": '    return bin(n).count("1")',
    },
    {
        "prompt": (
            'def is_power_of_two(n: int) -> bool:\n'
            '    """Return True if n is a positive power of two.\n'
            '    >>> is_power_of_two(8)\n'
            '    True\n'
            '    >>> is_power_of_two(6)\n'
            '    False\n'
            '    """'
        ),
        "body": '    return n > 0 and (n & (n - 1)) == 0',
    },
    {
        "prompt": (
            'def digit_root(n: int) -> int:\n'
            '    """Return digital root of n (repeatedly sum digits until single digit).\n'
            '    >>> digit_root(493)\n'
            '    7\n'
            '    >>> digit_root(0)\n'
            '    0\n'
            '    """'
        ),
        "body": (
            '    if n == 0:\n'
            '        return 0\n'
            '    return 1 + (n - 1) % 9'
        ),
    },
]

# ---------------------------------------------------------------------------
# 2. String processing
# ---------------------------------------------------------------------------
STRING_EXAMPLES = [
    {
        "prompt": (
            'def is_palindrome(s: str) -> bool:\n'
            '    """Return True if s is a palindrome (case-insensitive, alphanumeric only).\n'
            '    >>> is_palindrome("A man a plan a canal Panama")\n'
            '    True\n'
            '    >>> is_palindrome("hello")\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    cleaned = re.sub(r"[^a-zA-Z0-9]", "", s).lower()\n'
            '    return cleaned == cleaned[::-1]'
        ),
    },
    {
        "prompt": (
            'def count_vowels(s: str) -> int:\n'
            '    """Count vowels in string (case-insensitive).\n'
            '    >>> count_vowels("Hello World")\n'
            '    3\n'
            '    """'
        ),
        "body": '    return sum(1 for c in s.lower() if c in "aeiou")',
    },
    {
        "prompt": (
            'def reverse_words(sentence: str) -> str:\n'
            '    """Reverse the order of words in a sentence.\n'
            '    >>> reverse_words("hello world")\n'
            '    \'world hello\'\n'
            '    """'
        ),
        "body": '    return " ".join(sentence.split()[::-1])',
    },
    {
        "prompt": (
            'def capitalize_words(s: str) -> str:\n'
            '    """Capitalize the first letter of each word.\n'
            '    >>> capitalize_words("hello world foo")\n'
            '    \'Hello World Foo\'\n'
            '    """'
        ),
        "body": '    return " ".join(w.capitalize() for w in s.split())',
    },
    {
        "prompt": (
            'def remove_punctuation(s: str) -> str:\n'
            '    """Remove all punctuation from a string.\n'
            '    >>> remove_punctuation("Hello, World!")\n'
            '    \'Hello World\'\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    return re.sub(r"[^\\w\\s]", "", s)'
        ),
    },
    {
        "prompt": (
            'def extract_emails(text: str) -> list:\n'
            '    """Extract all email addresses from text.\n'
            '    >>> extract_emails("Contact alice@example.com or bob@test.org")\n'
            '    [\'alice@example.com\', \'bob@test.org\']\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    return re.findall(r\'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\', text)'
        ),
    },
    {
        "prompt": (
            'def extract_urls(text: str) -> list:\n'
            '    """Extract all http/https URLs from text.\n'
            '    >>> extract_urls("Visit https://example.com and http://test.org")\n'
            '    [\'https://example.com\', \'http://test.org\']\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    return re.findall(r\'https?://[^\\s]+\', text)'
        ),
    },
    {
        "prompt": (
            'def is_valid_ipv4(ip: str) -> bool:\n'
            '    """Return True if ip is a valid IPv4 address.\n'
            '    >>> is_valid_ipv4("192.168.1.1")\n'
            '    True\n'
            '    >>> is_valid_ipv4("256.1.1.1")\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    pattern = r\'^((25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\.){3}(25[0-5]|2[0-4]\\d|[01]?\\d\\d?)$\'\n'
            '    return bool(re.match(pattern, ip))'
        ),
    },
    {
        "prompt": (
            'def snake_to_camel(s: str) -> str:\n'
            '    """Convert snake_case string to camelCase.\n'
            '    >>> snake_to_camel("hello_world_foo")\n'
            '    \'helloWorldFoo\'\n'
            '    """'
        ),
        "body": (
            '    parts = s.split("_")\n'
            '    return parts[0] + "".join(w.capitalize() for w in parts[1:])'
        ),
    },
    {
        "prompt": (
            'def camel_to_snake(s: str) -> str:\n'
            '    """Convert camelCase string to snake_case.\n'
            '    >>> camel_to_snake("helloWorldFoo")\n'
            '    \'hello_world_foo\'\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    return re.sub(r\'([A-Z])\', r\'_\\1\', s).lower().lstrip("_")'
        ),
    },
    {
        "prompt": (
            'def count_words(s: str) -> dict:\n'
            '    """Return a dict mapping each word to its frequency.\n'
            '    >>> count_words("the cat sat on the mat")\n'
            '    {\'the\': 2, \'cat\': 1, \'sat\': 1, \'on\': 1, \'mat\': 1}\n'
            '    """'
        ),
        "body": (
            '    from collections import Counter\n'
            '    return dict(Counter(s.lower().split()))'
        ),
    },
    {
        "prompt": (
            'def longest_word(s: str) -> str:\n'
            '    """Return the longest word in the string (first if tied).\n'
            '    >>> longest_word("I love programming")\n'
            '    \'programming\'\n'
            '    """'
        ),
        "body": (
            '    words = s.split()\n'
            '    return max(words, key=len) if words else ""'
        ),
    },
    {
        "prompt": (
            'def compress_string(s: str) -> str:\n'
            '    """Run-length encode: "aaabbc" -> "a3b2c1".\n'
            '    >>> compress_string("aaabbc")\n'
            '    \'a3b2c1\'\n'
            '    """'
        ),
        "body": (
            '    if not s:\n'
            '        return ""\n'
            '    result = []\n'
            '    count = 1\n'
            '    for i in range(1, len(s)):\n'
            '        if s[i] == s[i-1]:\n'
            '            count += 1\n'
            '        else:\n'
            '            result.append(s[i-1] + str(count))\n'
            '            count = 1\n'
            '    result.append(s[-1] + str(count))\n'
            '    return "".join(result)'
        ),
    },
    {
        "prompt": (
            'def decompress_string(s: str) -> str:\n'
            '    """Decode run-length encoded string: "a3b2c1" -> "aaabbc".\n'
            '    >>> decompress_string("a3b2c1")\n'
            '    \'aaabbc\'\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    return "".join(ch * int(n) for ch, n in re.findall(r\'([a-zA-Z])(\\d+)\', s))'
        ),
    },
    {
        "prompt": (
            'def is_anagram(s1: str, s2: str) -> bool:\n'
            '    """Return True if s1 and s2 are anagrams (case-insensitive).\n'
            '    >>> is_anagram("listen", "silent")\n'
            '    True\n'
            '    >>> is_anagram("hello", "world")\n'
            '    False\n'
            '    """'
        ),
        "body": '    return sorted(s1.lower()) == sorted(s2.lower())',
    },
    {
        "prompt": (
            'def truncate(s: str, length: int, suffix: str = "...") -> str:\n'
            '    """Truncate s to at most length characters, appending suffix if truncated.\n'
            '    >>> truncate("hello world", 8)\n'
            '    \'hello...\'\n'
            '    """'
        ),
        "body": (
            '    if len(s) <= length:\n'
            '        return s\n'
            '    return s[:length - len(suffix)] + suffix'
        ),
    },
    {
        "prompt": (
            'def extract_digits(s: str) -> str:\n'
            '    """Return only the digit characters from s.\n'
            '    >>> extract_digits("a1b2c3")\n'
            '    \'123\'\n'
            '    """'
        ),
        "body": '    return "".join(c for c in s if c.isdigit())',
    },
    {
        "prompt": (
            'def title_to_snake(title: str) -> str:\n'
            '    """Convert "Title Case" string to snake_case.\n'
            '    >>> title_to_snake("Hello World")\n'
            '    \'hello_world\'\n'
            '    """'
        ),
        "body": '    return title.lower().replace(" ", "_")',
    },
    {
        "prompt": (
            'def find_repeated_chars(s: str) -> list:\n'
            '    """Return sorted list of characters that appear more than once.\n'
            '    >>> find_repeated_chars("abcabd")\n'
            '    [\'a\', \'b\']\n'
            '    """'
        ),
        "body": (
            '    from collections import Counter\n'
            '    return sorted(c for c, cnt in Counter(s).items() if cnt > 1)'
        ),
    },
    {
        "prompt": (
            'def is_balanced_brackets(s: str) -> bool:\n'
            '    """Return True if brackets (), [], {} are balanced in s.\n'
            '    >>> is_balanced_brackets("{[()]}")\n'
            '    True\n'
            '    >>> is_balanced_brackets("([)]")\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    stack = []\n'
            '    mapping = {")": "(", "]": "[", "}": "{"}\n'
            '    for ch in s:\n'
            '        if ch in "([{":\n'
            '            stack.append(ch)\n'
            '        elif ch in ")]}":\n'
            '            if not stack or stack[-1] != mapping[ch]:\n'
            '                return False\n'
            '            stack.pop()\n'
            '    return len(stack) == 0'
        ),
    },
    {
        "prompt": (
            'def reverse_string(s: str) -> str:\n'
            '    """Return s reversed.\n'
            '    >>> reverse_string("hello")\n'
            '    \'olleh\'\n'
            '    """'
        ),
        "body": '    return s[::-1]',
    },
    {
        "prompt": (
            'def mask_credit_card(number: str) -> str:\n'
            '    """Mask all but the last 4 digits of a credit card number string.\n'
            '    >>> mask_credit_card("1234567890123456")\n'
            '    \'************3456\'\n'
            '    """'
        ),
        "body": '    return "*" * (len(number) - 4) + number[-4:]',
    },
    {
        "prompt": (
            'def count_substrings(s: str, sub: str) -> int:\n'
            '    """Count non-overlapping occurrences of sub in s.\n'
            '    >>> count_substrings("banana", "an")\n'
            '    2\n'
            '    """'
        ),
        "body": '    return s.count(sub)',
    },
    {
        "prompt": (
            'def remove_whitespace(s: str) -> str:\n'
            '    """Remove all whitespace characters from a string.\n'
            '    >>> remove_whitespace("  hello   world  ")\n'
            '    \'helloworld\'\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    return re.sub(r"\\s+", "", s)'
        ),
    },
    {
        "prompt": (
            'def wrap_text(s: str, width: int) -> list:\n'
            '    """Wrap s into lines of at most width characters at word boundaries.\n'
            '    >>> wrap_text("hello world foo bar", 10)\n'
            '    [\'hello\', \'world foo\', \'bar\']\n'
            '    """'
        ),
        "body": (
            '    import textwrap\n'
            '    return textwrap.wrap(s, width)'
        ),
    },
    {
        "prompt": (
            'def rot13(s: str) -> str:\n'
            '    """Apply ROT13 encoding to the alphabetic characters of s.\n'
            '    >>> rot13("Hello")\n'
            '    \'Uryyb\'\n'
            '    """'
        ),
        "body": (
            '    import codecs\n'
            '    return codecs.encode(s, "rot_13")'
        ),
    },
    {
        "prompt": (
            'def is_numeric_string(s: str) -> bool:\n'
            '    """Return True if s represents a valid integer or float.\n'
            '    >>> is_numeric_string("3.14")\n'
            '    True\n'
            '    >>> is_numeric_string("abc")\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    try:\n'
            '        float(s)\n'
            '        return True\n'
            '    except ValueError:\n'
            '        return False'
        ),
    },
    {
        "prompt": (
            'def repeat_chars(s: str, n: int) -> str:\n'
            '    """Repeat each character in s n times.\n'
            '    >>> repeat_chars("abc", 2)\n'
            '    \'aabbcc\'\n'
            '    """'
        ),
        "body": '    return "".join(c * n for c in s)',
    },
    {
        "prompt": (
            'def parse_csv_line(line: str) -> list:\n'
            '    """Parse a CSV line into a list of fields (handles quoted fields).\n'
            '    >>> parse_csv_line(\'a,b,"c,d",e\')\n'
            '    [\'a\', \'b\', \'c,d\', \'e\']\n'
            '    """'
        ),
        "body": (
            '    import csv\n'
            '    import io\n'
            '    reader = csv.reader(io.StringIO(line))\n'
            '    return next(reader)'
        ),
    },
    {
        "prompt": (
            'def normalize_spaces(s: str) -> str:\n'
            '    """Replace multiple consecutive spaces with a single space and strip.\n'
            '    >>> normalize_spaces("  hello   world  ")\n'
            '    \'hello world\'\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    return re.sub(r" +", " ", s).strip()'
        ),
    },
]

# ---------------------------------------------------------------------------
# 3. List / array operations
# ---------------------------------------------------------------------------
LIST_EXAMPLES = [
    {
        "prompt": (
            'def two_sum(nums: list, target: int) -> list:\n'
            '    """Return indices of two numbers that add up to target.\n'
            '    >>> two_sum([2, 7, 11, 15], 9)\n'
            '    [0, 1]\n'
            '    """'
        ),
        "body": (
            '    seen = {}\n'
            '    for i, v in enumerate(nums):\n'
            '        if target - v in seen:\n'
            '            return [seen[target - v], i]\n'
            '        seen[v] = i\n'
            '    return []'
        ),
    },
    {
        "prompt": (
            'def max_subarray_sum(nums: list) -> int:\n'
            '    """Return the maximum contiguous subarray sum (Kadane\'s algorithm).\n'
            '    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n'
            '    6\n'
            '    """'
        ),
        "body": (
            '    max_sum = current = nums[0]\n'
            '    for n in nums[1:]:\n'
            '        current = max(n, current + n)\n'
            '        max_sum = max(max_sum, current)\n'
            '    return max_sum'
        ),
    },
    {
        "prompt": (
            'def rotate_list(lst: list, k: int) -> list:\n'
            '    """Rotate list right by k positions.\n'
            '    >>> rotate_list([1, 2, 3, 4, 5], 2)\n'
            '    [4, 5, 1, 2, 3]\n'
            '    """'
        ),
        "body": (
            '    if not lst:\n'
            '        return lst\n'
            '    k = k % len(lst)\n'
            '    return lst[-k:] + lst[:-k]'
        ),
    },
    {
        "prompt": (
            'def merge_sorted(a: list, b: list) -> list:\n'
            '    """Merge two sorted lists into one sorted list.\n'
            '    >>> merge_sorted([1, 3, 5], [2, 4, 6])\n'
            '    [1, 2, 3, 4, 5, 6]\n'
            '    """'
        ),
        "body": (
            '    result = []\n'
            '    i = j = 0\n'
            '    while i < len(a) and j < len(b):\n'
            '        if a[i] <= b[j]:\n'
            '            result.append(a[i]); i += 1\n'
            '        else:\n'
            '            result.append(b[j]); j += 1\n'
            '    return result + a[i:] + b[j:]'
        ),
    },
    {
        "prompt": (
            'def binary_search(lst: list, target: int) -> int:\n'
            '    """Return index of target in sorted lst, or -1 if not found.\n'
            '    >>> binary_search([1, 3, 5, 7, 9], 5)\n'
            '    2\n'
            '    >>> binary_search([1, 3, 5], 4)\n'
            '    -1\n'
            '    """'
        ),
        "body": (
            '    lo, hi = 0, len(lst) - 1\n'
            '    while lo <= hi:\n'
            '        mid = (lo + hi) // 2\n'
            '        if lst[mid] == target:\n'
            '            return mid\n'
            '        elif lst[mid] < target:\n'
            '            lo = mid + 1\n'
            '        else:\n'
            '            hi = mid - 1\n'
            '    return -1'
        ),
    },
    {
        "prompt": (
            'def product_except_self(nums: list) -> list:\n'
            '    """Return list where each element is the product of all others.\n'
            '    >>> product_except_self([1, 2, 3, 4])\n'
            '    [24, 12, 8, 6]\n'
            '    """'
        ),
        "body": (
            '    n = len(nums)\n'
            '    left = [1] * n\n'
            '    right = [1] * n\n'
            '    for i in range(1, n):\n'
            '        left[i] = left[i-1] * nums[i-1]\n'
            '    for i in range(n-2, -1, -1):\n'
            '        right[i] = right[i+1] * nums[i+1]\n'
            '    return [left[i] * right[i] for i in range(n)]'
        ),
    },
    {
        "prompt": (
            'def remove_element(lst: list, val: int) -> list:\n'
            '    """Remove all occurrences of val from list.\n'
            '    >>> remove_element([1, 2, 3, 2, 4], 2)\n'
            '    [1, 3, 4]\n'
            '    """'
        ),
        "body": '    return [x for x in lst if x != val]',
    },
    {
        "prompt": (
            'def flatten(nested: list) -> list:\n'
            '    """Flatten one level of nesting in a list.\n'
            '    >>> flatten([[1, 2], [3, 4], [5]])\n'
            '    [1, 2, 3, 4, 5]\n'
            '    """'
        ),
        "body": (
            '    from itertools import chain\n'
            '    return list(chain.from_iterable(nested))'
        ),
    },
    {
        "prompt": (
            'def zip_with(f, lst1: list, lst2: list) -> list:\n'
            '    """Apply binary function f element-wise to two lists.\n'
            '    >>> zip_with(lambda a, b: a + b, [1, 2, 3], [4, 5, 6])\n'
            '    [5, 7, 9]\n'
            '    """'
        ),
        "body": '    return [f(a, b) for a, b in zip(lst1, lst2)]',
    },
    {
        "prompt": (
            'def sliding_window_max(nums: list, k: int) -> list:\n'
            '    """Return maximum value in each sliding window of size k.\n'
            '    >>> sliding_window_max([1,3,-1,-3,5,3,6,7], 3)\n'
            '    [3, 3, 5, 5, 6, 7]\n'
            '    """'
        ),
        "body": (
            '    from collections import deque\n'
            '    dq = deque()\n'
            '    result = []\n'
            '    for i, v in enumerate(nums):\n'
            '        while dq and nums[dq[-1]] <= v:\n'
            '            dq.pop()\n'
            '        dq.append(i)\n'
            '        if dq[0] == i - k:\n'
            '            dq.popleft()\n'
            '        if i >= k - 1:\n'
            '            result.append(nums[dq[0]])\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def cumulative_sum(lst: list) -> list:\n'
            '    """Return list of cumulative sums.\n'
            '    >>> cumulative_sum([1, 2, 3, 4])\n'
            '    [1, 3, 6, 10]\n'
            '    """'
        ),
        "body": (
            '    from itertools import accumulate\n'
            '    return list(accumulate(lst))'
        ),
    },
    {
        "prompt": (
            'def interleave(lst1: list, lst2: list) -> list:\n'
            '    """Interleave two lists element by element.\n'
            '    >>> interleave([1, 3, 5], [2, 4, 6])\n'
            '    [1, 2, 3, 4, 5, 6]\n'
            '    """'
        ),
        "body": (
            '    from itertools import chain, zip_longest\n'
            '    sentinel = object()\n'
            '    result = []\n'
            '    for a, b in zip_longest(lst1, lst2, fillvalue=sentinel):\n'
            '        if a is not sentinel: result.append(a)\n'
            '        if b is not sentinel: result.append(b)\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def deduplicate(lst: list) -> list:\n'
            '    """Remove duplicates while preserving order.\n'
            '    >>> deduplicate([3, 1, 4, 1, 5, 9, 2, 6, 5])\n'
            '    [3, 1, 4, 5, 9, 2, 6]\n'
            '    """'
        ),
        "body": (
            '    seen = set()\n'
            '    result = []\n'
            '    for x in lst:\n'
            '        if x not in seen:\n'
            '            seen.add(x)\n'
            '            result.append(x)\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def find_missing(lst: list, n: int) -> int:\n'
            '    """Find the missing integer in a list of integers from 1 to n.\n'
            '    >>> find_missing([1, 2, 4, 5], 5)\n'
            '    3\n'
            '    """'
        ),
        "body": '    return n * (n + 1) // 2 - sum(lst)',
    },
    {
        "prompt": (
            'def group_by(lst: list, key_fn) -> dict:\n'
            '    """Group list elements by the result of key_fn.\n'
            '    >>> group_by([1, 2, 3, 4, 5, 6], lambda x: x % 2 == 0)\n'
            '    {False: [1, 3, 5], True: [2, 4, 6]}\n'
            '    """'
        ),
        "body": (
            '    from collections import defaultdict\n'
            '    result = defaultdict(list)\n'
            '    for item in lst:\n'
            '        result[key_fn(item)].append(item)\n'
            '    return dict(result)'
        ),
    },
    {
        "prompt": (
            'def sort_by_key(lst: list, key_fn) -> list:\n'
            '    """Return list sorted by key_fn.\n'
            '    >>> sort_by_key(["banana", "apple", "cherry"], len)\n'
            '    [\'apple\', \'banana\', \'cherry\']\n'
            '    """'
        ),
        "body": '    return sorted(lst, key=key_fn)',
    },
    {
        "prompt": (
            'def all_subsets(lst: list) -> list:\n'
            '    """Return all subsets of lst as sorted list of lists.\n'
            '    >>> sorted(all_subsets([1, 2]))\n'
            '    [[], [1], [1, 2], [2]]\n'
            '    """'
        ),
        "body": (
            '    from itertools import combinations\n'
            '    result = []\n'
            '    for r in range(len(lst) + 1):\n'
            '        result.extend(list(c) for c in combinations(lst, r))\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def moving_average(lst: list, window: int) -> list:\n'
            '    """Return moving average with given window size.\n'
            '    >>> [round(x, 2) for x in moving_average([1, 2, 3, 4, 5], 3)]\n'
            '    [2.0, 3.0, 4.0]\n'
            '    """'
        ),
        "body": (
            '    return [sum(lst[i:i+window]) / window\n'
            '            for i in range(len(lst) - window + 1)]'
        ),
    },
    {
        "prompt": (
            'def nth_largest(lst: list, n: int) -> int:\n'
            '    """Return the n-th largest element (1-indexed).\n'
            '    >>> nth_largest([3, 1, 4, 1, 5, 9, 2, 6], 3)\n'
            '    5\n'
            '    """'
        ),
        "body": (
            '    import heapq\n'
            '    return heapq.nlargest(n, lst)[n-1]'
        ),
    },
    {
        "prompt": (
            'def partition(lst: list, pivot) -> tuple:\n'
            '    """Partition lst into (less, equal, greater) than pivot.\n'
            '    >>> partition([3, 1, 4, 1, 5, 9, 2, 6], 4)\n'
            '    ([3, 1, 1, 2], [4], [5, 9, 6])\n'
            '    """'
        ),
        "body": (
            '    less = [x for x in lst if x < pivot]\n'
            '    equal = [x for x in lst if x == pivot]\n'
            '    greater = [x for x in lst if x > pivot]\n'
            '    return less, equal, greater'
        ),
    },
    {
        "prompt": (
            'def transpose(matrix: list) -> list:\n'
            '    """Transpose a 2D list (list of lists).\n'
            '    >>> transpose([[1, 2, 3], [4, 5, 6]])\n'
            '    [[1, 4], [2, 5], [3, 6]]\n'
            '    """'
        ),
        "body": '    return [list(row) for row in zip(*matrix)]',
    },
    {
        "prompt": (
            'def count_inversions(lst: list) -> int:\n'
            '    """Count the number of inversions (i<j but lst[i]>lst[j]).\n'
            '    >>> count_inversions([3, 1, 2])\n'
            '    2\n'
            '    """'
        ),
        "body": (
            '    count = 0\n'
            '    n = len(lst)\n'
            '    for i in range(n):\n'
            '        for j in range(i+1, n):\n'
            '            if lst[i] > lst[j]:\n'
            '                count += 1\n'
            '    return count'
        ),
    },
]

# ---------------------------------------------------------------------------
# 4. Recursive algorithms
# ---------------------------------------------------------------------------
RECURSIVE_EXAMPLES = [
    {
        "prompt": (
            'def power(base: float, exp: int) -> float:\n'
            '    """Compute base**exp recursively (exp >= 0).\n'
            '    >>> power(2, 10)\n'
            '    1024\n'
            '    >>> power(3, 0)\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    if exp == 0:\n'
            '        return 1\n'
            '    if exp % 2 == 0:\n'
            '        half = power(base, exp // 2)\n'
            '        return half * half\n'
            '    return base * power(base, exp - 1)'
        ),
    },
    {
        "prompt": (
            'def merge_sort(lst: list) -> list:\n'
            '    """Sort a list using merge sort.\n'
            '    >>> merge_sort([3, 1, 4, 1, 5, 9, 2, 6])\n'
            '    [1, 1, 2, 3, 4, 5, 6, 9]\n'
            '    """'
        ),
        "body": (
            '    if len(lst) <= 1:\n'
            '        return lst\n'
            '    mid = len(lst) // 2\n'
            '    left = merge_sort(lst[:mid])\n'
            '    right = merge_sort(lst[mid:])\n'
            '    result = []\n'
            '    i = j = 0\n'
            '    while i < len(left) and j < len(right):\n'
            '        if left[i] <= right[j]:\n'
            '            result.append(left[i]); i += 1\n'
            '        else:\n'
            '            result.append(right[j]); j += 1\n'
            '    return result + left[i:] + right[j:]'
        ),
    },
    {
        "prompt": (
            'def quick_sort(lst: list) -> list:\n'
            '    """Sort a list using quicksort.\n'
            '    >>> quick_sort([3, 1, 4, 1, 5, 9, 2, 6])\n'
            '    [1, 1, 2, 3, 4, 5, 6, 9]\n'
            '    """'
        ),
        "body": (
            '    if len(lst) <= 1:\n'
            '        return lst\n'
            '    pivot = lst[len(lst) // 2]\n'
            '    left = [x for x in lst if x < pivot]\n'
            '    mid = [x for x in lst if x == pivot]\n'
            '    right = [x for x in lst if x > pivot]\n'
            '    return quick_sort(left) + mid + quick_sort(right)'
        ),
    },
    {
        "prompt": (
            'def tower_of_hanoi(n: int, source: str = "A", target: str = "C", '
            'auxiliary: str = "B") -> list:\n'
            '    """Return list of moves to solve Tower of Hanoi.\n'
            '    >>> tower_of_hanoi(2)\n'
            '    [(\'A\', \'B\'), (\'A\', \'C\'), (\'B\', \'C\')]\n'
            '    """'
        ),
        "body": (
            '    if n == 0:\n'
            '        return []\n'
            '    moves = tower_of_hanoi(n - 1, source, auxiliary, target)\n'
            '    moves.append((source, target))\n'
            '    moves += tower_of_hanoi(n - 1, auxiliary, target, source)\n'
            '    return moves'
        ),
    },
    {
        "prompt": (
            'def flatten_deep(nested) -> list:\n'
            '    """Recursively flatten a deeply nested list.\n'
            '    >>> flatten_deep([1, [2, [3, [4, [5]]]]])\n'
            '    [1, 2, 3, 4, 5]\n'
            '    """'
        ),
        "body": (
            '    result = []\n'
            '    if isinstance(nested, list):\n'
            '        for item in nested:\n'
            '            result.extend(flatten_deep(item))\n'
            '    else:\n'
            '        result.append(nested)\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def max_depth(nested: list) -> int:\n'
            '    """Return the maximum depth of a nested list.\n'
            '    >>> max_depth([1, [2, [3]]])\n'
            '    3\n'
            '    >>> max_depth([1, 2, 3])\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    if not isinstance(nested, list):\n'
            '        return 0\n'
            '    if not nested:\n'
            '        return 1\n'
            '    return 1 + max(max_depth(item) for item in nested)'
        ),
    },
    {
        "prompt": (
            'def count_tree_nodes(root: dict) -> int:\n'
            '    """Count nodes in a tree represented as {val, children: [...]}.\n'
            '    >>> count_tree_nodes({"val": 1, "children": [{"val": 2, "children": []}, {"val": 3, "children": []}]})\n'
            '    3\n'
            '    """'
        ),
        "body": (
            '    if not root:\n'
            '        return 0\n'
            '    return 1 + sum(count_tree_nodes(child) for child in root.get("children", []))'
        ),
    },
    {
        "prompt": (
            'def sum_nested(nested) -> int:\n'
            '    """Return the sum of all integers in a deeply nested list.\n'
            '    >>> sum_nested([1, [2, [3, [4]]]])\n'
            '    10\n'
            '    """'
        ),
        "body": (
            '    if isinstance(nested, int):\n'
            '        return nested\n'
            '    return sum(sum_nested(item) for item in nested)'
        ),
    },
    {
        "prompt": (
            'def generate_parentheses(n: int) -> list:\n'
            '    """Return all valid combinations of n pairs of parentheses.\n'
            '    >>> sorted(generate_parentheses(2))\n'
            '    [\'(())\', \'()(\']\n'
            '    >>> "((()))" in generate_parentheses(3)\n'
            '    True\n'
            '    """'
        ),
        "body": (
            '    result = []\n'
            '    def backtrack(s, open_count, close_count):\n'
            '        if len(s) == 2 * n:\n'
            '            result.append(s)\n'
            '            return\n'
            '        if open_count < n:\n'
            '            backtrack(s + "(", open_count + 1, close_count)\n'
            '        if close_count < open_count:\n'
            '            backtrack(s + ")", open_count, close_count + 1)\n'
            '    backtrack("", 0, 0)\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def coin_change(coins: list, amount: int) -> int:\n'
            '    """Return min coins needed to make amount, or -1 if impossible.\n'
            '    >>> coin_change([1, 5, 10], 27)\n'
            '    4\n'
            '    >>> coin_change([2], 3)\n'
            '    -1\n'
            '    """'
        ),
        "body": (
            '    dp = [float("inf")] * (amount + 1)\n'
            '    dp[0] = 0\n'
            '    for coin in coins:\n'
            '        for x in range(coin, amount + 1):\n'
            '            dp[x] = min(dp[x], dp[x - coin] + 1)\n'
            '    return dp[amount] if dp[amount] != float("inf") else -1'
        ),
    },
    {
        "prompt": (
            'def longest_common_subsequence(s1: str, s2: str) -> int:\n'
            '    """Return the length of the longest common subsequence.\n'
            '    >>> longest_common_subsequence("abcde", "ace")\n'
            '    3\n'
            '    """'
        ),
        "body": (
            '    m, n = len(s1), len(s2)\n'
            '    dp = [[0] * (n + 1) for _ in range(m + 1)]\n'
            '    for i in range(1, m + 1):\n'
            '        for j in range(1, n + 1):\n'
            '            if s1[i-1] == s2[j-1]:\n'
            '                dp[i][j] = dp[i-1][j-1] + 1\n'
            '            else:\n'
            '                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n'
            '    return dp[m][n]'
        ),
    },
    {
        "prompt": (
            'def knapsack(weights: list, values: list, capacity: int) -> int:\n'
            '    """0/1 knapsack: return max value without exceeding capacity.\n'
            '    >>> knapsack([2, 3, 4, 5], [3, 4, 5, 6], 5)\n'
            '    7\n'
            '    """'
        ),
        "body": (
            '    n = len(weights)\n'
            '    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n'
            '    for i in range(1, n + 1):\n'
            '        for w in range(capacity + 1):\n'
            '            dp[i][w] = dp[i-1][w]\n'
            '            if weights[i-1] <= w:\n'
            '                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])\n'
            '    return dp[n][capacity]'
        ),
    },
    {
        "prompt": (
            'def edit_distance(s1: str, s2: str) -> int:\n'
            '    """Return the minimum edit distance (Levenshtein) between s1 and s2.\n'
            '    >>> edit_distance("kitten", "sitting")\n'
            '    3\n'
            '    """'
        ),
        "body": (
            '    m, n = len(s1), len(s2)\n'
            '    dp = list(range(n + 1))\n'
            '    for i in range(1, m + 1):\n'
            '        prev = dp[:]\n'
            '        dp[0] = i\n'
            '        for j in range(1, n + 1):\n'
            '            if s1[i-1] == s2[j-1]:\n'
            '                dp[j] = prev[j-1]\n'
            '            else:\n'
            '                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])\n'
            '    return dp[n]'
        ),
    },
    {
        "prompt": (
            'def num_ways_stairs(n: int) -> int:\n'
            '    """Number of ways to climb n stairs taking 1 or 2 steps at a time.\n'
            '    >>> num_ways_stairs(4)\n'
            '    5\n'
            '    >>> num_ways_stairs(1)\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    if n <= 1:\n'
            '        return 1\n'
            '    a, b = 1, 1\n'
            '    for _ in range(2, n + 1):\n'
            '        a, b = b, a + b\n'
            '    return b'
        ),
    },
    {
        "prompt": (
            'def permutations_recursive(lst: list) -> list:\n'
            '    """Return all permutations of lst recursively.\n'
            '    >>> sorted(permutations_recursive([1, 2, 3]))\n'
            '    [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]\n'
            '    """'
        ),
        "body": (
            '    if len(lst) <= 1:\n'
            '        return [lst[:]]\n'
            '    result = []\n'
            '    for i, v in enumerate(lst):\n'
            '        rest = lst[:i] + lst[i+1:]\n'
            '        for p in permutations_recursive(rest):\n'
            '            result.append([v] + p)\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def longest_increasing_subsequence(nums: list) -> int:\n'
            '    """Return the length of the longest strictly increasing subsequence.\n'
            '    >>> longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18])\n'
            '    4\n'
            '    """'
        ),
        "body": (
            '    if not nums:\n'
            '        return 0\n'
            '    dp = [1] * len(nums)\n'
            '    for i in range(1, len(nums)):\n'
            '        for j in range(i):\n'
            '            if nums[j] < nums[i]:\n'
            '                dp[i] = max(dp[i], dp[j] + 1)\n'
            '    return max(dp)'
        ),
    },
]

# ---------------------------------------------------------------------------
# 5. Dictionary / set operations
# ---------------------------------------------------------------------------
DICT_SET_EXAMPLES = [
    {
        "prompt": (
            'def invert_dict(d: dict) -> dict:\n'
            '    """Return a new dict with keys and values swapped.\n'
            '    >>> invert_dict({"a": 1, "b": 2})\n'
            '    {1: \'a\', 2: \'b\'}\n'
            '    """'
        ),
        "body": '    return {v: k for k, v in d.items()}',
    },
    {
        "prompt": (
            'def merge_dicts(*dicts) -> dict:\n'
            '    """Merge multiple dicts; later values override earlier ones.\n'
            '    >>> merge_dicts({"a": 1}, {"b": 2}, {"a": 3})\n'
            '    {\'a\': 3, \'b\': 2}\n'
            '    """'
        ),
        "body": (
            '    result = {}\n'
            '    for d in dicts:\n'
            '        result.update(d)\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def dict_diff(d1: dict, d2: dict) -> dict:\n'
            '    """Return keys in d1 whose values differ from d2 (or absent in d2).\n'
            '    >>> dict_diff({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 9})\n'
            '    {\'b\': 2, \'c\': 3}\n'
            '    """'
        ),
        "body": (
            '    return {k: v for k, v in d1.items() if d2.get(k) != v}'
        ),
    },
    {
        "prompt": (
            'def group_anagrams(words: list) -> list:\n'
            '    """Group words into lists of anagrams.\n'
            '    >>> sorted([sorted(g) for g in group_anagrams(["eat","tea","tan","ate","nat","bat"])])\n'
            '    [[\'ate\', \'eat\', \'tea\'], [\'bat\'], [\'nat\', \'tan\']]\n'
            '    """'
        ),
        "body": (
            '    from collections import defaultdict\n'
            '    groups = defaultdict(list)\n'
            '    for w in words:\n'
            '        groups[tuple(sorted(w))].append(w)\n'
            '    return list(groups.values())'
        ),
    },
    {
        "prompt": (
            'def word_frequency(text: str) -> dict:\n'
            '    """Return frequency dict of lowercased words in text.\n'
            '    >>> word_frequency("hello world hello")\n'
            '    {\'hello\': 2, \'world\': 1}\n'
            '    """'
        ),
        "body": (
            '    from collections import Counter\n'
            '    return dict(Counter(text.lower().split()))'
        ),
    },
    {
        "prompt": (
            'def symmetric_difference(s1: set, s2: set) -> set:\n'
            '    """Return elements in either set but not both.\n'
            '    >>> symmetric_difference({1, 2, 3}, {2, 3, 4})\n'
            '    {1, 4}\n'
            '    """'
        ),
        "body": '    return s1 ^ s2',
    },
    {
        "prompt": (
            'def power_set(s: set) -> list:\n'
            '    """Return all subsets of set s as list of frozensets.\n'
            '    >>> len(power_set({1, 2, 3}))\n'
            '    8\n'
            '    """'
        ),
        "body": (
            '    from itertools import combinations\n'
            '    lst = list(s)\n'
            '    result = []\n'
            '    for r in range(len(lst) + 1):\n'
            '        result.extend(frozenset(c) for c in combinations(lst, r))\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def top_n_keys(d: dict, n: int) -> list:\n'
            '    """Return the n keys with the highest values.\n'
            '    >>> top_n_keys({"a": 3, "b": 1, "c": 2}, 2)\n'
            '    [\'a\', \'c\']\n'
            '    """'
        ),
        "body": (
            '    return sorted(d, key=d.get, reverse=True)[:n]'
        ),
    },
    {
        "prompt": (
            'def nested_dict_get(d: dict, keys: list, default=None):\n'
            '    """Retrieve a value from a nested dict following a list of keys.\n'
            '    >>> nested_dict_get({"a": {"b": {"c": 42}}}, ["a", "b", "c"])\n'
            '    42\n'
            '    """'
        ),
        "body": (
            '    for key in keys:\n'
            '        if not isinstance(d, dict):\n'
            '            return default\n'
            '        d = d.get(key, default)\n'
            '    return d'
        ),
    },
    {
        "prompt": (
            'def count_unique(lst: list) -> int:\n'
            '    """Return the number of unique elements in lst.\n'
            '    >>> count_unique([1, 2, 2, 3, 3, 3])\n'
            '    3\n'
            '    """'
        ),
        "body": '    return len(set(lst))',
    },
    {
        "prompt": (
            'def most_frequent(lst: list):\n'
            '    """Return the most frequent element (first one if tied).\n'
            '    >>> most_frequent([1, 2, 2, 3])\n'
            '    2\n'
            '    """'
        ),
        "body": (
            '    from collections import Counter\n'
            '    return Counter(lst).most_common(1)[0][0]'
        ),
    },
    {
        "prompt": (
            'def dict_to_pairs(d: dict) -> list:\n'
            '    """Convert dict to sorted list of (key, value) tuples.\n'
            '    >>> dict_to_pairs({"b": 2, "a": 1})\n'
            '    [(\'a\', 1), (\'b\', 2)]\n'
            '    """'
        ),
        "body": '    return sorted(d.items())',
    },
    {
        "prompt": (
            'def filter_dict(d: dict, pred) -> dict:\n'
            '    """Keep only key-value pairs where pred(key, value) is True.\n'
            '    >>> filter_dict({"a": 1, "b": 2, "c": 3}, lambda k, v: v > 1)\n'
            '    {\'b\': 2, \'c\': 3}\n'
            '    """'
        ),
        "body": '    return {k: v for k, v in d.items() if pred(k, v)}',
    },
    {
        "prompt": (
            'def set_intersection(sets: list) -> set:\n'
            '    """Return the intersection of a list of sets.\n'
            '    >>> set_intersection([{1,2,3}, {2,3,4}, {3,4,5}])\n'
            '    {3}\n'
            '    """'
        ),
        "body": (
            '    if not sets:\n'
            '        return set()\n'
            '    result = sets[0]\n'
            '    for s in sets[1:]:\n'
            '        result = result & s\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def flatten_dict(d: dict, prefix: str = "") -> dict:\n'
            '    """Flatten a nested dict with dot-separated keys.\n'
            '    >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})\n'
            '    {\'a.b\': 1, \'a.c\': 2, \'d\': 3}\n'
            '    """'
        ),
        "body": (
            '    result = {}\n'
            '    for k, v in d.items():\n'
            '        new_key = f"{prefix}.{k}" if prefix else k\n'
            '        if isinstance(v, dict):\n'
            '            result.update(flatten_dict(v, new_key))\n'
            '        else:\n'
            '            result[new_key] = v\n'
            '    return result'
        ),
    },
]

# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------
EDGE_CASE_EXAMPLES = [
    {
        "prompt": (
            'def safe_get(lst: list, idx: int, default=None):\n'
            '    """Return lst[idx] or default if index is out of range.\n'
            '    >>> safe_get([1, 2, 3], 10)\n'
            '    >>> safe_get([1, 2, 3], 1)\n'
            '    2\n'
            '    """'
        ),
        "body": (
            '    try:\n'
            '        return lst[idx]\n'
            '    except IndexError:\n'
            '        return default'
        ),
    },
    {
        "prompt": (
            'def clamp(value: float, lo: float, hi: float) -> float:\n'
            '    """Clamp value to the range [lo, hi].\n'
            '    >>> clamp(5, 0, 10)\n'
            '    5\n'
            '    >>> clamp(-3, 0, 10)\n'
            '    0\n'
            '    """'
        ),
        "body": '    return max(lo, min(hi, value))',
    },
    {
        "prompt": (
            'def safe_mean(lst: list) -> float:\n'
            '    """Return mean of lst, or 0.0 for empty list.\n'
            '    >>> safe_mean([1, 2, 3])\n'
            '    2.0\n'
            '    >>> safe_mean([])\n'
            '    0.0\n'
            '    """'
        ),
        "body": '    return sum(lst) / len(lst) if lst else 0.0',
    },
    {
        "prompt": (
            'def safe_max(lst: list, default=None):\n'
            '    """Return max of lst, or default if empty.\n'
            '    >>> safe_max([3, 1, 2])\n'
            '    3\n'
            '    >>> safe_max([], default=-1)\n'
            '    -1\n'
            '    """'
        ),
        "body": '    return max(lst) if lst else default',
    },
    {
        "prompt": (
            'def safe_min(lst: list, default=None):\n'
            '    """Return min of lst, or default if empty.\n'
            '    >>> safe_min([3, 1, 2])\n'
            '    1\n'
            '    >>> safe_min([], default=99)\n'
            '    99\n'
            '    """'
        ),
        "body": '    return min(lst) if lst else default',
    },
    {
        "prompt": (
            'def none_safe_add(a, b):\n'
            '    """Add a and b; treat None as 0.\n'
            '    >>> none_safe_add(3, None)\n'
            '    3\n'
            '    >>> none_safe_add(None, None)\n'
            '    0\n'
            '    """'
        ),
        "body": '    return (a or 0) + (b or 0)',
    },
    {
        "prompt": (
            'def non_empty_lines(text: str) -> list:\n'
            '    """Return list of non-empty lines (stripped) from text.\n'
            '    >>> non_empty_lines("a\\n\\nb\\n  \\nc")\n'
            '    [\'a\', \'b\', \'c\']\n'
            '    """'
        ),
        "body": '    return [line.strip() for line in text.splitlines() if line.strip()]',
    },
    {
        "prompt": (
            'def first_truthy(lst: list, default=None):\n'
            '    """Return first truthy element in lst, or default.\n'
            '    >>> first_truthy([0, None, "", 5, 6])\n'
            '    5\n'
            '    >>> first_truthy([0, False], default="x")\n'
            '    \'x\'\n'
            '    """'
        ),
        "body": '    return next((x for x in lst if x), default)',
    },
    {
        "prompt": (
            'def coalesce(*values, default=None):\n'
            '    """Return first non-None value, or default.\n'
            '    >>> coalesce(None, None, 3, 4)\n'
            '    3\n'
            '    """'
        ),
        "body": '    return next((v for v in values if v is not None), default)',
    },
    {
        "prompt": (
            'def divide_or_default(a: int, b: int, default: int = 0) -> float:\n'
            '    """Divide a/b; return default if b is zero.\n'
            '    >>> divide_or_default(10, 2)\n'
            '    5.0\n'
            '    >>> divide_or_default(10, 0)\n'
            '    0\n'
            '    """'
        ),
        "body": '    return a / b if b != 0 else default',
    },
    {
        "prompt": (
            'def bounded_index(lst: list, idx: int) -> int:\n'
            '    """Clamp idx to valid range and return lst[clamped_idx].\n'
            '    >>> bounded_index([10, 20, 30], 5)\n'
            '    30\n'
            '    >>> bounded_index([10, 20, 30], -1)\n'
            '    10\n'
            '    """'
        ),
        "body": (
            '    idx = max(0, min(len(lst) - 1, idx))\n'
            '    return lst[idx]'
        ),
    },
    {
        "prompt": (
            'def ensure_list(value) -> list:\n'
            '    """Wrap non-list value in a list; return list as-is.\n'
            '    >>> ensure_list(3)\n'
            '    [3]\n'
            '    >>> ensure_list([1, 2])\n'
            '    [1, 2]\n'
            '    """'
        ),
        "body": '    return value if isinstance(value, list) else [value]',
    },
    {
        "prompt": (
            'def strip_nones(lst: list) -> list:\n'
            '    """Remove None values from a list.\n'
            '    >>> strip_nones([1, None, 2, None, 3])\n'
            '    [1, 2, 3]\n'
            '    """'
        ),
        "body": '    return [x for x in lst if x is not None]',
    },
    {
        "prompt": (
            'def normalize_to_range(values: list, lo: float = 0.0, hi: float = 1.0) -> list:\n'
            '    """Normalize values to [lo, hi]. Return original if all equal.\n'
            '    >>> normalize_to_range([0, 5, 10])\n'
            '    [0.0, 0.5, 1.0]\n'
            '    """'
        ),
        "body": (
            '    mn, mx = min(values), max(values)\n'
            '    if mn == mx:\n'
            '        return list(values)\n'
            '    return [lo + (hi - lo) * (x - mn) / (mx - mn) for x in values]'
        ),
    },
    {
        "prompt": (
            'def safe_cast_int(s: str, default: int = 0) -> int:\n'
            '    """Parse s as int; return default on failure.\n'
            '    >>> safe_cast_int("42")\n'
            '    42\n'
            '    >>> safe_cast_int("abc")\n'
            '    0\n'
            '    """'
        ),
        "body": (
            '    try:\n'
            '        return int(s)\n'
            '    except (ValueError, TypeError):\n'
            '        return default'
        ),
    },
]

# ---------------------------------------------------------------------------
# 7. Graph / search (bonus hard category)
# ---------------------------------------------------------------------------
GRAPH_EXAMPLES = [
    {
        "prompt": (
            'def bfs(graph: dict, start) -> list:\n'
            '    """BFS traversal returning visited nodes in order.\n'
            '    >>> bfs({1: [2, 3], 2: [4], 3: [], 4: []}, 1)\n'
            '    [1, 2, 3, 4]\n'
            '    """'
        ),
        "body": (
            '    from collections import deque\n'
            '    visited = []\n'
            '    seen = set()\n'
            '    queue = deque([start])\n'
            '    seen.add(start)\n'
            '    while queue:\n'
            '        node = queue.popleft()\n'
            '        visited.append(node)\n'
            '        for neighbor in graph.get(node, []):\n'
            '            if neighbor not in seen:\n'
            '                seen.add(neighbor)\n'
            '                queue.append(neighbor)\n'
            '    return visited'
        ),
    },
    {
        "prompt": (
            'def dfs(graph: dict, start) -> list:\n'
            '    """DFS traversal (iterative) returning visited nodes in order.\n'
            '    >>> dfs({1: [2, 3], 2: [4], 3: [], 4: []}, 1)\n'
            '    [1, 3, 2, 4]\n'
            '    """'
        ),
        "body": (
            '    visited = []\n'
            '    seen = set()\n'
            '    stack = [start]\n'
            '    while stack:\n'
            '        node = stack.pop()\n'
            '        if node in seen:\n'
            '            continue\n'
            '        seen.add(node)\n'
            '        visited.append(node)\n'
            '        for neighbor in reversed(graph.get(node, [])):\n'
            '            if neighbor not in seen:\n'
            '                stack.append(neighbor)\n'
            '    return visited'
        ),
    },
    {
        "prompt": (
            'def has_cycle(graph: dict) -> bool:\n'
            '    """Return True if undirected graph (adjacency list) has a cycle.\n'
            '    >>> has_cycle({1: [2], 2: [1, 3], 3: [2]})\n'
            '    False\n'
            '    >>> has_cycle({1: [2], 2: [1, 3], 3: [2, 1]})\n'
            '    True\n'
            '    """'
        ),
        "body": (
            '    visited = set()\n'
            '    def dfs_check(node, parent):\n'
            '        visited.add(node)\n'
            '        for neighbor in graph.get(node, []):\n'
            '            if neighbor not in visited:\n'
            '                if dfs_check(neighbor, node):\n'
            '                    return True\n'
            '            elif neighbor != parent:\n'
            '                return True\n'
            '        return False\n'
            '    for node in graph:\n'
            '        if node not in visited:\n'
            '            if dfs_check(node, None):\n'
            '                return True\n'
            '    return False'
        ),
    },
    {
        "prompt": (
            'def topological_sort(graph: dict) -> list:\n'
            '    """Return topological order of a DAG (adjacency list).\n'
            '    >>> topological_sort({1: [3], 2: [3], 3: [4], 4: []})\n'
            '    [1, 2, 3, 4]\n'
            '    """'
        ),
        "body": (
            '    from collections import deque\n'
            '    in_degree = {node: 0 for node in graph}\n'
            '    for node in graph:\n'
            '        for neighbor in graph[node]:\n'
            '            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1\n'
            '    queue = deque(sorted(n for n, d in in_degree.items() if d == 0))\n'
            '    order = []\n'
            '    while queue:\n'
            '        node = queue.popleft()\n'
            '        order.append(node)\n'
            '        for neighbor in sorted(graph.get(node, [])):\n'
            '            in_degree[neighbor] -= 1\n'
            '            if in_degree[neighbor] == 0:\n'
            '                queue.append(neighbor)\n'
            '    return order if len(order) == len(in_degree) else []'
        ),
    },
    {
        "prompt": (
            'def dijkstra(graph: dict, start) -> dict:\n'
            '    """Return shortest distances from start to all nodes.\n'
            '    graph: {node: [(neighbor, weight), ...]}\n'
            '    >>> dijkstra({0: [(1,4),(2,1)], 1: [(3,1)], 2: [(1,2),(3,5)], 3: []}, 0)\n'
            '    {0: 0, 1: 3, 2: 1, 3: 4}\n'
            '    """'
        ),
        "body": (
            '    import heapq\n'
            '    dist = {start: 0}\n'
            '    heap = [(0, start)]\n'
            '    while heap:\n'
            '        d, u = heapq.heappop(heap)\n'
            '        if d > dist.get(u, float("inf")):\n'
            '            continue\n'
            '        for v, w in graph.get(u, []):\n'
            '            nd = d + w\n'
            '            if nd < dist.get(v, float("inf")):\n'
            '                dist[v] = nd\n'
            '                heapq.heappush(heap, (nd, v))\n'
            '    return dist'
        ),
    },
    {
        "prompt": (
            'def connected_components(graph: dict) -> list:\n'
            '    """Return list of connected components (each as sorted list).\n'
            '    >>> sorted(connected_components({1:[2],2:[1],3:[4],4:[3],5:[]}))\n'
            '    [[1, 2], [3, 4], [5]]\n'
            '    """'
        ),
        "body": (
            '    visited = set()\n'
            '    components = []\n'
            '    def dfs(node, component):\n'
            '        visited.add(node)\n'
            '        component.append(node)\n'
            '        for neighbor in graph.get(node, []):\n'
            '            if neighbor not in visited:\n'
            '                dfs(neighbor, component)\n'
            '    for node in graph:\n'
            '        if node not in visited:\n'
            '            comp = []\n'
            '            dfs(node, comp)\n'
            '            components.append(sorted(comp))\n'
            '    return components'
        ),
    },
    {
        "prompt": (
            'def word_ladder_length(begin: str, end: str, word_list: list) -> int:\n'
            '    """Return min transformations from begin to end (one letter at a time).\n'
            '    Returns 0 if impossible.\n'
            '    >>> word_ladder_length("hit", "cog", ["hot","dot","dog","lot","log","cog"])\n'
            '    5\n'
            '    """'
        ),
        "body": (
            '    from collections import deque\n'
            '    word_set = set(word_list)\n'
            '    if end not in word_set:\n'
            '        return 0\n'
            '    queue = deque([(begin, 1)])\n'
            '    visited = {begin}\n'
            '    while queue:\n'
            '        word, length = queue.popleft()\n'
            '        for i in range(len(word)):\n'
            '            for c in "abcdefghijklmnopqrstuvwxyz":\n'
            '                new_word = word[:i] + c + word[i+1:]\n'
            '                if new_word == end:\n'
            '                    return length + 1\n'
            '                if new_word in word_set and new_word not in visited:\n'
            '                    visited.add(new_word)\n'
            '                    queue.append((new_word, length + 1))\n'
            '    return 0'
        ),
    },
]

# ---------------------------------------------------------------------------
# 8. Functional / higher-order
# ---------------------------------------------------------------------------
FUNCTIONAL_EXAMPLES = [
    {
        "prompt": (
            'def compose(*fns):\n'
            '    """Return a function that applies functions right-to-left.\n'
            '    >>> f = compose(lambda x: x*2, lambda x: x+1)\n'
            '    >>> f(3)\n'
            '    8\n'
            '    """'
        ),
        "body": (
            '    from functools import reduce\n'
            '    return reduce(lambda f, g: lambda x: f(g(x)), fns)'
        ),
    },
    {
        "prompt": (
            'def memoize(fn):\n'
            '    """Return a memoized version of fn using a dict cache.\n'
            '    >>> fib = memoize(lambda n: n if n < 2 else fib(n-1) + fib(n-2))\n'
            '    >>> fib(10)\n'
            '    55\n'
            '    """'
        ),
        "body": (
            '    cache = {}\n'
            '    def wrapper(*args):\n'
            '        if args not in cache:\n'
            '            cache[args] = fn(*args)\n'
            '        return cache[args]\n'
            '    return wrapper'
        ),
    },
    {
        "prompt": (
            'def curry(fn):\n'
            '    """Curry a two-argument function.\n'
            '    >>> add = curry(lambda a, b: a + b)\n'
            '    >>> add(3)(4)\n'
            '    7\n'
            '    """'
        ),
        "body": '    return lambda a: lambda b: fn(a, b)',
    },
    {
        "prompt": (
            'def pipe(value, *fns):\n'
            '    """Apply functions left-to-right to value.\n'
            '    >>> pipe(1, lambda x: x+1, lambda x: x*2, lambda x: x-3)\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    for fn in fns:\n'
            '        value = fn(value)\n'
            '    return value'
        ),
    },
    {
        "prompt": (
            'def retry(fn, times: int, exceptions=(Exception,)):\n'
            '    """Retry fn up to times times on given exceptions; return last result.\n'
            '    >>> attempts = [0]\n'
            '    >>> def flaky():\n'
            '    ...     attempts[0] += 1\n'
            '    ...     if attempts[0] < 3: raise ValueError("retry")\n'
            '    ...     return 42\n'
            '    >>> retry(flaky, 5)\n'
            '    42\n'
            '    """'
        ),
        "body": (
            '    last_exc = None\n'
            '    for _ in range(times):\n'
            '        try:\n'
            '            return fn()\n'
            '        except exceptions as e:\n'
            '            last_exc = e\n'
            '    raise last_exc'
        ),
    },
    {
        "prompt": (
            'def batch(lst: list, size: int):\n'
            '    """Yield successive batches of given size from lst.\n'
            '    >>> list(batch([1,2,3,4,5], 2))\n'
            '    [[1, 2], [3, 4], [5]]\n'
            '    """'
        ),
        "body": (
            '    for i in range(0, len(lst), size):\n'
            '        yield lst[i:i + size]'
        ),
    },
    {
        "prompt": (
            'def flatten_map(fn, lst: list) -> list:\n'
            '    """Apply fn to each element and flatten one level.\n'
            '    >>> flatten_map(lambda x: [x, x*2], [1, 2, 3])\n'
            '    [1, 2, 2, 4, 3, 6]\n'
            '    """'
        ),
        "body": (
            '    from itertools import chain\n'
            '    return list(chain.from_iterable(fn(x) for x in lst))'
        ),
    },
    {
        "prompt": (
            'def take_while(pred, lst: list) -> list:\n'
            '    """Return elements from start of lst while pred is True.\n'
            '    >>> take_while(lambda x: x < 5, [1, 3, 5, 2, 7])\n'
            '    [1, 3]\n'
            '    """'
        ),
        "body": (
            '    from itertools import takewhile\n'
            '    return list(takewhile(pred, lst))'
        ),
    },
    {
        "prompt": (
            'def drop_while(pred, lst: list) -> list:\n'
            '    """Drop elements from start while pred is True, return the rest.\n'
            '    >>> drop_while(lambda x: x < 5, [1, 3, 5, 2, 7])\n'
            '    [5, 2, 7]\n'
            '    """'
        ),
        "body": (
            '    from itertools import dropwhile\n'
            '    return list(dropwhile(pred, lst))'
        ),
    },
]

# ---------------------------------------------------------------------------
# 9. Additional math / number theory (extended)
# ---------------------------------------------------------------------------
MATH_EXTRA = [
    {
        "prompt": (
            'def is_abundant(n: int) -> bool:\n'
            '    """Return True if n is an abundant number (sum of proper divisors > n).\n'
            '    >>> is_abundant(12)\n'
            '    True\n'
            '    >>> is_abundant(15)\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    s = 1\n'
            '    for i in range(2, int(n**0.5) + 1):\n'
            '        if n % i == 0:\n'
            '            s += i\n'
            '            if i != n // i:\n'
            '                s += n // i\n'
            '    return s > n'
        ),
    },
    {
        "prompt": (
            'def is_deficient(n: int) -> bool:\n'
            '    """Return True if n is a deficient number (sum of proper divisors < n).\n'
            '    >>> is_deficient(15)\n'
            '    True\n'
            '    >>> is_deficient(12)\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    s = 1\n'
            '    for i in range(2, int(n**0.5) + 1):\n'
            '        if n % i == 0:\n'
            '            s += i\n'
            '            if i != n // i:\n'
            '                s += n // i\n'
            '    return s < n'
        ),
    },
    {
        "prompt": (
            'def sum_proper_divisors(n: int) -> int:\n'
            '    """Return the sum of proper divisors of n.\n'
            '    >>> sum_proper_divisors(12)\n'
            '    16\n'
            '    >>> sum_proper_divisors(1)\n'
            '    0\n'
            '    """'
        ),
        "body": (
            '    if n <= 1:\n'
            '        return 0\n'
            '    s = 1\n'
            '    for i in range(2, int(n**0.5) + 1):\n'
            '        if n % i == 0:\n'
            '            s += i\n'
            '            if i != n // i:\n'
            '                s += n // i\n'
            '    return s'
        ),
    },
    {
        "prompt": (
            'def next_prime(n: int) -> int:\n'
            '    """Return the smallest prime greater than n.\n'
            '    >>> next_prime(10)\n'
            '    11\n'
            '    >>> next_prime(13)\n'
            '    17\n'
            '    """'
        ),
        "body": (
            '    def is_prime(x):\n'
            '        if x < 2: return False\n'
            '        for i in range(2, int(x**0.5)+1):\n'
            '            if x % i == 0: return False\n'
            '        return True\n'
            '    candidate = n + 1\n'
            '    while not is_prime(candidate):\n'
            '        candidate += 1\n'
            '    return candidate'
        ),
    },
    {
        "prompt": (
            'def goldbach(n: int) -> tuple:\n'
            '    """Return two primes that sum to n (even n >= 4) by Goldbach\'s conjecture.\n'
            '    >>> sum(goldbach(28)) == 28\n'
            '    True\n'
            '    """'
        ),
        "body": (
            '    def is_prime(x):\n'
            '        if x < 2: return False\n'
            '        return all(x % i != 0 for i in range(2, int(x**0.5)+1))\n'
            '    for a in range(2, n // 2 + 1):\n'
            '        b = n - a\n'
            '        if is_prime(a) and is_prime(b):\n'
            '            return (a, b)\n'
            '    return None'
        ),
    },
    {
        "prompt": (
            'def modular_inverse(a: int, m: int) -> int:\n'
            '    """Return modular inverse of a mod m using Fermat\'s little theorem (m prime).\n'
            '    >>> modular_inverse(3, 11)\n'
            '    4\n'
            '    """'
        ),
        "body": '    return pow(a, m - 2, m)',
    },
    {
        "prompt": (
            'def chinese_remainder(remainders: list, moduli: list) -> int:\n'
            '    """Solve CRT: find x s.t. x ≡ r_i (mod m_i) for pairwise coprime m_i.\n'
            '    >>> chinese_remainder([2, 3, 2], [3, 5, 7])\n'
            '    23\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    M = 1\n'
            '    for m in moduli:\n'
            '        M *= m\n'
            '    x = 0\n'
            '    for r, m in zip(remainders, moduli):\n'
            '        Mi = M // m\n'
            '        x += r * Mi * pow(Mi, -1, m)\n'
            '    return x % M'
        ),
    },
    {
        "prompt": (
            'def sum_digits_factorial(n: int) -> int:\n'
            '    """Return the sum of digits of n!.\n'
            '    >>> sum_digits_factorial(10)\n'
            '    27\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    return sum(int(d) for d in str(math.factorial(n)))'
        ),
    },
    {
        "prompt": (
            'def count_trailing_zeros_factorial(n: int) -> int:\n'
            '    """Count trailing zeros in n!.\n'
            '    >>> count_trailing_zeros_factorial(25)\n'
            '    6\n'
            '    """'
        ),
        "body": (
            '    count = 0\n'
            '    power = 5\n'
            '    while power <= n:\n'
            '        count += n // power\n'
            '        power *= 5\n'
            '    return count'
        ),
    },
    {
        "prompt": (
            'def nearest_power_of_two(n: int) -> int:\n'
            '    """Return the nearest power of 2 to n (round down if tie).\n'
            '    >>> nearest_power_of_two(7)\n'
            '    8\n'
            '    >>> nearest_power_of_two(12)\n'
            '    8\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    lower = 2 ** int(math.log2(n))\n'
            '    upper = lower * 2\n'
            '    return lower if (n - lower) <= (upper - n) else upper'
        ),
    },
    {
        "prompt": (
            'def multiply_strings(num1: str, num2: str) -> str:\n'
            '    """Multiply two non-negative integers represented as strings.\n'
            '    >>> multiply_strings("123", "456")\n'
            '    \'56088\'\n'
            '    """'
        ),
        "body": '    return str(int(num1) * int(num2))',
    },
    {
        "prompt": (
            'def base_convert(n: int, base: int) -> str:\n'
            '    """Convert n to a string in given base (2-16).\n'
            '    >>> base_convert(255, 16)\n'
            '    \'ff\'\n'
            '    >>> base_convert(10, 2)\n'
            '    \'1010\'\n'
            '    """'
        ),
        "body": (
            '    digits = "0123456789abcdef"\n'
            '    if n == 0:\n'
            '        return "0"\n'
            '    result = ""\n'
            '    while n:\n'
            '        result = digits[n % base] + result\n'
            '        n //= base\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def floor_div(a: int, b: int) -> int:\n'
            '    """Floor division that matches mathematical floor for negative numbers.\n'
            '    >>> floor_div(-7, 2)\n'
            '    -4\n'
            '    >>> floor_div(7, 2)\n'
            '    3\n'
            '    """'
        ),
        "body": '    return a // b',
    },
    {
        "prompt": (
            'def harmonic_sum(n: int) -> float:\n'
            '    """Return the n-th harmonic number H_n = 1 + 1/2 + ... + 1/n.\n'
            '    >>> round(harmonic_sum(4), 4)\n'
            '    2.0833\n'
            '    """'
        ),
        "body": '    return sum(1.0 / i for i in range(1, n + 1))',
    },
    {
        "prompt": (
            'def is_square(n: int) -> bool:\n'
            '    """Return True if n is a perfect square.\n'
            '    >>> is_square(25)\n'
            '    True\n'
            '    >>> is_square(26)\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    if n < 0:\n'
            '        return False\n'
            '    r = math.isqrt(n)\n'
            '    return r * r == n'
        ),
    },
]

# ---------------------------------------------------------------------------
# 10. Additional string examples
# ---------------------------------------------------------------------------
STRING_EXTRA = [
    {
        "prompt": (
            'def count_sentences(text: str) -> int:\n'
            '    """Count the number of sentences ending with . ! or ?.\n'
            '    >>> count_sentences("Hello! How are you? Fine.")\n'
            '    3\n'
            '    """'
        ),
        "body": (
            '    import re\n'
            '    return len(re.findall(r\'[.!?]+\', text))'
        ),
    },
    {
        "prompt": (
            'def strip_accents(s: str) -> str:\n'
            '    """Remove accent marks from unicode characters.\n'
            '    >>> strip_accents("café")\n'
            '    \'cafe\'\n'
            '    """'
        ),
        "body": (
            '    import unicodedata\n'
            '    return "".join(\n'
            '        c for c in unicodedata.normalize("NFD", s)\n'
            '        if unicodedata.category(c) != "Mn"\n'
            '    )'
        ),
    },
    {
        "prompt": (
            'def find_first_non_repeating(s: str) -> str:\n'
            '    """Return first character that does not repeat; empty string if none.\n'
            '    >>> find_first_non_repeating("aabbcd")\n'
            '    \'c\'\n'
            '    """'
        ),
        "body": (
            '    from collections import Counter\n'
            '    counts = Counter(s)\n'
            '    for ch in s:\n'
            '        if counts[ch] == 1:\n'
            '            return ch\n'
            '    return ""'
        ),
    },
    {
        "prompt": (
            'def longest_common_prefix(words: list) -> str:\n'
            '    """Return the longest common prefix of a list of strings.\n'
            '    >>> longest_common_prefix(["flower", "flow", "flight"])\n'
            '    \'fl\'\n'
            '    """'
        ),
        "body": (
            '    if not words:\n'
            '        return ""\n'
            '    prefix = words[0]\n'
            '    for word in words[1:]:\n'
            '        while not word.startswith(prefix):\n'
            '            prefix = prefix[:-1]\n'
            '            if not prefix:\n'
            '                return ""\n'
            '    return prefix'
        ),
    },
    {
        "prompt": (
            'def count_capital_letters(s: str) -> int:\n'
            '    """Return the count of uppercase letters in s.\n'
            '    >>> count_capital_letters("Hello World")\n'
            '    2\n'
            '    """'
        ),
        "body": '    return sum(1 for c in s if c.isupper())',
    },
    {
        "prompt": (
            'def replace_nth(s: str, old: str, new: str, n: int) -> str:\n'
            '    """Replace only the n-th occurrence (1-indexed) of old with new in s.\n'
            '    >>> replace_nth("ababab", "a", "X", 2)\n'
            '    \'abXbab\'\n'
            '    """'
        ),
        "body": (
            '    parts = s.split(old)\n'
            '    if n > len(parts) - 1:\n'
            '        return s\n'
            '    return old.join(parts[:n]) + new + old.join(parts[n:])'
        ),
    },
    {
        "prompt": (
            'def abbreviate(s: str) -> str:\n'
            '    """Return initials of each word in s, uppercased.\n'
            '    >>> abbreviate("New York City")\n'
            '    \'NYC\'\n'
            '    """'
        ),
        "body": '    return "".join(w[0].upper() for w in s.split())',
    },
    {
        "prompt": (
            'def is_pangram(s: str) -> bool:\n'
            '    """Return True if s contains every letter of the alphabet.\n'
            '    >>> is_pangram("The quick brown fox jumps over the lazy dog")\n'
            '    True\n'
            '    """'
        ),
        "body": '    return set("abcdefghijklmnopqrstuvwxyz").issubset(set(s.lower()))',
    },
    {
        "prompt": (
            'def caesar_cipher(text: str, shift: int) -> str:\n'
            '    """Apply Caesar cipher with given shift to alphabetic chars.\n'
            '    >>> caesar_cipher("abc", 3)\n'
            '    \'def\'\n'
            '    >>> caesar_cipher("xyz", 3)\n'
            '    \'abc\'\n'
            '    """'
        ),
        "body": (
            '    result = []\n'
            '    for ch in text:\n'
            '        if ch.isalpha():\n'
            '            base = ord("A") if ch.isupper() else ord("a")\n'
            '            result.append(chr((ord(ch) - base + shift) % 26 + base))\n'
            '        else:\n'
            '            result.append(ch)\n'
            '    return "".join(result)'
        ),
    },
    {
        "prompt": (
            'def word_wrap(text: str, width: int) -> str:\n'
            '    """Return text with newlines inserted at word boundaries within width.\n'
            '    >>> word_wrap("hello world foo bar", 11)\n'
            '    \'hello world\\\\nfoo bar\'\n'
            '    """'
        ),
        "body": (
            '    import textwrap\n'
            '    return "\\n".join(textwrap.wrap(text, width))'
        ),
    },
    {
        "prompt": (
            'def remove_stop_words(text: str, stop_words: list) -> str:\n'
            '    """Remove stop words from text, preserving remaining word order.\n'
            '    >>> remove_stop_words("the cat sat on the mat", ["the", "on"])\n'
            '    \'cat sat mat\'\n'
            '    """'
        ),
        "body": (
            '    stop = set(stop_words)\n'
            '    return " ".join(w for w in text.split() if w not in stop)'
        ),
    },
    {
        "prompt": (
            'def char_frequency(s: str) -> dict:\n'
            '    """Return dict of character frequencies in s.\n'
            '    >>> char_frequency("abca")\n'
            '    {\'a\': 2, \'b\': 1, \'c\': 1}\n'
            '    """'
        ),
        "body": (
            '    from collections import Counter\n'
            '    return dict(Counter(s))'
        ),
    },
    {
        "prompt": (
            'def longest_palindrome_substr(s: str) -> str:\n'
            '    """Return the longest palindromic substring of s.\n'
            '    >>> longest_palindrome_substr("babad")\n'
            '    \'bab\'\n'
            '    """'
        ),
        "body": (
            '    if not s:\n'
            '        return ""\n'
            '    start, end = 0, 0\n'
            '    def expand(l, r):\n'
            '        while l >= 0 and r < len(s) and s[l] == s[r]:\n'
            '            l -= 1; r += 1\n'
            '        return l + 1, r - 1\n'
            '    for i in range(len(s)):\n'
            '        l1, r1 = expand(i, i)\n'
            '        l2, r2 = expand(i, i + 1)\n'
            '        if r1 - l1 > end - start:\n'
            '            start, end = l1, r1\n'
            '        if r2 - l2 > end - start:\n'
            '            start, end = l2, r2\n'
            '    return s[start:end+1]'
        ),
    },
]

# ---------------------------------------------------------------------------
# 11. Additional list / algorithm examples
# ---------------------------------------------------------------------------
LIST_EXTRA = [
    {
        "prompt": (
            'def count_pairs_sum(lst: list, target: int) -> int:\n'
            '    """Count pairs of indices (i < j) where lst[i] + lst[j] == target.\n'
            '    >>> count_pairs_sum([1, 5, 3, 3, 5], 6)\n'
            '    4\n'
            '    """'
        ),
        "body": (
            '    from collections import Counter\n'
            '    count = 0\n'
            '    seen = Counter()\n'
            '    for x in lst:\n'
            '        count += seen[target - x]\n'
            '        seen[x] += 1\n'
            '    return count'
        ),
    },
    {
        "prompt": (
            'def matrix_multiply(A: list, B: list) -> list:\n'
            '    """Multiply two matrices A (m×k) and B (k×n).\n'
            '    >>> matrix_multiply([[1,2],[3,4]], [[5,6],[7,8]])\n'
            '    [[19, 22], [43, 50]]\n'
            '    """'
        ),
        "body": (
            '    m = len(A)\n'
            '    k = len(B)\n'
            '    n = len(B[0])\n'
            '    C = [[0]*n for _ in range(m)]\n'
            '    for i in range(m):\n'
            '        for j in range(n):\n'
            '            for l in range(k):\n'
            '                C[i][j] += A[i][l] * B[l][j]\n'
            '    return C'
        ),
    },
    {
        "prompt": (
            'def spiral_order(matrix: list) -> list:\n'
            '    """Return elements of matrix in spiral order.\n'
            '    >>> spiral_order([[1,2,3],[4,5,6],[7,8,9]])\n'
            '    [1, 2, 3, 6, 9, 8, 7, 4, 5]\n'
            '    """'
        ),
        "body": (
            '    result = []\n'
            '    while matrix:\n'
            '        result += matrix.pop(0)\n'
            '        matrix = list(zip(*matrix))[::-1]\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def bucket_sort(lst: list, num_buckets: int = 10) -> list:\n'
            '    """Sort list of floats in [0, 1) using bucket sort.\n'
            '    >>> bucket_sort([0.9, 0.1, 0.5, 0.3, 0.7])\n'
            '    [0.1, 0.3, 0.5, 0.7, 0.9]\n'
            '    """'
        ),
        "body": (
            '    buckets = [[] for _ in range(num_buckets)]\n'
            '    for x in lst:\n'
            '        idx = int(x * num_buckets)\n'
            '        if idx == num_buckets:\n'
            '            idx -= 1\n'
            '        buckets[idx].append(x)\n'
            '    return [x for bucket in buckets for x in sorted(bucket)]'
        ),
    },
    {
        "prompt": (
            'def count_greater(lst: list, threshold) -> int:\n'
            '    """Count elements strictly greater than threshold.\n'
            '    >>> count_greater([1, 5, 3, 7, 2], 4)\n'
            '    2\n'
            '    """'
        ),
        "body": '    return sum(1 for x in lst if x > threshold)',
    },
    {
        "prompt": (
            'def zip_dict(keys: list, values: list) -> dict:\n'
            '    """Zip keys and values into a dict.\n'
            '    >>> zip_dict(["a", "b", "c"], [1, 2, 3])\n'
            '    {\'a\': 1, \'b\': 2, \'c\': 3}\n'
            '    """'
        ),
        "body": '    return dict(zip(keys, values))',
    },
    {
        "prompt": (
            'def alternating_sum(lst: list) -> int:\n'
            '    """Return alternating sum: lst[0] - lst[1] + lst[2] - ...\n'
            '    >>> alternating_sum([1, 2, 3, 4, 5])\n'
            '    3\n'
            '    """'
        ),
        "body": '    return sum(v * (-1)**i for i, v in enumerate(lst))',
    },
    {
        "prompt": (
            'def running_max(lst: list) -> list:\n'
            '    """Return running maximum list.\n'
            '    >>> running_max([3, 1, 4, 1, 5, 9, 2, 6])\n'
            '    [3, 3, 4, 4, 5, 9, 9, 9]\n'
            '    """'
        ),
        "body": (
            '    result = []\n'
            '    current_max = float("-inf")\n'
            '    for x in lst:\n'
            '        current_max = max(current_max, x)\n'
            '        result.append(current_max)\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def longest_run(lst: list) -> int:\n'
            '    """Return the length of the longest run of equal consecutive elements.\n'
            '    >>> longest_run([1, 1, 2, 2, 2, 3])\n'
            '    3\n'
            '    """'
        ),
        "body": (
            '    if not lst:\n'
            '        return 0\n'
            '    max_run = current_run = 1\n'
            '    for i in range(1, len(lst)):\n'
            '        if lst[i] == lst[i-1]:\n'
            '            current_run += 1\n'
            '            max_run = max(max_run, current_run)\n'
            '        else:\n'
            '            current_run = 1\n'
            '    return max_run'
        ),
    },
    {
        "prompt": (
            'def rotate_matrix_90(matrix: list) -> list:\n'
            '    """Rotate a square matrix 90 degrees clockwise.\n'
            '    >>> rotate_matrix_90([[1,2],[3,4]])\n'
            '    [[3, 1], [4, 2]]\n'
            '    """'
        ),
        "body": (
            '    return [list(row) for row in zip(*matrix[::-1])]'
        ),
    },
    {
        "prompt": (
            'def find_duplicates(lst: list) -> list:\n'
            '    """Return sorted list of elements that appear more than once.\n'
            '    >>> find_duplicates([1, 2, 2, 3, 3, 4])\n'
            '    [2, 3]\n'
            '    """'
        ),
        "body": (
            '    from collections import Counter\n'
            '    return sorted(x for x, c in Counter(lst).items() if c > 1)'
        ),
    },
    {
        "prompt": (
            'def sum_matrix(matrix: list) -> int:\n'
            '    """Return the sum of all elements in a 2D list.\n'
            '    >>> sum_matrix([[1, 2], [3, 4]])\n'
            '    10\n'
            '    """'
        ),
        "body": '    return sum(x for row in matrix for x in row)',
    },
    {
        "prompt": (
            'def diagonal_sum(matrix: list) -> int:\n'
            '    """Return the sum of the main diagonal of a square matrix.\n'
            '    >>> diagonal_sum([[1,2,3],[4,5,6],[7,8,9]])\n'
            '    15\n'
            '    """'
        ),
        "body": '    return sum(matrix[i][i] for i in range(len(matrix)))',
    },
]

# ---------------------------------------------------------------------------
# 12. Additional recursive / DP examples
# ---------------------------------------------------------------------------
RECURSIVE_EXTRA = [
    {
        "prompt": (
            'def word_break(s: str, word_dict: list) -> bool:\n'
            '    """Return True if s can be segmented into words from word_dict.\n'
            '    >>> word_break("leetcode", ["leet", "code"])\n'
            '    True\n'
            '    >>> word_break("catsandog", ["cats","dog","sand","and","cat"])\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    word_set = set(word_dict)\n'
            '    n = len(s)\n'
            '    dp = [False] * (n + 1)\n'
            '    dp[0] = True\n'
            '    for i in range(1, n + 1):\n'
            '        for j in range(i):\n'
            '            if dp[j] and s[j:i] in word_set:\n'
            '                dp[i] = True\n'
            '                break\n'
            '    return dp[n]'
        ),
    },
    {
        "prompt": (
            'def max_product_subarray(nums: list) -> int:\n'
            '    """Return the maximum product of a contiguous subarray.\n'
            '    >>> max_product_subarray([2, 3, -2, 4])\n'
            '    6\n'
            '    >>> max_product_subarray([-2, 0, -1])\n'
            '    0\n'
            '    """'
        ),
        "body": (
            '    max_p = min_p = result = nums[0]\n'
            '    for n in nums[1:]:\n'
            '        candidates = (n, max_p * n, min_p * n)\n'
            '        max_p = max(candidates)\n'
            '        min_p = min(candidates)\n'
            '        result = max(result, max_p)\n'
            '    return result'
        ),
    },
    {
        "prompt": (
            'def subset_sum_exists(nums: list, target: int) -> bool:\n'
            '    """Return True if any subset of nums sums to target.\n'
            '    >>> subset_sum_exists([3, 34, 4, 12, 5, 2], 9)\n'
            '    True\n'
            '    >>> subset_sum_exists([3, 34, 4, 12, 5, 2], 30)\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    dp = {0}\n'
            '    for n in nums:\n'
            '        dp = {x + n for x in dp} | dp\n'
            '    return target in dp'
        ),
    },
    {
        "prompt": (
            'def count_paths(m: int, n: int) -> int:\n'
            '    """Count unique paths in an m×n grid from top-left to bottom-right.\n'
            '    >>> count_paths(3, 3)\n'
            '    6\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    return math.comb(m + n - 2, m - 1)'
        ),
    },
    {
        "prompt": (
            'def house_robber(nums: list) -> int:\n'
            '    """Max amount you can rob from non-adjacent houses.\n'
            '    >>> house_robber([2, 7, 9, 3, 1])\n'
            '    12\n'
            '    """'
        ),
        "body": (
            '    prev2 = prev1 = 0\n'
            '    for n in nums:\n'
            '        prev2, prev1 = prev1, max(prev1, prev2 + n)\n'
            '    return prev1'
        ),
    },
    {
        "prompt": (
            'def minimum_path_sum(grid: list) -> int:\n'
            '    """Return minimum path sum from top-left to bottom-right of grid.\n'
            '    >>> minimum_path_sum([[1,3,1],[1,5,1],[4,2,1]])\n'
            '    7\n'
            '    """'
        ),
        "body": (
            '    m, n = len(grid), len(grid[0])\n'
            '    dp = [row[:] for row in grid]\n'
            '    for i in range(1, m):\n'
            '        dp[i][0] += dp[i-1][0]\n'
            '    for j in range(1, n):\n'
            '        dp[0][j] += dp[0][j-1]\n'
            '    for i in range(1, m):\n'
            '        for j in range(1, n):\n'
            '            dp[i][j] += min(dp[i-1][j], dp[i][j-1])\n'
            '    return dp[m-1][n-1]'
        ),
    },
    {
        "prompt": (
            'def jump_game(nums: list) -> bool:\n'
            '    """Return True if you can reach the last index from index 0.\n'
            '    >>> jump_game([2, 3, 1, 1, 4])\n'
            '    True\n'
            '    >>> jump_game([3, 2, 1, 0, 4])\n'
            '    False\n'
            '    """'
        ),
        "body": (
            '    reach = 0\n'
            '    for i, v in enumerate(nums):\n'
            '        if i > reach:\n'
            '            return False\n'
            '        reach = max(reach, i + v)\n'
            '    return True'
        ),
    },
    {
        "prompt": (
            'def decode_ways(s: str) -> int:\n'
            '    """Count the number of ways to decode a digit string to letters (A=1..Z=26).\n'
            '    >>> decode_ways("226")\n'
            '    3\n'
            '    >>> decode_ways("06")\n'
            '    0\n'
            '    """'
        ),
        "body": (
            '    if not s or s[0] == "0":\n'
            '        return 0\n'
            '    n = len(s)\n'
            '    dp = [0] * (n + 1)\n'
            '    dp[0] = dp[1] = 1\n'
            '    for i in range(2, n + 1):\n'
            '        if s[i-1] != "0":\n'
            '            dp[i] += dp[i-1]\n'
            '        two_digit = int(s[i-2:i])\n'
            '        if 10 <= two_digit <= 26:\n'
            '            dp[i] += dp[i-2]\n'
            '    return dp[n]'
        ),
    },
]

# ---------------------------------------------------------------------------
# 13. Extra / miscellaneous (fill-in to reach 200+ unique)
# ---------------------------------------------------------------------------
MISC_EXTRA = [
    {
        "prompt": (
            'def int_to_bin_str(n: int) -> str:\n'
            '    """Return binary string representation of non-negative integer.\n'
            '    >>> int_to_bin_str(10)\n'
            '    \'1010\'\n'
            '    >>> int_to_bin_str(0)\n'
            '    \'0\'\n'
            '    """'
        ),
        "body": '    return bin(n)[2:] if n else "0"',
    },
    {
        "prompt": (
            'def reverse_number(n: int) -> int:\n'
            '    """Reverse the digits of a non-negative integer.\n'
            '    >>> reverse_number(12345)\n'
            '    54321\n'
            '    >>> reverse_number(100)\n'
            '    1\n'
            '    """'
        ),
        "body": '    return int(str(n)[::-1])',
    },
    {
        "prompt": (
            'def is_leap_year(year: int) -> bool:\n'
            '    """Return True if year is a leap year.\n'
            '    >>> is_leap_year(2000)\n'
            '    True\n'
            '    >>> is_leap_year(1900)\n'
            '    False\n'
            '    >>> is_leap_year(2024)\n'
            '    True\n'
            '    """'
        ),
        "body": '    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)',
    },
    {
        "prompt": (
            'def sign(n: float) -> int:\n'
            '    """Return -1, 0, or 1 representing the sign of n.\n'
            '    >>> sign(-5)\n'
            '    -1\n'
            '    >>> sign(0)\n'
            '    0\n'
            '    >>> sign(3)\n'
            '    1\n'
            '    """'
        ),
        "body": (
            '    if n < 0: return -1\n'
            '    if n > 0: return 1\n'
            '    return 0'
        ),
    },
    {
        "prompt": (
            'def zip_lists(*lists) -> list:\n'
            '    """Zip multiple lists, truncating to the shortest.\n'
            '    >>> zip_lists([1,2,3], ["a","b","c"], [True, False, True])\n'
            '    [(1, \'a\', True), (2, \'b\', False), (3, \'c\', True)]\n'
            '    """'
        ),
        "body": '    return list(zip(*lists))',
    },
    {
        "prompt": (
            'def count_less_than(lst: list, threshold) -> int:\n'
            '    """Count elements strictly less than threshold.\n'
            '    >>> count_less_than([1, 5, 3, 7, 2], 4)\n'
            '    3\n'
            '    """'
        ),
        "body": '    return sum(1 for x in lst if x < threshold)',
    },
    {
        "prompt": (
            'def geometric_mean(lst: list) -> float:\n'
            '    """Return the geometric mean of a list of positive numbers.\n'
            '    >>> round(geometric_mean([4, 9]), 4)\n'
            '    6.0\n'
            '    """'
        ),
        "body": (
            '    import math\n'
            '    product = 1.0\n'
            '    for x in lst:\n'
            '        product *= x\n'
            '    return product ** (1.0 / len(lst))'
        ),
    },
    {
        "prompt": (
            'def repeat_list(lst: list, n: int) -> list:\n'
            '    """Return lst repeated n times.\n'
            '    >>> repeat_list([1, 2], 3)\n'
            '    [1, 2, 1, 2, 1, 2]\n'
            '    """'
        ),
        "body": '    return lst * n',
    },
    {
        "prompt": (
            'def unique_sorted(lst: list) -> list:\n'
            '    """Return sorted list of unique elements.\n'
            '    >>> unique_sorted([3, 1, 4, 1, 5, 9, 2, 6, 5])\n'
            '    [1, 2, 3, 4, 5, 6, 9]\n'
            '    """'
        ),
        "body": '    return sorted(set(lst))',
    },
    {
        "prompt": (
            'def is_sorted(lst: list) -> bool:\n'
            '    """Return True if lst is non-decreasingly sorted.\n'
            '    >>> is_sorted([1, 2, 3, 3, 5])\n'
            '    True\n'
            '    >>> is_sorted([1, 3, 2])\n'
            '    False\n'
            '    """'
        ),
        "body": '    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))',
    },
    {
        "prompt": (
            'def range_sum(lst: list, lo: int, hi: int) -> int:\n'
            '    """Sum elements in lst that are between lo and hi (inclusive).\n'
            '    >>> range_sum([1, 5, 3, 7, 9, 2], 3, 7)\n'
            '    15\n'
            '    """'
        ),
        "body": '    return sum(x for x in lst if lo <= x <= hi)',
    },
    {
        "prompt": (
            'def every_other(lst: list) -> list:\n'
            '    """Return every other element starting from index 0.\n'
            '    >>> every_other([1, 2, 3, 4, 5, 6])\n'
            '    [1, 3, 5]\n'
            '    """'
        ),
        "body": '    return lst[::2]',
    },
]

# ---------------------------------------------------------------------------
# Collect and write
# ---------------------------------------------------------------------------

ALL_EXAMPLES = (
    MATH_EXAMPLES
    + MATH_EXTRA
    + STRING_EXAMPLES
    + STRING_EXTRA
    + LIST_EXAMPLES
    + LIST_EXTRA
    + RECURSIVE_EXAMPLES
    + RECURSIVE_EXTRA
    + DICT_SET_EXAMPLES
    + EDGE_CASE_EXAMPLES
    + GRAPH_EXAMPLES
    + FUNCTIONAL_EXAMPLES
    + MISC_EXTRA
)

REPETITIONS = 5


def main():
    repeated = []
    for ex in ALL_EXAMPLES:
        for _ in range(REPETITIONS):
            repeated.append(ex)

    with open(OUTPUT_PATH, "w") as f:
        for ex in repeated:
            f.write(json.dumps({"prompt": ex["prompt"], "body": ex["body"]}) + "\n")

    print(f"Wrote {len(repeated)} lines to {OUTPUT_PATH}")
    print(f"Unique examples  : {len(ALL_EXAMPLES)}")
    print(f"  Math           : {len(MATH_EXAMPLES)}")
    print(f"  String         : {len(STRING_EXAMPLES)}")
    print(f"  List/array     : {len(LIST_EXAMPLES)}")
    print(f"  Recursive/DP   : {len(RECURSIVE_EXAMPLES)}")
    print(f"  Dict/set       : {len(DICT_SET_EXAMPLES)}")
    print(f"  Edge cases     : {len(EDGE_CASE_EXAMPLES)}")
    print(f"  Graph/search   : {len(GRAPH_EXAMPLES)}")
    print(f"  Functional     : {len(FUNCTIONAL_EXAMPLES)}")
    print(f"  Math extra     : {len(MATH_EXTRA)}")
    print(f"  String extra   : {len(STRING_EXTRA)}")
    print(f"  List extra     : {len(LIST_EXTRA)}")
    print(f"  Recursive extra: {len(RECURSIVE_EXTRA)}")
    print(f"  Misc extra     : {len(MISC_EXTRA)}")
    print(f"Repetitions      : {REPETITIONS}")


if __name__ == "__main__":
    main()
