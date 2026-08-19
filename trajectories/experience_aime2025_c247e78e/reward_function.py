"""Custom reward function for math answer verification.
Extracts answer from \boxed{} or 'Answer:' format and compares to ground truth.
"""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Reward function for verl. Returns dict with score and acc."""
    answer = extract_answer(solution_str)
    if answer is not None and is_equiv(answer, ground_truth):
        return {"score": 1.0, "acc": 1.0}
    return {"score": 0.0, "acc": 0.0}


def extract_answer(text):
    """Extract the final answer from model output. Tries \boxed{} first, then 'Answer:' pattern."""
    # Try \boxed{} format first
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed)

    # Fallback: try "Answer:" or "answer:" pattern
    # Look for the last occurrence of "Answer:" followed by the answer
    patterns = [
        r'[Aa]nswer\s*:\s*\$?\\?boxed\{([^}]+)\}\$?',  # Answer: \boxed{X}
        r'[Aa]nswer\s*:\s*\$?([^\n$]+?)\$?\s*$',  # Answer: X at end of string
        r'[Tt]he\s+(?:final\s+)?answer\s+is\s*\$?\\?boxed\{([^}]+)\}\$?',  # The answer is \boxed{X}
        r'[Tt]he\s+(?:final\s+)?answer\s+is\s*\$?\s*([^\n$.]+?)\s*\$?\s*\.?\s*$',  # The answer is X
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return matches[-1].strip()

    return None


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str(str1))
        ss2 = strip_string(str(str2))
        if ss1 == ss2:
            return True
        # Try numeric comparison
        try:
            return abs(float(ss1) - float(ss2)) < 1e-6
        except (ValueError, TypeError):
            pass
        return False
    except Exception:
        return str(str1).strip() == str(str2).strip()


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        if s[:len(left)] == left:
            return s[len(left):]
    left = "\\boxed{"
    if s[:len(left)] == left and s[-1] == "}":
        return s[len(left):-1]
    return s


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace("\\\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = string.replace(" ", "")
    # Fix fracs
    if "\\frac" in string:
        substrs = string.split("\\frac")
        new_str = substrs[0]
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) >= 1 and substr[0] == "{":
                new_str += substr
            elif len(substr) >= 2:
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
            else:
                new_str += substr
        string = new_str
    # Fix a/b -> \frac{a}{b}
    parts = string.split("/")
    if len(parts) == 2:
        try:
            a, b = int(parts[0]), int(parts[1])
            if string == f"{a}/{b}":
                string = f"\\frac{{{a}}}{{{b}}}"
        except (ValueError, TypeError):
            pass
    if string == "0.5":
        string = "\\frac{1}{2}"
    return string
