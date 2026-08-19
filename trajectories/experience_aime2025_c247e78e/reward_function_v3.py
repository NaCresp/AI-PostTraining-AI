"""Reward function v3: format-shaped reward.
Gives partial credit (0.1) for producing 'Answer:' format even with wrong answer.
Full credit (1.0) for correct answer in any format.
This accelerates format learning while still rewarding correctness.
"""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    answer = extract_answer(solution_str)
    if answer is not None and is_equiv(answer, str(ground_truth)):
        return {"score": 1.0, "acc": 1.0}

    has_format = has_answer_format(solution_str)
    if has_format:
        return {"score": 0.1, "acc": 0.0}
    return {"score": 0.0, "acc": 0.0}


def has_answer_format(text):
    tail = text[-300:] if len(text) > 300 else text
    return bool(re.search(r'[Aa]nswer\s*:', tail))


def extract_answer(text):
    """Extract answer ONLY from 'Answer:' format. No \boxed{} fallback.
    This prevents the model from reward-hacking via pretraining's \boxed{} habit.
    """
    tail = text[-300:] if len(text) > 300 else text
    patterns = [
        r'[Aa]nswer\s*:\s*\$?\\?boxed\{([^}]+)\}\$?',
        r'[Aa]nswer\s*:\s*\$?\s*([^\n$,]+?)\s*\$?\s*$',
        r'[Aa]nswer\s*:\s*\$?\s*([^\n$,]+?)\s*\$?\s*\n',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, tail, re.MULTILINE)
        if matches:
            ans = matches[-1].strip().rstrip('.')
            if ans and ans != '$Answer (without quotes)$' and ans != '<your answer>':
                return ans
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
