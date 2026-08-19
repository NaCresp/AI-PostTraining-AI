#!/usr/bin/env python3
"""
parse_tokens.py  -- Figure A data extraction.

Parses agent trajectory logs and aggregates LLM token consumption for the
baseline vs. experience-driven (+ human-guidance) settings on the three
PostTrainBench benchmarks.

Two log formats are handled transparently:
  * baseline `solve_out.txt`  : each agent stream event is a line
                                `[2026-06-09T06:27:18Z] {json...}`
  * experience/human `trajectory.jsonl` / `eval_trajectory.jsonl`
                                : clean one-JSON-per-line

IMPORTANT (token accounting caveat)
-----------------------------------
The two harness versions split "input" differently:
  - baseline reports most input under `input_tokens` (cache_creation absent);
  - experience/human report input under `cache_creation_input_tokens` +
    `cache_read_input_tokens`, leaving `input_tokens` ~ 0.
Therefore `input_tokens` alone is NOT comparable across settings. The
format-robust, comparable quantity is the TOTAL tokens processed:
    total = input_tokens + cache_creation_input_tokens
            + cache_read_input_tokens + output_tokens
We report that as the headline, plus the components for transparency.

Output: analysis/token_cost/tokens.csv
"""
import json
import re
import glob
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TS_PREFIX = re.compile(r"^\[[^\]]+\]\s*(\{.*)$")


def iter_assistant_usage(path):
    """Yield the `usage` dict of every assistant message in a log file,
    transparently handling both the `[ts] {json}` and plain-jsonl formats."""
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = TS_PREFIX.match(line)
            payload = m.group(1) if m else line
            if '"type":"assistant"' not in payload and '"type": "assistant"' not in payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "assistant":
                continue
            usage = obj.get("message", {}).get("usage", {})
            if usage:
                yield usage


def aggregate(path):
    agg = dict(msgs=0, input=0, output=0, cache_creation=0, cache_read=0)
    for u in iter_assistant_usage(path):
        agg["msgs"] += 1
        agg["input"] += u.get("input_tokens", 0) or 0
        agg["output"] += u.get("output_tokens", 0) or 0
        agg["cache_creation"] += u.get("cache_creation_input_tokens", 0) or 0
        agg["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
    agg["total"] = agg["input"] + agg["output"] + agg["cache_creation"] + agg["cache_read"]
    return agg


def g(pattern):
    hits = glob.glob(os.path.join(ROOT, pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


# (setting, benchmark, role, path)  -- role: "main" agent or "evaluator" agent
#
# Every row reads an archived run under trajectories/, so this resolves from a
# checkout with no extra data. The autonomous rows have no evaluator agent to
# account for -- that is the condition, not an omission.
RUNS = [
    ("autonomous", "GSM8K", "main",
     "trajectories/autonomous_gsm8k_42890926/trajectory.jsonl"),
    ("autonomous", "HumanEval", "main",
     "trajectories/autonomous_humaneval_fc9a969e/trajectory.jsonl"),
    ("autonomous", "AIME2025", "main",
     "trajectories/autonomous_aime2025_250c7e3e/trajectory.jsonl"),

    ("experience", "GSM8K", "main",   "trajectories/experience_gsm8k_855100f4/trajectory.jsonl"),
    ("experience", "GSM8K", "evaluator", "trajectories/experience_gsm8k_855100f4/eval_trajectory.jsonl"),
    ("experience", "HumanEval", "main",   "trajectories/experience_humaneval_d26cfb40/trajectory.jsonl"),
    ("experience", "HumanEval", "evaluator", "trajectories/experience_humaneval_d26cfb40/eval_trajectory.jsonl"),
    ("experience", "AIME2025", "main",   "trajectories/experience_aime2025_c247e78e/trajectory.jsonl"),
    ("experience", "AIME2025", "evaluator", "trajectories/experience_aime2025_c247e78e/eval_trajectory.jsonl"),

    ("human", "AIME2025", "main",   "trajectories/human_aime2025_14ce9dfd/trajectory.jsonl"),
    ("human", "AIME2025", "evaluator", "trajectories/human_aime2025_14ce9dfd/eval_trajectory.jsonl"),
    ("human", "AIME2025", "planning", "trajectories/human_aime2025_14ce9dfd/planning_trajectory.jsonl"),
]


def main():
    rows = []
    for setting, bench, role, pattern in RUNS:
        try:
            path = g(pattern)
        except FileNotFoundError:
            print(f"  [skip missing] {setting}/{bench}/{role}: {pattern}")
            continue
        a = aggregate(path)
        rows.append(dict(setting=setting, benchmark=bench, role=role, **a))
        print(f"{setting:11s} {bench:9s} {role:9s} msgs={a['msgs']:5d} "
              f"total={a['total']:>12,d}  (out={a['output']:>9,d} "
              f"in={a['input']:>10,d} cc={a['cache_creation']:>11,d} cr={a['cache_read']:>12,d})")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["setting", "benchmark", "role", "msgs",
                                           "input", "output", "cache_creation",
                                           "cache_read", "total"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
