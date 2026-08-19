#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, concatenate_datasets, load_dataset


USER_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n"
    'The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) '
    "where $ANSWER is the answer to the problem.\n\n"
    "{prompt}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) '
    "where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command."
)


BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
GSM8K_RE = re.compile(r"####\s*([-+]?\d+)\s*$")
OPENR1_BAD_HINTS = re.compile(r"\bAIME\s*2025\b|\baime25\b|math-ai/aime25", re.IGNORECASE)


def _strip_boxed(text: str) -> str:
    return BOX_RE.sub(r"\1", text)


def _extract_boxed_int(text: str) -> int | None:
    matches = BOX_RE.findall(text)
    if not matches:
        return None
    candidate = matches[-1].strip()
    candidate = re.sub(r"[^\d+-]", "", candidate)
    if not candidate or not re.fullmatch(r"[-+]?\d+", candidate):
        return None
    try:
        value = int(candidate)
    except ValueError:
        return None
    return value


def _extract_gsm8k_int(answer: str) -> int | None:
    m = GSM8K_RE.search(answer.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _format_example(problem: str, solution: str, answer: int) -> dict[str, Any]:
    user = USER_PROMPT_TEMPLATE.format(prompt=problem.strip())
    cleaned_solution = _strip_boxed(solution).strip()
    assistant = f"{cleaned_solution}\n\nANSWER: {answer}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _load_competition_math() -> Dataset:
    ds = load_dataset("dim/competition_math", split="train")

    def to_row(row: dict[str, Any]) -> dict[str, Any] | None:
        ans = _extract_boxed_int(row["solution"])
        if ans is None:
            return None
        if not (0 <= ans <= 999):
            return None
        return _format_example(row["problem"], row["solution"], ans)

    return ds.map(to_row, remove_columns=ds.column_names).filter(lambda x: x is not None)


def _load_aime2024() -> Dataset:
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    def to_row(row: dict[str, Any]) -> dict[str, Any]:
        ans = int(row["Answer"])
        return _format_example(row["Problem"], row["Solution"], ans)

    return ds.map(to_row, remove_columns=ds.column_names)

def _load_aime_1983_2024_solutions() -> Dataset:
    ds = load_dataset("Prompt48/AIME_Problem_Set_1983-2024", split="train")

    def to_row(row: dict[str, Any]) -> dict[str, Any] | None:
        solution = row.get("Solution")
        question = row.get("Question")
        if not isinstance(solution, str) or not isinstance(question, str):
            return None
        ans = _extract_boxed_int(solution)
        if ans is None:
            return None
        if not (0 <= ans <= 999):
            return None
        if OPENR1_BAD_HINTS.search(question) or OPENR1_BAD_HINTS.search(solution):
            return None
        return _format_example(question, solution, ans)

    return ds.map(to_row, remove_columns=ds.column_names).filter(lambda x: x is not None)


def _load_gsm8k(max_rows: int | None) -> Dataset:
    ds = load_dataset("gsm8k", "main", split="train")
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, ds.num_rows)))

    def to_row(row: dict[str, Any]) -> dict[str, Any] | None:
        ans = _extract_gsm8k_int(row["answer"])
        if ans is None:
            return None
        if ans < 0:
            return None
        solution = row["answer"].split("####")[0].rstrip()
        return _format_example(row["question"], solution, ans)

    return ds.map(to_row, remove_columns=ds.column_names).filter(lambda x: x is not None)


def _extract_any_int(text: str) -> int | None:
    boxed = _extract_boxed_int(text)
    if boxed is not None:
        return boxed
    m = re.search(r"(?i)(final answer|answer)\s*[:：]?\s*([-+]?\d+)\b", text)
    if m:
        try:
            return int(m.group(2))
        except ValueError:
            return None
    m2 = re.search(r"([-+]?\d+)\s*$", text.strip())
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            return None
    return None


def _load_openr1(max_rows: int | None) -> Dataset:
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, ds.num_rows)))

    def to_row(row: dict[str, Any]) -> dict[str, Any] | None:
        conv = row.get("messages")
        if not isinstance(conv, list) or not conv:
            return None
        user = next((m for m in conv if m.get("role") == "user"), None)
        assistant = next((m for m in conv if m.get("role") == "assistant"), None)
        if not user or not assistant:
            return None

        problem = str(user.get("content", "")).strip()
        solution = str(assistant.get("content", "")).strip()
        if not problem or not solution:
            return None
        if OPENR1_BAD_HINTS.search(problem) or OPENR1_BAD_HINTS.search(solution):
            return None

        ans = _extract_any_int(solution)
        if ans is None:
            return None
        if not (0 <= ans <= 999):
            return None
        solution = re.sub(r"</?think>", "", solution).strip()
        return _format_example(problem, solution, ans)

    return ds.map(to_row, remove_columns=ds.column_names).filter(lambda x: x is not None)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class BuildStats:
    train_rows: int
    val_rows: int


def build_dataset(
    out_dir: Path,
    seed: int,
    val_ratio: float,
    gsm8k_max_rows: int | None,
    openr1_max_rows: int | None,
) -> BuildStats:
    random.seed(seed)

    parts = [
        _load_competition_math(),
        _load_aime2024(),
        _load_aime_1983_2024_solutions(),
        _load_gsm8k(gsm8k_max_rows),
        _load_openr1(openr1_max_rows) if openr1_max_rows else None,
    ]
    parts = [p for p in parts if p is not None]
    ds = concatenate_datasets(parts).shuffle(seed=seed)

    n_val = max(64, int(ds.num_rows * val_ratio))
    n_val = min(n_val, max(64, ds.num_rows // 10))
    val = ds.select(range(n_val))
    train = ds.select(range(n_val, ds.num_rows))

    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)
    return BuildStats(train_rows=train.num_rows, val_rows=val.num_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/sft"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--gsm8k-max-rows", type=int, default=3000)
    p.add_argument("--openr1-max-rows", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gsm8k_max = None if args.gsm8k_max_rows <= 0 else args.gsm8k_max_rows
    openr1_max = None if args.openr1_max_rows <= 0 else args.openr1_max_rows
    stats = build_dataset(
        out_dir=args.out_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        gsm8k_max_rows=gsm8k_max,
        openr1_max_rows=openr1_max,
    )
    print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
