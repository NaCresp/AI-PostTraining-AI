#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
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
        return int(candidate)
    except ValueError:
        return None


def _format_example(problem: str, solution: str, answer: int) -> dict[str, Any]:
    user = USER_PROMPT_TEMPLATE.format(prompt=problem.strip())
    assistant = f"{_strip_boxed(solution).strip()}\n\nANSWER: {answer}"
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def _load_prompt48() -> Dataset:
    ds = load_dataset("Prompt48/AIME_Problem_Set_1983-2024", split="train")

    def to_row(row: dict[str, Any]) -> dict[str, Any] | None:
        sol = row.get("Solution")
        q = row.get("Question")
        if not isinstance(sol, str) or not isinstance(q, str):
            return None
        ans = _extract_boxed_int(sol)
        if ans is None or not (0 <= ans <= 999):
            return None
        return _format_example(q, sol, ans)

    return ds.map(to_row, remove_columns=ds.column_names).filter(lambda x: x is not None)


def _load_aime2024_maxwell() -> Dataset:
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    def to_row(row: dict[str, Any]) -> dict[str, Any]:
        return _format_example(row["Problem"], row["Solution"], int(row["Answer"]))

    return ds.map(to_row, remove_columns=ds.column_names)


def _load_aime2024_math_ai() -> Dataset:
    ds = load_dataset("math-ai/aime24", split="test")

    def to_row(row: dict[str, Any]) -> dict[str, Any]:
        prob = str(row["problem"])
        sol = str(row.get("solution", ""))
        ans = _extract_boxed_int(sol)
        if ans is None:
            raise ValueError("Failed to extract boxed answer from math-ai/aime24")
        return _format_example(prob, sol, ans)

    return ds.map(to_row, remove_columns=ds.column_names)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/aime_sft"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--val-ratio", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    ds = concatenate_datasets([_load_prompt48(), _load_aime2024_maxwell(), _load_aime2024_math_ai()]).shuffle(seed=args.seed)
    n_val = max(32, int(ds.num_rows * args.val_ratio))
    val = ds.select(range(n_val))
    train = ds.select(range(n_val, ds.num_rows))

    _write_jsonl(args.out_dir / "train.jsonl", train)
    _write_jsonl(args.out_dir / "val.jsonl", val)
    print(json.dumps({"train_rows": train.num_rows, "val_rows": val.num_rows}, indent=2))


if __name__ == "__main__":
    main()
