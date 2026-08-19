"""Parser for Claude Code JSONL embedded in sanitized solve_out files."""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(line: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse a JSON object after an optional warning/timestamp prefix."""
    stripped = line.strip()
    if not stripped:
        return None, False
    candidates = [stripped] if stripped.startswith("{") else []
    start = stripped.find("{")
    if start > 0:
        candidates.append(stripped[start:])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, True
    return None, bool(start >= 0)


def parse_claude_records(text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats = {"lines": 0, "json_lines": 0, "malformed_json_lines": 0, "warnings": 0}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stats["lines"] += 1
        record, looked_like_json = extract_json_object(line)
        if record is None:
            if looked_like_json:
                stats["malformed_json_lines"] += 1
            elif line.strip():
                stats["warnings"] += 1
            continue
        stats["json_lines"] += 1
        record["_source_line_start"] = line_number
        record["_source_line_end"] = line_number
        records.append(record)
    return records, stats


def looks_like_claude(records: list[dict[str, Any]]) -> bool:
    types = {str(record.get("type", "")) for record in records}
    return bool(types.intersection({"system", "assistant", "user", "result"}))
