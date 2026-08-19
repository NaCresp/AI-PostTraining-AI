"""Parser for Codex item/turn JSONL records."""

from __future__ import annotations

from typing import Any

try:
    from .parse_claude import extract_json_object
except ImportError:  # pragma: no cover - supports direct script execution
    from parse_claude import extract_json_object


def parse_codex_records(text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
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


def looks_like_codex(records: list[dict[str, Any]]) -> bool:
    types = {str(record.get("type", "")) for record in records}
    return bool(types.intersection({"thread.started", "turn.started", "turn.completed",
                                    "item.started", "item.updated", "item.completed", "error"}))
