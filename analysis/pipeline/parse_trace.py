"""Conservative fallback parser for human-readable trace.txt files."""

from __future__ import annotations

import re
from typing import Any


TURN_RE = re.compile(r"(?:Assistant\s*[—-]\s*turn\s*|turn\s+)(\d+)", re.I)
TOOL_RE = re.compile(r"Tool call\s*[—:-]?\s*([^\n]+)", re.I)


def parse_plain_trace(text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    stats = {"lines": 0, "turns": 0, "tool_calls": 0, "text_blocks": 0}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stats["lines"] += 1
        turn = TURN_RE.search(line)
        if turn:
            current = {"type": "assistant_text", "text": line.strip(), "turn": int(turn.group(1)),
                       "_source_line_start": line_number, "_source_line_end": line_number}
            records.append(current)
            stats["turns"] += 1
            continue
        tool = TOOL_RE.search(line)
        if tool:
            current = {"type": "tool_call", "tool_name": tool.group(1).strip(), "text": line.strip(),
                       "_source_line_start": line_number, "_source_line_end": line_number}
            records.append(current)
            stats["tool_calls"] += 1
            continue
        if current is not None and line.strip():
            current["text"] = (current.get("text", "") + "\n" + line.rstrip()).strip()
            current["_source_line_end"] = line_number
    stats["text_blocks"] = len(records)
    return records, stats
