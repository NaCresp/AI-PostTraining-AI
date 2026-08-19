"""Convert harness-specific records into a small event vocabulary."""

from __future__ import annotations

import json
from typing import Any


MAX_EVENT_FIELD_CHARS = 4_000


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _input_value(value: Any, *keys: str) -> Any:
    if not isinstance(value, dict):
        return None
    for key in keys:
        if value.get(key) is not None:
            return value[key]
    return None


def _event(base: dict[str, Any], **values: Any) -> dict[str, Any]:
    result = {
        "event_type": "assistant_text", "role": "unknown", "tool_name": None,
        "text": None, "command": None, "command_output": None, "file_path": None,
        "exit_code": None, "source_line_start": base.get("_source_line_start"),
        "source_line_end": base.get("_source_line_end"),
        "parser_confidence": base.get("_parser_confidence", "high"),
    }
    result.update(values)
    truncated: list[str] = []
    for field in ("text", "command", "command_output"):
        value = result.get(field)
        if isinstance(value, str) and len(value) > MAX_EVENT_FIELD_CHARS:
            result[field] = value[:MAX_EVENT_FIELD_CHARS] + "\n...[truncated]"
            truncated.append(field)
    result["truncated_fields"] = truncated
    return result


def _claude_content(record: dict[str, Any]) -> list[dict[str, Any]]:
    message = record.get("message")
    content = message.get("content") if isinstance(message, dict) else record.get("content")
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if content is not None:
        return [{"type": "text", "text": content}]
    return []


def normalize_claude(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for record in records:
        kind = record.get("type")
        role = "assistant" if kind == "assistant" else "user" if kind == "user" else "system" if kind == "system" else "unknown"
        if kind == "result":
            text = _text(record.get("result") or record.get("content"))
            events.append(_event(record, event_type="result", role="unknown", text=text,
                                 command_output=None))
            continue
        if kind not in {"assistant", "user", "system"}:
            continue
        content_items = _claude_content(record)
        if not content_items:
            continue
        for item in content_items:
            item_type = item.get("type")
            if item_type == "text":
                text = _text(item.get("text"))
                if text:
                    events.append(_event(record, role=role, text=text))
            elif item_type in {"tool_use", "tool_call"}:
                inp = item.get("input") or item.get("arguments") or {}
                command = _input_value(inp, "command", "cmd", "script")
                path = _input_value(inp, "file_path", "path", "filename")
                text = _text(item.get("text"))
                events.append(_event(record, event_type="tool_call", role=role,
                                     tool_name=item.get("name") or item.get("tool_name"),
                                     text=text, command=_text(command), file_path=_text(path)))
            elif item_type in {"tool_result", "tool_response"}:
                output = item.get("content") or item.get("output") or item.get("text")
                events.append(_event(record, event_type="tool_result", role="user",
                                     text=_text(output), command_output=_text(output),
                                     exit_code=item.get("exit_code")))
    return events


def normalize_codex(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for record in records:
        kind = record.get("type")
        if kind == "error":
            message = record.get("message") or record.get("error") or record
            events.append(_event(record, event_type="error", text=_text(message), role="unknown"))
            continue
        item = record.get("item") if isinstance(record.get("item"), dict) else record
        item_type = item.get("type")
        if item_type == "reasoning":
            events.append(_event(record, role="assistant", text=_text(item.get("text"))))
        elif item_type == "agent_message":
            events.append(_event(record, role="assistant", text=_text(item.get("text") or item.get("message"))))
        elif item_type == "command_execution":
            command = item.get("command")
            output = item.get("aggregated_output") or item.get("output")
            exit_code = item.get("exit_code")
            if exit_code is None and isinstance(item.get("status"), str) and item.get("status") == "completed":
                exit_code = 0
            events.append(_event(record, event_type="command_execution", role="assistant",
                                 tool_name="command_execution", command=_text(command),
                                 text=_text(command), command_output=_text(output),
                                 exit_code=exit_code))
        elif kind in {"thread.started", "turn.started", "turn.completed", "item.started", "item.updated", "item.completed"}:
            text = _text(item.get("text") or record.get("message"))
            if text:
                events.append(_event(record, role="unknown", text=text))
    return events


def normalize_plain(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for record in records:
        record = dict(record)
        record["_parser_confidence"] = "low"
        if record.get("type") == "tool_call":
            events.append(_event(record, event_type="tool_call", role="assistant",
                                 tool_name=record.get("tool_name"), text=record.get("text"),
                                 parser_confidence="low"))
        else:
            events.append(_event(record, role="assistant", text=record.get("text"),
                                 parser_confidence="low"))
    return events


def add_event_indices(trajectory_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        row = dict(event)
        row["trajectory_id"] = trajectory_id
        row["event_index"] = index
        output.append(row)
    return output
