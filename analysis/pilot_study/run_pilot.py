#!/usr/bin/env python3
"""Full-corpus Pilot Study measurement pipeline.

The script intentionally keeps the normalized data layer independent from the
older high-recall pilot-pipeline experiment table.  It parses the raw logs and
uses conservative evidence rules for launches, evaluations, and strategy
states.  All rows are keyed by trajectory_id and all source paths/line spans
are retained for auditability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO / "analysis"))

try:
    from pipeline.inventory import scan_task_dirs
except Exception:  # pragma: no cover - direct standalone fallback
    scan_task_dirs = None


EVENT_FIELDS = [
    "trajectory_id", "event_index", "source_format", "raw_event_type",
    "event_type", "role", "tool_name", "command", "command_output",
    "text", "file_path", "file_content", "exit_code", "timestamp", "call_id", "item_id",
    "lifecycle_status", "source_line_start", "source_line_end",
    "parser_confidence",
]

CALL_FIELDS = [
    "trajectory_id", "call_index", "event_index", "source_format",
    "tool_name", "command", "command_output", "exit_code", "call_id",
    "item_id", "lifecycle_status", "action_category", "action_confidence",
    "training_launched", "evaluation_launched", "checkpoint_created",
    "debugging_action", "data_preparation", "monitoring_action",
    "proposal_only", "source_line_start", "source_line_end", "result_reference",
]


def jdump(value: Any) -> str:
    # Raw logs occasionally contain lone UTF-16 surrogates.  ASCII escaping
    # keeps JSONL valid while preserving the exact code point escape sequence.
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(jdump(row) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f) for f in fields})


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bounded_text(value: Any, limit: int = 12000) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        value = jdump(value)
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def extract_json_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    starts = [0] if stripped.startswith("{") else []
    first = stripped.find("{")
    if first >= 0 and first not in starts:
        starts.append(first)
    for start in starts:
        try:
            obj = json.loads(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_json_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats = {"lines": 0, "json_lines": 0, "malformed_json_lines": 0, "warnings": 0}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return records, stats
    for line_no, line in enumerate(lines, 1):
        stats["lines"] += 1
        if not line.strip():
            continue
        obj = extract_json_line(line)
        if obj is None:
            if "{" in line:
                stats["malformed_json_lines"] += 1
            else:
                stats["warnings"] += 1
            continue
        stats["json_lines"] += 1
        obj = dict(obj)
        obj["_source_line_start"] = line_no
        obj["_source_line_end"] = line_no
        records.append(obj)
    return records, stats


def classify_records(records: list[dict[str, Any]]) -> str:
    types = {str(r.get("type", "")) for r in records}
    if types & {"step_start", "step_finish", "tool_use", "text", "reasoning"}:
        return "opencode_jsonl"
    if types & {"thread.started", "turn.started", "turn.completed", "item.started", "item.updated", "item.completed"}:
        return "codex_jsonl"
    if types & {"system", "assistant", "user", "result"}:
        return "claude_jsonl"
    return "unparsed"


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "thinking", "input_text"}:
                    chunks.append(str(item.get("text", item.get("thinking", ""))))
                elif item.get("type") == "tool_result":
                    chunks.append(str(item.get("content", "")))
        return "\n".join(x for x in chunks if x)
    return "" if content is None else str(content)


def input_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


def command_from(value: Any) -> str | None:
    if isinstance(value, str):
        return value if re.search(r"(?:python|bash|sh|accelerate|torchrun|\.py|evaluate|train|eval)", value, re.I) else None
    if isinstance(value, dict):
        for key in ("command", "cmd", "shell"):
            if isinstance(value.get(key), str):
                return value[key]
        nested = value.get("input")
        if nested is not value:
            return command_from(nested)
    return None


def file_path_from(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("file_path", "filePath", "path", "file", "filename", "notebook_path"):
            if value.get(key):
                return str(value[key])
        nested = value.get("input")
        if nested is not value:
            return file_path_from(nested)
    return None


def file_content_from(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in ("content", "newString", "new_string", "newText", "new_text", "patch"):
        if isinstance(value.get(key), str):
            return value[key]
    nested = value.get("input")
    if nested is not value:
        return file_content_from(nested)
    return None


def normalize_event(base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    row = {f: None for f in EVENT_FIELDS}
    row.update(base)
    row.update(updates)
    row["text"] = bounded_text(row.get("text"))
    row["command"] = bounded_text(row.get("command"), 20000)
    row["command_output"] = bounded_text(row.get("command_output"), 20000)
    row["file_path"] = bounded_text(row.get("file_path"), 2000)
    row["file_content"] = bounded_text(row.get("file_content"), 100000)
    return row


def normalize_claude(records: list[dict[str, Any]], trajectory_id: str, source_format: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for rec in records:
        typ = str(rec.get("type", ""))
        line = rec.get("_source_line_start")
        base = {"trajectory_id": trajectory_id, "source_format": source_format,
                "raw_event_type": typ, "source_line_start": line,
                "source_line_end": rec.get("_source_line_end", line),
                "parser_confidence": "high"}
        if typ == "system":
            events.append(normalize_event(base, event_type="system", role="system",
                                          text=bounded_text(rec.get("subtype"))))
        elif typ == "result":
            events.append(normalize_event(base, event_type="result", role="system",
                                          text=rec.get("result"), lifecycle_status="error" if rec.get("is_error") else "completed"))
        elif typ in {"assistant", "user"}:
            message = rec.get("message") if isinstance(rec.get("message"), dict) else {}
            role = str(message.get("role", typ))
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = str(part.get("type", ""))
                    if ptype in {"text", "thinking"}:
                        events.append(normalize_event(base, event_type="reasoning" if ptype == "thinking" else "assistant_text",
                                                      role=role, text=part.get("thinking", part.get("text"))))
                    elif ptype == "tool_use":
                        inp = part.get("input", {})
                        events.append(normalize_event(base, event_type="tool_call", role="assistant",
                                                      tool_name=part.get("name"), command=command_from(inp),
                                                      file_path=file_path_from(inp), file_content=file_content_from(inp),
                                                      text=bounded_text(inp),
                                                      call_id=part.get("id"), item_id=part.get("id"), lifecycle_status="completed"))
                    elif ptype == "tool_result":
                        content_text = text_from_content(part.get("content"))
                        events.append(normalize_event(base, event_type="tool_result", role="user",
                                                      command_output=content_text, text=content_text,
                                                      call_id=part.get("tool_use_id"), item_id=part.get("tool_use_id"),
                                                      lifecycle_status="error" if part.get("is_error") else "completed"))
            else:
                text = text_from_content(content)
                if text:
                    events.append(normalize_event(base, event_type="assistant_text" if role == "assistant" else "user_text",
                                                  role=role, text=text))
            # Claude sometimes stores a structured result alongside a user record.
            tool_result = rec.get("tool_use_result")
            if isinstance(tool_result, dict):
                out = tool_result.get("stdout") or tool_result.get("output") or tool_result.get("stderr")
                if out:
                    events.append(normalize_event(base, event_type="tool_result", role="user",
                                                  command_output=out, text=out,
                                                  lifecycle_status="error" if tool_result.get("is_error") else "completed"))
        else:
            text = text_from_content(rec.get("message")) or text_from_content(rec.get("part"))
            if text:
                events.append(normalize_event(base, event_type="unknown", role="unknown", text=text,
                                              parser_confidence="medium"))
    return events


def normalize_codex(records: list[dict[str, Any]], trajectory_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for rec in records:
        typ = str(rec.get("type", ""))
        line = rec.get("_source_line_start")
        base = {"trajectory_id": trajectory_id, "source_format": "codex_jsonl",
                "raw_event_type": typ, "source_line_start": line,
                "source_line_end": rec.get("_source_line_end", line),
                "parser_confidence": "high"}
        item = rec.get("item") if isinstance(rec.get("item"), dict) else {}
        item_type = str(item.get("type", ""))
        ident = item.get("id")
        if typ in {"item.started", "item.updated", "item.completed"}:
            status = item.get("status") or ("in_progress" if typ == "item.started" else "completed")
            if item_type == "command_execution":
                events.append(normalize_event(base, event_type="command_execution", role="assistant",
                                              command=item.get("command"), command_output=item.get("aggregated_output"),
                                              exit_code=item.get("exit_code"), call_id=ident, item_id=ident,
                                              lifecycle_status=status))
            elif item_type in {"agent_message", "message"}:
                events.append(normalize_event(base, event_type="assistant_text", role="assistant",
                                              text=item.get("text", item.get("content")), item_id=ident,
                                              lifecycle_status=status))
            elif item_type == "reasoning":
                events.append(normalize_event(base, event_type="reasoning", role="assistant",
                                              text=item.get("text"), item_id=ident, lifecycle_status=status))
            else:
                text = text_from_content(item)
                if text:
                    events.append(normalize_event(base, event_type="unknown", role="assistant", text=text,
                                                  item_id=ident, lifecycle_status=status, parser_confidence="medium"))
        elif typ == "turn.completed":
            events.append(normalize_event(base, event_type="result", role="system", text="turn.completed",
                                          lifecycle_status="completed"))
        elif typ == "error":
            events.append(normalize_event(base, event_type="error", role="system", text=text_from_content(rec),
                                          lifecycle_status="error"))
        elif typ in {"thread.started", "turn.started"}:
            events.append(normalize_event(base, event_type=typ.replace(".", "_"), role="system",
                                          text=typ, lifecycle_status="started"))
    return events


def normalize_opencode(records: list[dict[str, Any]], trajectory_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for rec in records:
        typ = str(rec.get("type", ""))
        line = rec.get("_source_line_start")
        part = rec.get("part") if isinstance(rec.get("part"), dict) else {}
        ptype = str(part.get("type", ""))
        base = {"trajectory_id": trajectory_id, "source_format": "opencode_jsonl",
                "raw_event_type": typ, "timestamp": rec.get("timestamp"),
                "source_line_start": line, "source_line_end": rec.get("_source_line_end", line),
                "parser_confidence": "high"}
        if typ == "tool_use" or ptype == "tool":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            inp = state.get("input", {})
            out = state.get("output") or state.get("metadata", {}).get("output") if isinstance(state.get("metadata"), dict) else state.get("output")
            events.append(normalize_event(base, event_type="tool_call", role="assistant",
                                          tool_name=part.get("tool"), command=command_from(inp),
                                          file_path=file_path_from(inp), file_content=file_content_from(inp),
                                          text=bounded_text(inp),
                                          command_output=out, exit_code=state.get("exit"),
                                          call_id=part.get("callID"), item_id=part.get("id"),
                                          lifecycle_status=state.get("status", "completed")))
        elif typ == "text" or ptype == "text":
            events.append(normalize_event(base, event_type="assistant_text", role="assistant", text=part.get("text")))
        elif typ == "step_finish":
            events.append(normalize_event(base, event_type="result", role="system", text=part.get("reason"),
                                          lifecycle_status="completed" if part.get("reason") in {None, "stop", "completed", "tool-calls"} else part.get("reason")))
        elif typ == "step_start":
            events.append(normalize_event(base, event_type="step_start", role="system", text="step_start", lifecycle_status="started"))
        else:
            text = text_from_content(part)
            if text:
                events.append(normalize_event(base, event_type="unknown", role="unknown", text=text,
                                              parser_confidence="medium"))
    return events


def normalize_plain_trace(path: Path, trajectory_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return events
    current_tool = None
    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if re.search(r"assistant\s*(?:[-—:]|turn)", stripped, re.I):
            events.append(normalize_event({"trajectory_id": trajectory_id, "source_format": "plain_trace",
                                           "raw_event_type": "assistant", "source_line_start": idx,
                                           "source_line_end": idx, "parser_confidence": "low"},
                                          event_type="assistant_text", role="assistant", text=stripped))
        elif low.startswith("tool call") or low.startswith("tool:"):
            current_tool = stripped.split(":", 1)[-1].strip()
            events.append(normalize_event({"trajectory_id": trajectory_id, "source_format": "plain_trace",
                                           "raw_event_type": "tool_call", "source_line_start": idx,
                                           "source_line_end": idx, "parser_confidence": "low"},
                                          event_type="tool_call", role="assistant", tool_name=current_tool))
        elif low.startswith("tool result") or low.startswith("output:"):
            events.append(normalize_event({"trajectory_id": trajectory_id, "source_format": "plain_trace",
                                           "raw_event_type": "tool_result", "source_line_start": idx,
                                           "source_line_end": idx, "parser_confidence": "low"},
                                          event_type="tool_result", role="user", command_output=stripped, text=stripped))
        elif re.search(r"(?:^|\s)(?:python|bash|sh)\s+[^ ]+", stripped):
            events.append(normalize_event({"trajectory_id": trajectory_id, "source_format": "plain_trace",
                                           "raw_event_type": "command", "source_line_start": idx,
                                           "source_line_end": idx, "parser_confidence": "low"},
                                          event_type="command_execution", role="assistant", command=stripped))
        else:
            events.append(normalize_event({"trajectory_id": trajectory_id, "source_format": "plain_trace",
                                           "raw_event_type": "text", "source_line_start": idx,
                                           "source_line_end": idx, "parser_confidence": "low"},
                                          event_type="assistant_text", role="assistant", text=stripped))
    return events


def harness_from_batch(batch_id: str) -> str:
    lower = batch_id.lower()
    if lower.startswith("claude_"):
        return "Claude"
    if lower.startswith("codex_"):
        return "Codex"
    if lower.startswith("opencode_"):
        return "OpenCode"
    if lower.startswith("glmx_"):
        return "GLM-X"
    if lower.startswith("qwen3max_"):
        return "Qwen3Max"
    return "Unknown"


def agent_from_batch(batch_id: str) -> str:
    lower = batch_id.lower()
    patterns = [
        (r"claude-opus-4-8", "Claude Opus 4.8"), (r"claude-opus-4-7", "Claude Opus 4.7"),
        (r"claude-opus-4-6", "Claude Opus 4.6"), (r"claude-opus-4-5", "Claude Opus 4.5"),
        (r"claude-sonnet-4-6", "Claude Sonnet 4.6"),
        (r"fable-5", "Claude Fable 5"), (r"gpt-5\.5", "GPT-5.5 Codex"),
        (r"gpt-5\.4", "GPT-5.4 Codex"), (r"gpt-5\.3-codex", "GPT-5.3 Codex"),
        (r"gemini-3\.1-pro", "Gemini 3.1 Pro"), (r"gemini-3-pro", "Gemini 3 Pro"),
        (r"glm-5\.2-preview", "GLM-5.2 Preview"), (r"zai_glm-5", "GLM-5"),
        (r"gpt-5\.1-codex-max", "GPT-5.1 Codex Max"), (r"kimi-k2\.5", "Kimi K2.5"),
        (r"kimi-k2-thinking", "Kimi K2 Thinking"), (r"minimax-m2\.5", "MiniMax M2.5"),
        (r"minimax-m2\.1", "MiniMax M2.1"), (r"glm-4\.7", "GLM-4.7"),
        (r"qwen3-max", "Qwen3 Max"),
    ]
    for pattern, value in patterns:
        if re.search(pattern, lower):
            return value
    return "Unknown"


def batch_run_label(batch_id: str) -> str:
    m = re.search(r"(?:^|_)(run\d+)(?:_|$)", batch_id.lower())
    return m.group(1) if m else ("reprompt" if "reprompt" in batch_id.lower() else "single")


def canonical_model(value: str | None) -> str | None:
    if not value:
        return None
    x = value.lower().replace(" ", "")
    if "qwen3-1.7b" in x:
        return "Qwen3-1.7B-Base"
    if "qwen3-4b" in x:
        return "Qwen3-4B-Base"
    if "smollm3" in x:
        return "SmolLM3-3B-Base"
    if "gemma-3-4b" in x:
        return "Gemma-3-4B-PT"
    return value


def action_category(command: str | None, tool_name: str | None = None) -> tuple[str, str]:
    if not command:
        return "other", "low"
    c = command.strip()
    low = c.lower()
    # Process-control commands may contain a training command as a quoted
    # argument.  They do not update model parameters themselves.
    if re.search(r"\b(?:pkill|killall)\b", low) or re.search(r"(?:^|[;&|]\s*)(?:sudo\s+)?kill\b", low):
        return "debugging", "high"
    if re.search(r"(?:^|[;&|]\s*)(?:cat|tee|printf|echo)\b[^\n]*<<[-]?\s*['\"]?[a-z0-9_-]+", low):
        # A heredoc containing ``python train.py`` is a file-write proposal,
        # not evidence that the generated script was executed.  A later shell
        # call will be captured separately if it really runs.
        return "inspection", "high"
    if re.search(r"(?:^|[;&|]\s*|['\"]\s*)(?:uv\s+)?(?:pip|conda|mamba|poetry)\s+(?:install|add)\b", low) or re.search(r"\b(?:pip|conda|mamba|poetry)\s+(?:install|add)\b", low):
        return "environment_setup", "high"
    # Shell inspection and environment probes must not become experiments.
    if re.match(r"^(?:/bin/)?(?:bash|sh)\s+-lc\s+['\"]?(?:ls|cat|sed|grep|rg|find|head|tail|less|pwd|du|wc|stat|file)\b", low):
        return "inspection", "high"
    if re.match(r"^(?:ls|cat|sed|grep|rg|find|head|tail|less|pwd|du|wc|stat|file|tree)\b", low):
        return "inspection", "high"
    if any(x in low for x in ["nvidia-smi", "timer.sh", "remaining time", "ps aux", "watch ", "tail -f", "sleep "]):
        return "monitoring", "high"
    if re.search(r"(?:^|[;&|]\s*|['\"]\s*)(?:ps|pgrep|pidof)\b", low) or re.search(r"grep\s+-[a-z]*e[^\n]*python\s+(?:train|eval)", low):
        return "monitoring", "high"
    if "py_compile" in low or "compileall" in low:
        return "debugging", "high"
    eval_signal = re.search(r"(?:evaluate|evaluation|eval\.py|inspect_eval|lm_eval|benchmark|score|pass@\d|accuracy)", low)
    if eval_signal and not re.search(r"(?:cat|sed|grep|rg|head|tail)\s+[^\n]*(?:evaluate|eval\.py)", low):
        return "evaluation", "medium"
    checkpoint_signal = re.search(r"(?:save_pretrained|save_model|save_checkpoint|merge_adapter|merge_and_unload|merge_lora|checkpoint|final_model)", low)
    if checkpoint_signal and not re.search(r"(?:ls|find|cat|grep|rg|sed|du)\s+", low):
        return "checkpoint", "medium"
    data_signal = re.search(r"(?:preprocess|preprocess|prepare_data|prepare-data|convert_data|generate_data|filter_data|sample_data|write.*dataset|save.*dataset)", low)
    if data_signal and "load_dataset" not in low:
        return "data_preparation", "medium"
    # Match an executable at a shell command boundary.  Without the boundary,
    # package names such as ``accelerate`` in ``pip install accelerate`` look
    # like a launcher.
    launcher = re.search(
        r"(?:^|[;&|]\s*|['\"]\s*)(?:(?:python(?:\d(?:\.\d+)?)?)\s+(?:[^\n]*\s)?(?:[\w./-]+\.py|[-\w]+)|(?:accelerate\s+launch|torchrun|deepspeed)\s+(?:[^\n]*\s)?[\w./-]+)",
        low,
    )
    train_signal = re.search(r"(?:train|finetun|fine-tun|sft|grpo|ppo|dpo|rft|rlhf|trl|trainer|lora|qlora|peft|reward_model|policy_gradient|reinforce)", low)
    inspection_signal = re.search(r"(?:load_dataset|dataset\s*=|--help|help\(|--version|import\s+|from\s+\w+\s+import)", low)
    inline_python = re.search(r"(?:python(?:\d(?:\.\d+)?)?)\s+(?:[^\n]*\s)?(?:-c\s|<<[-<]?\s*['\"]?\w+)", low)
    actual_update_call = re.search(r"(?:trainer\s*\.\s*(?:train|fit)\s*\(|\.fit\s*\(|loss\s*\.\s*backward\s*\(|optimizer\s*\.\s*step\s*\(|accelerator\s*\.\s*backward\s*\()", low)
    if inline_python and not actual_update_call:
        # Inline snippets are overwhelmingly inspection/data-purchase checks;
        # only count one when the body contains an explicit update call.
        return "other", "high"
    script_candidates = re.findall(r"(?:python(?:\d(?:\.\d+)?)?|bash|sh)\s+(?:-[a-z]+\s+)*([\w./-]+\.(?:py|sh))", low)
    utility_prefix = re.compile(r"^(?:prep|prepare|preprocess|build|assemble|combine|convert|generate|filter|sample|check|test|eval|evaluate|merge|inspect)(?:[_-]|$)")
    script_is_training = any(
        re.search(r"(?:^|[_./-])(?:train|sft|finetun|fine[-_]?tune|grpo|ppo|dpo|rft)(?:[_./-]|$)", Path(s).name)
        and not utility_prefix.search(Path(s).name.lower())
        for s in script_candidates
    )
    if launcher and (script_is_training or bool(actual_update_call)) and not (inspection_signal and not actual_update_call):
        return "training", "high"
    if re.search(r"(?:pytest|unittest|debug|traceback|error\.log|kill|pkill)", low):
        return "debugging", "medium"
    return "other", "low"


def strategy_family(command: str | None, text: str | None = None, category: str | None = None) -> str:
    s = ((command or "") + " " + (text or "")).lower()
    if re.search(r"(?:^|[\s_/.-])(?:grpo|ppo|reinforce|policy[_ -]?gradient|rlhf|reward model|rl trainer)(?:$|[\s_/.-])", s):
        return "rl"
    if re.search(r"(?:^|[\s_/.-])(?:dpo|ipo|kto|preference|pairwise preference)(?:$|[\s_/.-])", s):
        return "preference"
    if re.search(r"(?:^|[\s_/.-])(?:distill|knowledge distillation|teacher model)(?:$|[\s_/.-])", s):
        return "distillation"
    if re.search(r"(?:^|[\s_/.-])(?:lora|qlora|peft|adapter|low[- ]rank)(?:$|[\s_/.-])", s):
        return "peft_sft"
    if re.search(r"\b(?:sft|supervised fine[- ]?tuning|fine[- ]?tune|finetune|trainer|train\.py)\b", s):
        return "full_sft"
    if category in {"data_preparation", "evaluation"}:
        return "data_prompt_only" if category == "data_preparation" else "no_parameter_update"
    if category in {"checkpoint", "debugging", "monitoring", "inspection", "other"}:
        return "no_parameter_update"
    return "other_unknown"


# Objective labels are deliberately coarser than implementation labels.  In
# particular, full-parameter and parameter-efficient supervised training share
# the same objective.  The signature is retained so that clearly different
# objective forms (for example GRPO and PPO) are not silently merged.
OBJECTIVE_FORMS = {
    "supervised_likelihood",
    "reward_optimization",
    "preference_optimization",
    "on_policy_distillation",
}


def launched_script_names(command: str | None) -> list[str]:
    if not command:
        return []
    names = re.findall(
        r"(?:python(?:\d(?:\.\d+)?)?|bash|sh)\s+(?:-[a-z]+\s+)*([\w./-]+\.(?:py|sh))",
        command.lower(),
    )
    return [Path(name).name.lower() for name in names]


def objective_context(events: list[dict[str, Any]], call: dict[str, Any]) -> str:
    """Recover the most recent source text for the script being launched.

    Objective labels must come from the executed training script, not from a
    dataset name or an evaluator message.  Write/Edit payloads are retained in
    ``file_content``; shell heredocs and patches remain in ``command``.
    """
    targets = set(launched_script_names(call.get("command")))
    cutoff = call.get("event_index")
    candidates: list[str] = []
    for event in events:
        index = event.get("event_index")
        if cutoff is not None and index is not None and index > cutoff:
            continue
        path = Path(str(event.get("file_path") or "")).name.lower()
        command = str(event.get("command") or "")
        content = str(event.get("file_content") or "")
        text = str(event.get("text") or "")
        source = content or command or text
        if not source:
            continue
        is_write = bool(event.get("file_path")) or bool(re.search(
            r"(?:cat|tee|printf|echo)\s+[^\n]*(?:>|>>)|apply_patch|filePath|newString|new_text",
            source,
            re.I,
        ))
        if path and path in targets:
            candidates.append(source)
        elif targets and is_write and any(target in source.lower() for target in targets):
            candidates.append(source)
    # Keep the latest few edits.  A script is often assembled in several small
    # patches, while older plan text can contain misleading alternatives.
    return "\n".join(candidates[-8:])[-400000:]


def classify_objective(command: str | None, script_context: str | None = None) -> tuple[str, str, str, str]:
    """Return (objective form, signature, confidence, evidence).

    The classifier is intentionally conservative.  A keyword in a data path or
    command output is not enough; objective evidence must come from an executed
    trainer class or an explicit loss/reward structure in the script.
    """
    command_low = (command or "").lower()
    code = (script_context or "").lower()
    names = " ".join(launched_script_names(command))

    # The order prevents a script named ``train_sft.py`` with teacher-generated
    # data from being mistaken for distillation.  The objective must be present
    # in the training loop itself.
    if re.search(r"(?:gkdtrainer|on[_ -]?policy.{0,30}distill|teacher.{0,160}(?:student|logits).{0,160}(?:kl[_ -]?div|js[_ -]?div|logit[_ -]?matching)|(?:kl[_ -]?div|js[_ -]?div).{0,160}(?:teacher|student))", code, re.S):
        return "on_policy_distillation", "on_policy_distillation", "high", "teacher/student distribution matching"
    if re.search(r"(?:dpotrainer|ipotrainer|kto.?trainer|orpotrainer|cpo.?trainer|simpo|(?:chosen|rejected).{0,120}(?:log.?ratio|preference).{0,120}loss)", code, re.S):
        signature = "preference_dpo" if "dpotrainer" in code or "dpo" in names else "preference_other"
        return "preference_optimization", signature, "high", "preference objective in training code"
    if re.search(r"(?:grpotrainer|ppotrainer|rlootrainer|\breinforce\s*\(|policy[_ -]?gradient|reward[_ -]?funcs?\s*=|reward[_ -]?function\s*=|compute[_ -]?reward\s*\(|policy[_ -]?loss\s*=)", code, re.S):
        if "grpo" in code or "grpo" in names:
            signature = "reward_grpo"
        elif "ppo" in code or "ppo" in names:
            signature = "reward_ppo"
        elif "rloo" in code or "rloo" in names:
            signature = "reward_rloo"
        else:
            signature = "reward_other"
        return "reward_optimization", signature, "high", "reward objective in training code"
    if re.search(r"(?:sfttrainer|crossentropyloss|cross_entropy|data_collatorforlanguagemodeling|supervised fine|causal.?lm.{0,120}labels)", code, re.S) or (
        re.search(r"\btrainer\s*\(", code) and
        re.search(r"\b(?:labels|datacollatorforseq2seq|datacollatorforlanguagemodeling)\b", code)
    ):
        return "supervised_likelihood", "supervised_likelihood", "high", "supervised likelihood objective in training code"

    # A script name is a weaker fallback, used only when the source body is not
    # available.  Dataset/model path names are never inspected here.
    if re.search(r"(?:^|[_-])(?:grpo|ppo|rloo|reinforce)(?:[_-]|\.)", names):
        signature = "reward_grpo" if "grpo" in names else "reward_ppo" if "ppo" in names else "reward_other"
        return "reward_optimization", signature, "medium", "objective-bearing training script name"
    if re.search(r"(?:^|[_-])(?:dpo|ipo|kto|orpo|cpo|simpo)(?:[_-]|\.)", names):
        signature = "preference_dpo" if "dpo" in names else "preference_other"
        return "preference_optimization", signature, "medium", "objective-bearing training script name"
    if re.search(r"(?:^|[_-])(?:gkd|opd|distill)(?:[_-]|\.)", names):
        return "on_policy_distillation", "on_policy_distillation", "medium", "objective-bearing training script name"
    if re.search(r"(?:^|[_-])(?:sft|finetune|fine_tune)(?:[_-]|\.)", names):
        return "supervised_likelihood", "supervised_likelihood", "medium", "objective-bearing training script name"
    return "objective_unknown", "objective_unknown", "none", "no objective evidence recovered"


def objective_transitions(episodes: list[dict[str, Any]], trajectory_id: str) -> list[dict[str, Any]]:
    """Build transitions only between adjacent, recognized training updates."""
    transitions: list[dict[str, Any]] = []
    for left, right in zip(episodes, episodes[1:]):
        left_sig = left.get("objective_signature")
        right_sig = right.get("objective_signature")
        if left_sig not in OBJECTIVE_FORMS and not str(left_sig or "").startswith(("reward_", "preference_")):
            continue
        if right_sig not in OBJECTIVE_FORMS and not str(right_sig or "").startswith(("reward_", "preference_")):
            continue
        changed = left_sig != right_sig
        transitions.append({
            "trajectory_id": trajectory_id,
            "transition_index": len(transitions),
            "from_objective": left.get("objective_form"),
            "to_objective": right.get("objective_form"),
            "from_signature": left_sig,
            "to_signature": right_sig,
            "transition_type": "objective_change" if changed else "same_objective",
            "from_experiment": left.get("experiment_index"),
            "to_experiment": right.get("experiment_index"),
            "normalized_progress": right.get("normalized_progress"),
        })
    return transitions


def normalise_metric_value(value: float) -> float:
    # Most benchmark metrics are fractions; a handful of logs report percentages.
    if value > 1.0 and value <= 100.0:
        return value / 100.0
    return value


METRIC_KEYS = {"accuracy", "acc", "score", "exact_match", "pass@1", "pass@8", "reward", "success_rate", "win_rate", "mean"}


def walk_metrics(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            number = safe_float(child)
            key_low = str(key).lower().replace(" ", "_")
            if any(token in key_low for token in ("sample", "count", "grader_calls", "n_examples", "num_examples", "total_")):
                number = None
            if number is not None and (key_low in METRIC_KEYS or any(k in key_low for k in ("accuracy", "pass@", "score", "reward", "exact"))):
                found.append((name, number))
            elif isinstance(child, (dict, list)):
                found.extend(walk_metrics(child, name))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            if isinstance(child, (dict, list)):
                found.extend(walk_metrics(child, f"{prefix}[{idx}]"))
    return found


def metric_status(path: Path) -> tuple[str, list[tuple[str, float]]]:
    if not path.is_file():
        return "missing", []
    obj = read_json(path)
    if obj is None:
        return "invalid", []
    pairs = walk_metrics(obj)
    return ("valid" if pairs else "unrecognized"), pairs


def judgement_status(path: str | None, kind: str) -> tuple[str, str | None]:
    if not path or not Path(path).is_file():
        return "missing", None
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown", None
    low = text.lower()
    # Explicit negative flags are treated as flagged; missing/ambiguous remains unknown.
    # The corpus uses short natural-language verdicts rather than a stable
    # JSON schema.  Match complete phrases first, then fall back to explicit
    # boolean-style labels.  Ambiguous prose remains unknown.
    clean_phrases = (
        "no contamination detected", "no contamination", "only allowed use detected",
        "no disallowed model detected", "no disallowed use detected", "clean",
        "not detected", "not found", "false", "pass",
    )
    flagged_phrases = (
        "contamination detected", "disallowed model detected", "disallowed use detected",
        "violation detected", "unsafe", "flagged", "true", "yes", "fail",
    )
    if any(phrase in low for phrase in flagged_phrases):
        # Negated phrases must win over a substring such as "contamination"
        # in "no contamination detected".
        if not any(phrase in low for phrase in clean_phrases):
            return "flagged", bounded_text(text, 4000)
    if any(phrase in low for phrase in clean_phrases):
        return "clean", bounded_text(text, 4000)
    return "unknown", bounded_text(text, 4000)


def parse_duration(value: str | None) -> float | None:
    if not value:
        return None
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", value)]
    if not nums:
        return None
    if ":" in value:
        if len(nums) >= 3:
            return nums[-3] * 3600 + nums[-2] * 60 + nums[-1]
        if len(nums) == 2:
            return nums[-2] * 60 + nums[-1]
    return nums[-1]


def duplicate_groups(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        h = row.get("duplicate_content_hash")
        if h:
            groups[str(h)].append(row["trajectory_id"])
    return {h: ids for h, ids in groups.items() if len(ids) > 1}


def workspace_metric_points(task_dir: Path, benchmark: str, max_bytes: int) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    skip_parts = {".cache", "checkpoints", "final_model", "model", "wandb", "__pycache__"}
    try:
        files = task_dir.rglob("*")
    except OSError:
        return points
    for path in files:
        if not path.is_file() or path.name == "metrics.json":
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        if path.stat().st_size > max_bytes:
            continue
        low = path.name.lower()
        if not any(x in low for x in ("eval", "metric", "result", "score", "accuracy")):
            continue
        obj = read_json(path)
        if obj is None:
            continue
        for name, value in walk_metrics(obj):
            points.append({"metric_name": name, "metric_value": normalise_metric_value(value),
                           "metric_source": str(path.relative_to(task_dir)), "metric_schema": "workspace_json",
                           "observation_level": "intermediate"})
    return points


def output_metric_points(text: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    patterns = [
        r"(?i)\b(accuracy|acc|score|pass@\d+|exact[_ -]?match|reward)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"(?i)\b(accuracy|score)\b[^\d]{0,20}([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+)",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text or ""):
            name = m.group(1)
            if m.lastindex and m.lastindex >= 3:
                value = safe_float(m.group(2)); denom = safe_float(m.group(3))
                value = value / denom if value is not None and denom and value <= denom else None
            else:
                value = safe_float(m.group(2))
                if value is not None and "%" in m.group(0):
                    value /= 100.0
                elif value is not None and value > 1.0:
                    # Bare counts printed next to the word ``accuracy`` are
                    # common in evaluator logs (e.g. 632 samples).  Without
                    # an explicit percent sign they are not benchmark scores.
                    value = None
            if value is not None:
                points.append({"metric_name": name, "metric_value": value,
                               "metric_source": "command_output", "metric_schema": "regex",
                               "observation_level": "diagnostic"})
    return points


def make_proposals(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proposals = []
    for event in events:
        # A proposal is an explicit assistant/reasoning statement, not a
        # command payload or a tool result containing the word "train".
        if event.get("event_type") not in {"assistant_text", "reasoning"}:
            continue
        text = " ".join(str(event.get(k) or "") for k in ("text", "command"))
        low = text.lower()
        if not re.search(r"(?:plan|approach|strategy|we will|i will|try|train|fine[- ]?tun|experiment|iteration|switch|instead|next)", low):
            continue
        category, _ = action_category(event.get("command"), event.get("tool_name"))
        family = strategy_family(event.get("command"), text, category)
        if family in {"other_unknown", "no_parameter_update"} or len(text) < 30:
            continue
        proposals.append({"event_index": event.get("event_index"), "family": family,
                          "proposal_text": bounded_text(text, 4000),
                          "source_line_start": event.get("source_line_start"),
                          "source_line_end": event.get("source_line_end")})
    return proposals


def cluster_bootstrap(rows: list[dict[str, Any]], value_key: str, cluster_key: str = "batch_id", seed: int = 1, reps: int = 200) -> tuple[float | None, float | None, float | None]:
    vals = [(safe_float(r.get(value_key)), str(r.get(cluster_key) or r.get("trajectory_id"))) for r in rows]
    vals = [(v, c) for v, c in vals if v is not None]
    if not vals:
        return None, None, None
    mean = sum(v for v, _ in vals) / len(vals)
    clusters = sorted({c for _, c in vals})
    by_cluster = defaultdict(list)
    for v, c in vals:
        by_cluster[c].append(v)
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        draw = [rng.choice(clusters) for _ in clusters]
        flat = [v for c in draw for v in by_cluster[c]]
        if flat:
            samples.append(sum(flat) / len(flat))
    samples.sort()
    return mean, samples[int(0.025 * len(samples))] if samples else None, samples[int(0.975 * len(samples))] if samples else None


def extract_trajectory(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    tid = row["trajectory_id"]
    task_dir = Path(row["absolute_task_dir"])
    source_path = Path(row["source_path"]) if row.get("source_path") else None
    records: list[dict[str, Any]] = []
    parse_stats = {"lines": 0, "json_lines": 0, "malformed_json_lines": 0, "warnings": 0}
    source_format = "unparsed"
    if source_path and source_path.is_file():
        records, parse_stats = parse_json_records(source_path)
        source_format = classify_records(records)
        # A warning-only solve_out is not a plain trace.  The documented
        # fallback is trace.txt; otherwise retain an explicitly unparsed row.
        if not records and source_path.name == "solve_out.txt":
            trace_candidate = task_dir / "trace.txt"
            if trace_candidate.is_file():
                source_path = trace_candidate
                records, parse_stats = [], {"lines": 0, "json_lines": 0, "malformed_json_lines": 0, "warnings": 0}
                source_format = "plain_trace"
            else:
                source_format = "unparsed"
        elif not records and source_path.name == "trace.txt":
            source_format = "plain_trace"
    if source_format == "claude_jsonl":
        events = normalize_claude(records, tid, source_format)
    elif source_format == "codex_jsonl":
        events = normalize_codex(records, tid)
    elif source_format == "opencode_jsonl":
        events = normalize_opencode(records, tid)
    elif source_format == "plain_trace" and source_path and source_path.is_file():
        events = normalize_plain_trace(source_path, tid)
        source_format = "plain_trace" if events else "unparsed"
    else:
        events = []
    for idx, event in enumerate(events):
        event["event_index"] = idx
    # Lifecycle records are kept as events; calls merge only copies sharing an id.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    loose: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") in {"tool_call", "command_execution"}:
            key = event.get("call_id") or event.get("item_id")
            if key:
                grouped[str(key)].append(event)
            else:
                loose.append(event)
    calls: list[dict[str, Any]] = []
    for group in list(grouped.values()) + [[e] for e in loose]:
        group = sorted(group, key=lambda e: (e.get("event_index") or 0, e.get("source_line_start") or 0))
        command = next((e.get("command") for e in group if e.get("command")), None)
        output = next((e.get("command_output") for e in reversed(group) if e.get("command_output")), None)
        exit_code = next((e.get("exit_code") for e in reversed(group) if e.get("exit_code") is not None), None)
        tool = next((e.get("tool_name") for e in group if e.get("tool_name")), None)
        category, confidence = action_category(command, tool)
        # A tool call with no shell command can still be a file edit/proposal, but not an executed experiment.
        calls.append({"trajectory_id": tid, "call_index": len(calls),
                      "source_format": source_format, "tool_name": tool,
                      "command": command, "command_output": output, "exit_code": exit_code,
                      "event_index": group[0].get("event_index"),
                      "call_id": group[0].get("call_id"), "item_id": group[0].get("item_id"),
                      "lifecycle_status": next((e.get("lifecycle_status") for e in reversed(group) if e.get("lifecycle_status")), None),
                      "action_category": category, "action_confidence": confidence,
                      "training_launched": category == "training", "evaluation_launched": category == "evaluation",
                      "checkpoint_created": category == "checkpoint", "debugging_action": category == "debugging",
                      "data_preparation": category == "data_preparation", "monitoring_action": category == "monitoring",
                      "proposal_only": False, "source_line_start": group[0].get("source_line_start"),
                      "source_line_end": group[-1].get("source_line_end"),
                      "result_reference": f"{tid}:{group[0].get('event_index')}"})
    calls.sort(key=lambda x: (x.get("source_line_start") or 0, x.get("call_index") or 0))
    for i, call in enumerate(calls):
        call["call_index"] = i
    # Objective-level states are built only from actual training launches.
    # Evaluation, data preparation, debugging, and checkpoint actions remain
    # trajectory activity but never create a strategy state.
    objective_episodes: list[dict[str, Any]] = []
    objective_training_calls = [c for c in calls if c.get("training_launched")]
    for experiment_index, call in enumerate(objective_training_calls):
        context = objective_context(events, call)
        form, signature, confidence, evidence = classify_objective(call.get("command"), context)
        call["objective_form"] = form
        call["objective_signature"] = signature
        call["objective_confidence"] = confidence
        call["objective_evidence"] = evidence
        objective_episodes.append({
            "trajectory_id": tid,
            "experiment_index": experiment_index,
            "start_event_index": call.get("call_index"),
            "end_event_index": call.get("call_index"),
            "objective_form": form,
            "objective_signature": signature,
            "objective_confidence": confidence,
            "objective_evidence": evidence,
            "executed": True,
            "action_categories": ["training"],
            "source_line_start": call.get("source_line_start"),
            "source_line_end": call.get("source_line_end"),
            "launch_reference": call.get("result_reference"),
            "normalized_progress": experiment_index / max(1, len(objective_training_calls) - 1)
            if len(objective_training_calls) > 1 else 1.0,
            "configuration_id": f"{row.get('batch_id')}::{row.get('benchmark')}::{row.get('base_model')}",
        })
    objective_transition_rows = objective_transitions(objective_episodes, tid)
    proposals = make_proposals(events)
    meaningful = [c for c in calls if c["action_category"] in {"training", "evaluation", "checkpoint", "debugging", "data_preparation"}]
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for call in meaningful:
        is_training = bool(call["training_launched"])
        if current is None or is_training:
            if current is not None:
                episodes.append(current)
            fam = strategy_family(call.get("command"), call.get("command_output"), call.get("action_category"))
            call_event = call.get("event_index") if call.get("event_index") is not None else call.get("call_index")
            prior = [p for p in proposals
                     if (p.get("event_index") or -1) <= (call_event or 0)
                     and p.get("family") == fam]
            current = {"trajectory_id": tid, "episode_index": len(episodes),
                       "start_event_index": call.get("call_index"), "end_event_index": call.get("call_index"),
                       "strategy_family": fam, "method_family": fam,
                       "update_mechanism": "peft" if fam == "peft_sft" else ("parameter_update" if fam in {"full_sft", "rl", "preference", "distillation"} else "none"),
                       "data_regime": None, "objective_or_reward": None,
                       "executed": True, "proposal_only": False, "action_categories": [call["action_category"]],
                       "source_line_start": call.get("source_line_start"), "source_line_end": call.get("source_line_end"),
                       "launch_reference": call.get("result_reference"), "proposal_reference": prior[-1] if prior else None}
        else:
            current["end_event_index"] = call.get("call_index")
            current["action_categories"].append(call["action_category"])
    if current is not None:
        episodes.append(current)
    if not episodes and meaningful:
        c = meaningful[0]
        episodes.append({"trajectory_id": tid, "episode_index": 0, "start_event_index": c.get("call_index"),
                         "end_event_index": meaningful[-1].get("call_index"), "strategy_family": "no_parameter_update",
                         "method_family": "no_parameter_update", "update_mechanism": "none", "executed": False,
                         "proposal_only": True, "action_categories": [x["action_category"] for x in meaningful],
                         "source_line_start": c.get("source_line_start"), "source_line_end": meaningful[-1].get("source_line_end")})
    # An entirely text-only trajectory is still retained as an explicit unknown state.
    if not episodes:
        episodes = [{"trajectory_id": tid, "episode_index": 0, "start_event_index": None, "end_event_index": None,
                     "strategy_family": "other_unknown", "method_family": "other_unknown", "update_mechanism": "unknown",
                     "executed": False, "proposal_only": False, "action_categories": [],
                     "source_line_start": None, "source_line_end": None}]
    for ep in episodes:
        ep["configuration_id"] = f"{row.get('batch_id')}::{row.get('benchmark')}::{row.get('base_model')}"
    # Evaluation points: workspace JSON -> command output -> top-level metrics.
    eval_points: list[dict[str, Any]] = []
    for point in workspace_metric_points(task_dir, row.get("benchmark") or "", int(cfg.get("max_workspace_file_bytes", 2097152))):
        eval_points.append(point)
    for call in calls:
        if call.get("evaluation_launched") and call.get("command_output"):
            eval_points.extend(output_metric_points(call["command_output"]))
    status, final_pairs = metric_status(Path(row["metrics_path"]) if row.get("metrics_path") else Path(""))
    for name, value in final_pairs:
        eval_points.append({"metric_name": name, "metric_value": normalise_metric_value(value),
                            "metric_source": "metrics.json", "metric_schema": "top_level_metrics",
                            "observation_level": "final_only"})
    # Keep one representative per source/name/value, never invent missing intermediate values.
    unique_points = []
    seen = set()
    for point in eval_points:
        key = (point.get("metric_name"), point.get("metric_source"), point.get("metric_value"))
        if key in seen:
            continue
        seen.add(key)
        unique_points.append(point)
    eval_points = unique_points
    for point in eval_points:
        preceding = [ep for ep in episodes if ep.get("start_event_index") is not None]
        point["episode_index"] = preceding[-1]["episode_index"] if preceding else 0
        point["source_line_start"] = point.get("source_line_start")
        point["source_line_end"] = point.get("source_line_end")
        point["trajectory_id"] = tid
        point["benchmark"] = row.get("benchmark")
        point["base_model"] = row.get("base_model")
        point["harness_family"] = row.get("harness_family")
        point["agent_model"] = row.get("agent_model")
        point["batch_id"] = row.get("batch_id")
        point["configuration_id"] = f"{row.get('batch_id')}::{row.get('benchmark')}::{row.get('base_model')}"
        point["checkpoint_reference"] = None
        point["evaluation_sample_size"] = None
        point["metric_value_raw"] = point.get("metric_value")
        if point.get("metric_source") == "metrics.json":
            point["source_reference"] = row.get("metrics_path")
        elif point.get("metric_source") == "command_output":
            point["source_reference"] = f"{tid}:command_output"
        else:
            point["source_reference"] = point.get("metric_source")
    # Assign normalized episode progress; final-only points are at the final episode.
    max_ep = max(1, len(episodes) - 1)
    for point in eval_points:
        point["normalized_progress"] = (point.get("episode_index") or 0) / max_ep if len(episodes) > 1 else 1.0
    # Transitions and per-trajectory summaries.
    transitions: list[dict[str, Any]] = []
    local_count = 0
    core_count = 0
    for left, right in zip(episodes, episodes[1:]):
        lf, rf = left.get("strategy_family"), right.get("strategy_family")
        if lf != rf and lf not in {"other_unknown"} and rf not in {"other_unknown"}:
            kind = "core"
            core_count += 1
        else:
            kind = "local"
            local_count += 1
        evidence = "unknown"
        if eval_points:
            vals = [p["metric_value"] for p in eval_points if safe_float(p.get("metric_value")) is not None]
            if len(vals) >= 2:
                evidence = "positive" if vals[-1] > vals[0] + 1e-9 else "regression" if vals[-1] < vals[0] - 1e-9 else "plateau"
        transitions.append({"trajectory_id": tid, "transition_index": len(transitions),
                            "from_strategy": lf, "to_strategy": rf, "transition_type": kind,
                            "from_episode": left.get("episode_index"), "to_episode": right.get("episode_index"),
                            "normalized_progress": (right.get("episode_index") or 0) / max_ep,
                            "evidence_type": evidence, "proposal_to_execution": bool(right.get("proposal_reference")),
                            "source_line_start": right.get("source_line_start"), "source_line_end": right.get("source_line_end"),
                            "evidence_span": f"{tid}:{right.get('source_line_start')}"})
    actual_training = [c for c in calls if c.get("training_launched")]
    actual_eval = [c for c in calls if c.get("evaluation_launched")]
    final_values = [safe_float(p.get("metric_value")) for p in eval_points if p.get("metric_source") == "metrics.json"]
    all_values = [safe_float(p.get("metric_value")) for p in eval_points]
    final_metric = final_values[-1] if final_values else (all_values[-1] if all_values else None)
    best_metric = max(all_values) if all_values else None
    terminal = any(e.get("event_type") == "result" and e.get("lifecycle_status") == "completed" for e in events)
    error = any(e.get("event_type") == "error" or (e.get("event_type") == "result" and e.get("lifecycle_status") == "error") for e in events)
    behavior_status = "complete" if terminal else "partial" if events else "unavailable"
    trajectory_status = "failed" if error and not terminal else "complete" if terminal else "incomplete" if events else "unknown"
    contam, contam_text = judgement_status(row.get("contamination_path"), "contamination")
    disallowed, disallowed_text = judgement_status(row.get("disallowed_path"), "disallowed")
    # Initial strategy is the first executed parameter-update family.  A
    # pre-training evaluation/checkpoint block is retained as an episode but
    # is diagnostic context, not the agent's initial strategy.
    executed_families = [ep.get("strategy_family") for ep in episodes
                         if ep.get("executed") and ep.get("strategy_family") in
                         {"full_sft", "peft_sft", "rl", "preference", "distillation"}]
    if source_format == "unparsed":
        initial = "other_unknown"
    else:
        initial = executed_families[0] if executed_families else "no_parameter_update" if episodes else "other_unknown"
    if objective_episodes:
        objective_initial = objective_episodes[0].get("objective_form")
    elif source_format == "unparsed":
        objective_initial = "observation_unavailable"
    else:
        objective_initial = "no_parameter_update"
    objective_initial_signature = objective_episodes[0].get("objective_signature") if objective_episodes else ""
    same_objective_count = sum(t.get("transition_type") == "same_objective" for t in objective_transition_rows)
    objective_change_count = sum(t.get("transition_type") == "objective_change" for t in objective_transition_rows)
    valid_objective_pair_count = len(objective_transition_rows)
    first_objective_change = next((t for t in objective_transition_rows if t.get("transition_type") == "objective_change"), None)
    proposal_count = len(proposals)
    converted = sum(1 for t in transitions if t.get("proposal_to_execution"))
    activity_count = len(meaningful)
    progress_rows = []
    cumulative = cumulative_local = cumulative_core = cumulative_eval = 0
    action_to_episode = []
    for call in sorted(calls, key=lambda x: x.get("call_index") or 0):
        if call["action_category"] not in {"training", "evaluation", "checkpoint", "debugging", "data_preparation", "monitoring"}:
            continue
        cumulative += 1
        if call["action_category"] == "evaluation":
            cumulative_eval += 1
        # number of transition boundaries before this call (episode index proxy)
        ep_idx = 0
        for ep in episodes:
            if (ep.get("start_event_index") or 0) <= (call.get("call_index") or 0):
                ep_idx = ep.get("episode_index", 0)
        p = ep_idx / max_ep if len(episodes) > 1 else 1.0
        # local/core cumulative values are episode-boundary counts.
        cumulative_local = sum(1 for t in transitions if t["transition_type"] == "local" and (t.get("to_episode") or 0) <= ep_idx)
        cumulative_core = sum(1 for t in transitions if t["transition_type"] == "core" and (t.get("to_episode") or 0) <= ep_idx)
        state = episodes[min(ep_idx, len(episodes)-1)].get("strategy_family")
        progress_rows.append({"trajectory_id": tid, "benchmark": row.get("benchmark"), "base_model": row.get("base_model"),
                              "harness_family": row.get("harness_family"), "agent_model": row.get("agent_model"),
                              "batch_id": row.get("batch_id"), "configuration_id": row.get("configuration_id"),
                              "episode_index": ep_idx, "normalized_progress": p, "cumulative_activity": cumulative,
                              "cumulative_evaluations": cumulative_eval, "cumulative_local_refinements": cumulative_local,
                              "cumulative_core_transitions": cumulative_core, "action_category": call["action_category"],
                              "strategy_state": state, "evidence_type": "unknown", "final_metric": final_metric,
                              "best_metric": best_metric, "observation_level": "activity", "source_line_start": call.get("source_line_start")})
    if not progress_rows:
        progress_rows.append({"trajectory_id": tid, "benchmark": row.get("benchmark"), "base_model": row.get("base_model"),
                              "harness_family": row.get("harness_family"), "agent_model": row.get("agent_model"),
                              "batch_id": row.get("batch_id"), "configuration_id": row.get("configuration_id"),
                              "episode_index": 0, "normalized_progress": 1.0, "cumulative_activity": 0,
                              "cumulative_evaluations": 0, "cumulative_local_refinements": 0, "cumulative_core_transitions": 0,
                              "action_category": "unavailable", "strategy_state": initial, "evidence_type": "unknown",
                              "final_metric": final_metric, "best_metric": best_metric, "observation_level": "unavailable",
                              "source_line_start": None})
    metadata = dict(row)
    metadata.update({"harness_family": row.get("harness_family"), "source_format": source_format,
                     "parser_confidence": "high" if source_format.endswith("jsonl") else "low" if source_format == "plain_trace" else "none",
                     "behavior_status": behavior_status, "trajectory_status": trajectory_status,
                     "metrics_status": status, "contamination_status": contam, "disallowed_model_status": disallowed,
                     "terminal_event": terminal, "event_count": len(events), "command_call_count": len(calls),
                     "activity_count": activity_count, "training_count": len(actual_training), "evaluation_count": len(actual_eval),
                     "checkpoint_count": sum(1 for c in calls if c.get("checkpoint_created")),
                     "debug_count": sum(1 for c in calls if c.get("debugging_action")),
                     "data_preparation_count": sum(1 for c in calls if c.get("data_preparation")),
                     "monitoring_count": sum(1 for c in calls if c.get("monitoring_action")),
                     "training_experiment_count": len(objective_episodes),
                     "initial_objective": objective_initial,
                     "initial_objective_signature": objective_initial_signature,
                     "known_objective_count": sum(ep.get("objective_signature") not in {None, "", "objective_unknown"} for ep in objective_episodes),
                     "valid_objective_pair_count": valid_objective_pair_count,
                     "same_objective_count": same_objective_count,
                     "objective_change_count": objective_change_count,
                     "objective_change_rate": objective_change_count / valid_objective_pair_count if valid_objective_pair_count else None,
                     "within_objective_rate": same_objective_count / valid_objective_pair_count if valid_objective_pair_count else None,
                     "observed_objective_change": objective_change_count > 0,
                     "first_objective_change_progress": first_objective_change.get("normalized_progress") if first_objective_change else None,
                     "episode_count": len(episodes), "initial_strategy_family": initial,
                     "local_refinement_count": local_count, "core_transition_count": core_count,
                     "proposal_count": proposal_count, "proposal_to_execution_count": converted,
                     "proposal_to_execution_rate": converted / proposal_count if proposal_count else None,
                     "never_switched": core_count == 0, "local_development_ratio": local_count / max(1, local_count + core_count),
                     "first_transition_progress": transitions[0]["normalized_progress"] if transitions else None,
                     "final_metric": final_metric, "best_metric": best_metric,
                     "evaluation_point_count": len(eval_points), "workspace_metric_count": sum(1 for p in eval_points if p.get("metric_schema") == "workspace_json"),
                     "unknown_evidence": not bool(eval_points), "parse_lines": parse_stats["lines"],
                     "json_lines": parse_stats["json_lines"], "malformed_json_lines": parse_stats["malformed_json_lines"],
                     "filter_reasons": []})
    reasons = metadata["filter_reasons"]
    if not source_path:
        reasons.append("missing_solve_and_trace")
    if source_format == "unparsed":
        reasons.append("unparsed_source")
    if not eval_points:
        reasons.append("missing_evaluation_points")
    if status in {"missing", "invalid", "unrecognized"}:
        reasons.append(f"{status}_metrics")
    if contam in {"unknown", "missing", "flagged"}:
        reasons.append(f"{contam}_contamination_judgement")
    if disallowed in {"unknown", "missing", "flagged"}:
        reasons.append(f"{disallowed}_disallowed_model_judgement")
    return {"metadata": metadata, "events": events, "calls": calls, "episodes": episodes,
            "objective_episodes": objective_episodes, "objective_transitions": objective_transition_rows,
            "proposals": proposals, "eval_points": eval_points, "transitions": transitions,
            "progress": progress_rows, "contamination_text": contam_text, "disallowed_text": disallowed_text}


def aggregate_tables(metadata: list[dict[str, Any]], output: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    group_keys = ["harness_family", "benchmark", "base_model"]
    for key in group_keys:
        groups = defaultdict(list)
        for row in metadata:
            groups[str(row.get(key) or "Unknown")].append(row)
        out = []
        for group, rows in sorted(groups.items()):
            out.append({"group": group, "n_trajectories": len(rows),
                        "n_behavior": sum(r.get("behavior_status") in {"complete", "partial"} for r in rows),
                        "n_objective_ready": sum(r.get("initial_objective") in OBJECTIVE_FORMS for r in rows),
                        "n_outcome": sum(r.get("metrics_status") == "valid" for r in rows),
                        "mean_activity": sum(r.get("activity_count", 0) for r in rows) / max(1, len(rows)),
                        "mean_evaluations": sum(r.get("evaluation_point_count", 0) for r in rows) / max(1, len(rows)),
                        "mean_training_experiments": sum(r.get("training_experiment_count", 0) for r in rows) / max(1, len(rows)),
                        "mean_valid_objective_pairs": sum(r.get("valid_objective_pair_count", 0) for r in rows) / max(1, len(rows)),
                        "mean_objective_changes": sum(r.get("objective_change_count", 0) for r in rows) / max(1, len(rows)),
                        "objective_change_rate": (
                            sum(r.get("objective_change_count", 0) for r in rows)
                            / max(1, sum(r.get("valid_objective_pair_count", 0) for r in rows))
                        )})
        write_csv(tables / f"summary_by_{key}.csv", out)
    # Full 28-cell coverage, including zero rows if a future snapshot is incomplete.
    benchmarks = cfg.get("benchmarks") or sorted({r.get("benchmark") for r in metadata if r.get("benchmark")})
    models = cfg.get("base_models") or sorted({r.get("base_model") for r in metadata if r.get("base_model")})
    cells = []
    for benchmark in benchmarks:
        for model in models:
            rows = [r for r in metadata if r.get("benchmark") == benchmark and r.get("base_model") == model]
            cells.append({"benchmark": benchmark, "base_model": model, "cell_id": f"{benchmark}__{model}",
                          "n_trajectories": len(rows), "n_behavior": sum(r.get("behavior_status") in {"complete", "partial"} for r in rows),
                          "n_structured": sum(str(r.get("source_format", "")).endswith("jsonl") for r in rows),
                          "n_evaluation": sum(r.get("evaluation_point_count", 0) > 0 for r in rows),
                          "n_objective_ready": sum(r.get("initial_objective") in OBJECTIVE_FORMS for r in rows),
                          "n_valid_metrics": sum(r.get("metrics_status") == "valid" for r in rows),
                          "n_clean_judgement": sum(r.get("contamination_status") == "clean" and r.get("disallowed_model_status") == "clean" for r in rows),
                          "n_duplicate_rows": sum((r.get("duplicate_group_size") or 1) > 1 for r in rows)})
    write_csv(tables / "cell_coverage.csv", cells)
    # Initial objective distribution.
    initial = defaultdict(Counter)
    for r in metadata:
        if r.get("initial_objective") in OBJECTIVE_FORMS:
            initial[(r.get("harness_family"), r.get("benchmark"))][r.get("initial_objective")] += 1
    rows = []
    for (harness, benchmark), counter in sorted(initial.items()):
        total = sum(counter.values())
        entropy = -sum((n / total) * math.log(n / total) for n in counter.values() if n)
        rows.append({"harness_family": harness, "benchmark": benchmark, "n": total,
                     "objective_counts": jdump(dict(counter)), "top_objective": counter.most_common(1)[0][0] if counter else None,
                     "top_share": counter.most_common(1)[0][1] / total if counter else None,
                     "entropy": entropy, "normalized_entropy": entropy / math.log(max(2, len(counter))) if counter else 0})
    write_csv(tables / "rq1_initial_strategy.csv", rows)

    # Later objective updates.  Only adjacent known objectives enter the
    # denominator; unknown experiments are neither changes nor refinements.
    rq2_rows = []
    for h in sorted({r.get("harness_family") for r in metadata}):
        rs = [r for r in metadata if r.get("harness_family") == h]
        valid_pairs = sum(r.get("valid_objective_pair_count", 0) for r in rs)
        changes = sum(r.get("objective_change_count", 0) for r in rs)
        same = sum(r.get("same_objective_count", 0) for r in rs)
        rq2_rows.append({"harness_family": h, "n": len(rs),
                         "n_objective_ready": sum(r.get("initial_objective") in OBJECTIVE_FORMS for r in rs),
                         "n_with_valid_pair": sum(r.get("valid_objective_pair_count", 0) > 0 for r in rs),
                         "n_with_objective_change": sum(r.get("objective_change_count", 0) > 0 for r in rs),
                         "valid_objective_pairs": valid_pairs,
                         "same_objective_pairs": same,
                         "objective_change_pairs": changes,
                         "objective_change_rate": changes / valid_pairs if valid_pairs else None,
                         "within_objective_rate": same / valid_pairs if valid_pairs else None})
    write_csv(tables / "rq2_online_development.csv", rq2_rows)
    # Outcome association remains benchmark-stratified and uses recognized
    # initial objectives only.
    outcome_rows = []
    for benchmark in sorted({r.get("benchmark") for r in metadata}):
        rs = [r for r in metadata if r.get("benchmark") == benchmark and safe_float(r.get("final_metric")) is not None and r.get("initial_objective") in OBJECTIVE_FORMS]
        if not rs:
            continue
        mean_out = sum(float(r["final_metric"]) for r in rs) / len(rs)
        for objective in sorted({r.get("initial_objective") for r in rs}):
            fr = [r for r in rs if r.get("initial_objective") == objective]
            outcome_rows.append({"benchmark": benchmark, "initial_objective": objective, "n": len(fr),
                                 "mean_final_metric": sum(float(r["final_metric"]) for r in fr) / len(fr),
                                 "benchmark_mean": mean_out,
                                 "mean_centered_outcome": sum(float(r["final_metric"]) - mean_out for r in fr) / len(fr)})
    write_csv(tables / "rq4_outcome_by_strategy.csv", outcome_rows)
    return {"cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO / "data/PostTrainBench-Trajectories")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    cfg = read_json(args.config) or {}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = args.input.resolve()
    if scan_task_dirs is not None:
        inventory = scan_task_dirs(root)
    else:
        inventory = []
        for batch in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            for task in sorted(p for p in batch.iterdir() if p.is_dir() and not p.name.startswith(".")):
                inventory.append({"batch_id": batch.name, "task_dir": str(task.relative_to(root)), "absolute_task_dir": str(task),
                                  "source_path": str(next((x for x in [task / "solve_out.txt", task / "trace.txt"] if x.is_file()), "")) or None,
                                  "benchmark": task.name.split("_")[0], "base_model": None})
    # Canonicalize metadata without deduplicating content hashes.
    for row in inventory:
        row["harness_family"] = harness_from_batch(row.get("batch_id", ""))
        row["agent_model"] = agent_from_batch(row.get("batch_id", ""))
        row["batch_run"] = batch_run_label(row.get("batch_id", ""))
        row["base_model"] = canonical_model(row.get("base_model"))
        row["configuration_id"] = f"{row.get('batch_id')}::{row.get('benchmark')}::{row.get('base_model')}"
        row["cell_id"] = f"{row.get('benchmark')}__{row.get('base_model')}"
    groups = duplicate_groups(inventory)
    sizes = {tid: len(ids) for ids in groups.values() for tid in ids}
    for row in inventory:
        row["duplicate_group_size"] = sizes.get(row["trajectory_id"], 1)
        row["duplicate_group_id"] = row.get("duplicate_content_hash") if row["trajectory_id"] in sizes else None
    results = []
    all_events = []
    all_calls = []
    all_objective_eps = []
    all_eval = []
    all_objective_trans = []
    all_progress = []
    all_annotations = []
    review_rows = []
    for index, row in enumerate(sorted(inventory, key=lambda x: x.get("trajectory_id", ""))):
        result = extract_trajectory(row, cfg)
        results.append(result)
        all_events.extend(result["events"])
        all_calls.extend(result["calls"])
        all_objective_eps.extend(result["objective_episodes"])
        all_eval.extend(result["eval_points"])
        all_objective_trans.extend(result["objective_transitions"])
        all_progress.extend(result["progress"])
        meta = result["metadata"]
        # Deterministic consistency checks.  This is not independent human
        # annotation and is not reported as inter-rater reliability.
        corrections = []
        if meta["objective_change_count"] > meta["valid_objective_pair_count"]:
            meta["objective_change_count"] = meta["valid_objective_pair_count"]
            corrections.append("objective_change_count_bounded")
        for ep in result["objective_episodes"]:
            if ep.get("objective_form") not in OBJECTIVE_FORMS and not str(ep.get("objective_signature") or "").startswith(("reward_", "preference_", "objective_unknown")):
                ep["objective_form"] = "objective_unknown"
                ep["objective_signature"] = "objective_unknown"
                corrections.append("unknown_objective_normalized")
        meta["consistency_check_status"] = "checked"
        meta["consistency_checker"] = "deterministic_rules"
        meta["review_correction_count"] = len(corrections)
        meta["review_corrections"] = corrections
        all_annotations.append({"trajectory_id": meta["trajectory_id"], "benchmark": meta.get("benchmark"),
                                "base_model": meta.get("base_model"), "harness_family": meta.get("harness_family"),
                                "source_format": meta.get("source_format"), "initial_objective": meta.get("initial_objective"),
                                "training_experiment_count": meta.get("training_experiment_count"),
                                "known_objective_count": meta.get("known_objective_count"),
                                "valid_objective_pair_count": meta.get("valid_objective_pair_count"),
                                "objective_change_count": meta.get("objective_change_count"),
                                "objective_confidence": ";".join(sorted({str(ep.get("objective_confidence")) for ep in result["objective_episodes"]})),
                                "consistency_check_status": "checked", "review_correction_count": len(corrections),
                                "evidence_event_count": len(result["events"])})
        review_rows.append({"trajectory_id": meta["trajectory_id"], "checker": "deterministic_rules", "check_status": "checked",
                            "correction_count": len(corrections), "corrections": jdump(corrections),
                            "checked_fields": "objective_form,objective_signature,adjacent_objective_pair",
                            "source_event_count": len(result["events"])})
    metadata = [r["metadata"] for r in results]
    # Stable order by trajectory/event index, not wall-clock run time.
    metadata.sort(key=lambda r: r["trajectory_id"])
    all_events.sort(key=lambda r: (r["trajectory_id"], r.get("event_index") or 0))
    all_calls.sort(key=lambda r: (r["trajectory_id"], r.get("call_index") or 0))
    all_objective_eps.sort(key=lambda r: (r["trajectory_id"], r.get("experiment_index") or 0))
    all_eval.sort(key=lambda r: (r["trajectory_id"], r.get("episode_index") or 0, str(r.get("metric_name"))))
    all_objective_trans.sort(key=lambda r: (r["trajectory_id"], r.get("transition_index") or 0))
    all_progress.sort(key=lambda r: (r["trajectory_id"], r.get("normalized_progress") or 0, r.get("source_line_start") or 0))
    # Do not expose the superseded family-level labels in canonical trajectory
    # tables.  They remain available only in the in-memory parser because the
    # evaluation/activity extractor still uses its episode boundaries.
    legacy_metadata_fields = {
        "initial_strategy_family", "local_refinement_count", "core_transition_count",
        "proposal_count", "proposal_to_execution_count", "proposal_to_execution_rate",
        "never_switched", "local_development_ratio", "first_transition_progress",
    }
    for row in metadata:
        for field in legacy_metadata_fields:
            row.pop(field, None)
    for row in all_progress:
        for field in ("cumulative_local_refinements", "cumulative_core_transitions", "strategy_state"):
            row.pop(field, None)
    # Canonical normalized outputs.
    write_jsonl(output / "intermediate/events_enriched.jsonl", all_events)
    write_jsonl(output / "intermediate/execution_calls.jsonl", all_calls)
    # Objective-level experiments are the canonical experiment records.  The
    # older episode parser is retained in memory for evaluation extraction but
    # is not written as a competing strategy annotation.
    write_jsonl(output / "intermediate/experiment_episodes.jsonl", all_objective_eps)
    write_jsonl(output / "intermediate/objective_states.jsonl", all_objective_eps)
    write_jsonl(output / "intermediate/evaluation_points.jsonl", all_eval)
    write_csv(output / "intermediate/objective_transitions.csv", all_objective_trans)
    write_csv(output / "intermediate/trajectory_progress.csv", all_progress)
    dup_rows = []
    for group_id, ids in sorted(groups.items()):
        for tid in ids:
            dup_rows.append({"duplicate_group_id": group_id, "duplicate_group_size": len(ids), "trajectory_id": tid,
                             "deduplicated": False, "content_hash": group_id})
    write_csv(output / "intermediate/duplicate_groups.csv", dup_rows)
    write_jsonl(output / "annotations/trajectory_annotations.jsonl", all_annotations)
    write_jsonl(output / "annotations/episode_annotations.jsonl", all_objective_eps)
    write_jsonl(output / "annotations/consistency_check_log.jsonl", review_rows)
    write_csv(output / "annotations/annotation_summary.csv", [{"n_trajectories": len(metadata),
        "n_consistency_checked": len(review_rows), "consistency_check_coverage": len(review_rows) / max(1, len(metadata)),
        "n_with_correction": sum(r["correction_count"] > 0 for r in review_rows),
        "correction_rate": sum(r["correction_count"] > 0 for r in review_rows) / max(1, len(review_rows)),
        "unknown_objective_rate": sum(r.get("initial_objective") == "objective_unknown" for r in all_annotations) / max(1, len(all_annotations)),
        "note": "automatic objective extraction followed by deterministic consistency checks; not independent double annotation"}])
    write_csv(output / "tables/trajectory_analysis.csv", metadata)
    aggregate_tables(metadata, output, cfg)
    # Figure source tables are copied from the normalized layer so plotting
    # scripts never need to read raw logs.  These tables intentionally retain
    # every trajectory, including rows with unavailable activity/evaluation.
    figure_data = output / "figure_data"
    write_csv(figure_data / "main_claim_activity.csv", all_progress)
    write_csv(figure_data / "main_claim_evaluations.csv", all_eval)
    write_csv(figure_data / "main_claim_transitions.csv", all_objective_trans)
    write_csv(figure_data / "objective_transitions.csv", all_objective_trans)
    write_csv(figure_data / "objective_trajectories.csv", metadata)
    write_csv(figure_data / "main_claim_trajectories.csv", metadata)
    # Cohort flags, preserving every row in all_inventory.
    cohort_rows = []
    for row in metadata:
        structured = str(row.get("source_format", "")).endswith("jsonl")
        behavior = row.get("behavior_status") in {"complete", "partial"}
        artifact = structured and row.get("metrics_status") == "valid" and row.get("total_duration") and row.get("contamination_status") == "clean" and row.get("disallowed_model_status") == "clean"
        objective_ready = row.get("initial_objective") in OBJECTIVE_FORMS
        outcome_ready = objective_ready and row.get("metrics_status") == "valid" and row.get("contamination_status") == "clean" and row.get("disallowed_model_status") == "clean"
        cohort_rows.append({"trajectory_id": row["trajectory_id"], "all_inventory": True, "behavior_cohort": behavior,
                            "artifact_ready": bool(artifact), "objective_ready": bool(objective_ready), "strategy_ready": bool(objective_ready), "strategy_outcome_ready": bool(outcome_ready),
                            "matched_cell": row.get("base_model") in (cfg.get("base_models") or []) and row.get("benchmark") in (cfg.get("benchmarks") or [])})
    write_csv(output / "tables/cohort_membership.csv", cohort_rows)
    counts = []
    for name in ("all_inventory", "behavior_cohort", "artifact_ready", "objective_ready", "strategy_ready", "strategy_outcome_ready", "matched_cell"):
        counts.append({"cohort": name, "n": sum(bool(r.get(name)) for r in cohort_rows)})
    write_csv(output / "tables/cohort_counts.csv", counts)
    # Validation report in machine-readable and Chinese markdown form.
    source_counts = Counter(r.get("source_format") for r in metadata)
    harness_counts = Counter(r.get("harness_family") for r in metadata)
    bench_counts = Counter(r.get("benchmark") for r in metadata)
    model_counts = Counter(r.get("base_model") for r in metadata)
    report = {"version": cfg.get("version"), "input": str(root), "n_trajectories": len(metadata),
              "n_batches": len({r.get("batch_id") for r in metadata}), "n_cells": len({(r.get("benchmark"), r.get("base_model")) for r in metadata}),
              "source_format_counts": dict(source_counts), "harness_counts": dict(harness_counts),
              "benchmark_counts": dict(bench_counts), "base_model_counts": dict(model_counts),
              "cohort_counts": {r["cohort"]: r["n"] for r in counts},
              "event_count": len(all_events), "execution_call_count": len(all_calls),
              "objective_experiment_count": len(all_objective_eps), "evaluation_point_count": len(all_eval),
              "objective_transition_count": len(all_objective_trans),
              "duplicate_group_count": len(groups), "duplicate_row_count": sum(len(v) for v in groups.values()),
              "parser_line_totals": {k: sum(int(r.get(k, 0) or 0) for r in metadata) for k in ("parse_lines", "json_lines", "malformed_json_lines")},
              "annotation_check": {"n_checked": len(review_rows), "coverage": len(review_rows) / max(1, len(metadata)),
                                    "n_corrected": sum(r["correction_count"] > 0 for r in review_rows)}}
    (output / "tables/validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    # A concise run manifest with hashes for reproducibility.  Use an explicit
    # scope so files left by deprecated plotting scripts are not attributed to
    # the current objective-level run.
    pipeline_files = [
        p for folder in ("intermediate", "annotations", "tables")
        for p in (output / folder).rglob("*") if p.is_file()
    ]
    for name in (
        "main_claim_activity.csv", "main_claim_evaluations.csv",
        "main_claim_transitions.csv", "main_claim_trajectories.csv",
        "objective_transitions.csv", "objective_trajectories.csv",
    ):
        path = output / "figure_data" / name
        if path.is_file():
            pipeline_files.append(path)
    manifest = {"version": cfg.get("version"), "seed": cfg.get("seed"), "input": str(root), "n_trajectories": len(metadata),
                "input_inventory_sha256": hashlib.sha256(jdump(metadata).encode()).hexdigest(),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest() if args.config.is_file() else None,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "outputs": sorted(str(p.relative_to(output)) for p in pipeline_files)}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
