"""Conservative parser for anti-cheating judgement artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MAX_JUDGEMENT_TEXT_CHARS = 20_000


def _status(text: str | None, kind: str) -> str:
    if not text:
        return "missing"
    lower = text.lower()
    # A placeholder saying the artifact was not produced is evidence of
    # missingness, not evidence that the run was clean.
    if "not produced" in lower or ("judgement.txt" in lower and "produced" in lower and "no " in lower):
        return "unknown"
    if kind == "contamination" and any(token in lower for token in (
        "no contamination", "not contaminated", "contamination not detected", "\"contamination\": false", "clean",
    )):
        return "clean"
    if kind == "disallowed" and any(token in lower for token in (
        "only allowed", "no disallowed", "no prohibited", "not used", '\"disallowed\": false', "clean",
    )):
        return "clean"
    if any(token in lower for token in (
        "flagged", "contamination detected", "contaminated", "disallowed model detected",
        "prohibited model", '\"contamination\": true', '\"disallowed\": true',
    )):
        return "flagged"
    return "unknown"


def _read(path: Path | None) -> tuple[str | None, str | None]:
    if path is None or not path.is_file():
        return None, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None, None
    return text or None, str(path)


def read_judgements(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    contamination_text, contamination_source = _read(Path(row["contamination_path"]) if row.get("contamination_path") else None)
    disallowed_text, disallowed_source = _read(Path(row["disallowed_path"]) if row.get("disallowed_path") else None)
    extra_text, extra_source = _read(Path(row["judge_output_path"]) if row.get("judge_output_path") else Path(row["judgement_log_path"]) if row.get("judgement_log_path") else None)
    if extra_text and extra_source and extra_source.endswith("judge_output.json"):
        try:
            parsed = json.loads(extra_text)
            extra_text = json.dumps(parsed, ensure_ascii=True, sort_keys=True)
        except json.JSONDecodeError:
            pass
    contamination = _status(contamination_text, "contamination")
    disallowed = _status(disallowed_text, "disallowed")
    status = "parsed" if contamination_text or disallowed_text or extra_text else "missing"
    combined = "\n".join(x for x in (contamination_text, disallowed_text, extra_text) if x)
    if len(combined) > MAX_JUDGEMENT_TEXT_CHARS:
        combined = combined[:MAX_JUDGEMENT_TEXT_CHARS] + "\n...[truncated]"
        status = "parsed_truncated"
    source = ";".join(x for x in (contamination_source, disallowed_source, extra_source) if x) or None
    row_out = {
        "trajectory_id": row["trajectory_id"],
        "contamination_status": contamination,
        "disallowed_model_status": disallowed,
        "judgement_source": source,
        "judgement_text": combined or None,
        "judgement_parse_status": status,
    }
    return row_out, {"contamination_status": contamination, "disallowed_model_status": disallowed}
