"""Schema-aware, lossless-enough extraction of heterogeneous metrics.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _flatten_numeric(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(value, bool):
        return rows
    if isinstance(value, (int, float)):
        rows.append((prefix, float(value)))
    elif isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_numeric(value[key], child))
    return rows


def read_metrics(path: Path | None, trajectory_id: str, benchmark: str | None) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    if path is None or not path.is_file():
        return [], "missing", {"parse_status": "missing"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return [], "invalid", {"parse_status": "invalid_json"}
    if not isinstance(raw, dict) or not raw:
        return [], "invalid", {"parse_status": "not_nonempty_object"}
    if any(key in raw for key in ("eval_loss", "epoch", "train_loss", "learning_rate")):
        schema = "training"
    elif "by_axis" in raw or "by_theme" in raw or "total_grader_calls" in raw:
        schema = "healthbench"
    else:
        schema = "benchmark"
    rows: list[dict[str, Any]] = []
    top_stderr = raw.get("stderr") if isinstance(raw.get("stderr"), (int, float)) else None
    n_examples = raw.get("n_examples") if isinstance(raw.get("n_examples"), (int, float)) else None
    for name, value in _flatten_numeric(raw):
        if name in {"stderr", "n_examples"}:
            continue
        rows.append({
            "trajectory_id": trajectory_id,
            "benchmark": benchmark,
            "metric_name": name,
            "metric_value": value,
            "stderr": top_stderr if name == "accuracy" else None,
            "n_examples": n_examples,
            "metric_source": str(path),
            "metric_schema": schema,
            "metrics_status": "valid",
        })
    if not rows:
        return [], "invalid", {"parse_status": "no_numeric_metrics", "raw": raw}
    return rows, "valid", {"parse_status": "valid", "schema": schema, "raw": raw}
