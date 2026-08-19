"""Shared schemas and small serialization helpers for the pilot pipeline.

The pipeline intentionally uses dictionaries at its public boundary.  This keeps
the JSONL/CSV outputs stable while allowing adapters to preserve fields that are
specific to a particular agent harness.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EVENT_FIELDS = [
    "trajectory_id", "event_index", "event_type", "role", "tool_name",
    "text", "command", "command_output", "file_path", "exit_code",
    "source_line_start", "source_line_end", "parser_confidence",
]

METRIC_FIELDS = [
    "trajectory_id", "benchmark", "metric_name", "metric_value", "stderr",
    "n_examples", "metric_source", "metric_schema", "metrics_status",
]

EXPERIMENT_FIELDS = [
    "trajectory_id", "experiment_index", "executed_or_proposed", "command",
    "script_path", "training_launched", "evaluation_launched",
    "checkpoint_created", "checkpoint_parent", "method_family",
    "update_mechanism", "data_regime", "objective_or_reward", "local_config",
    "result_reference", "extraction_confidence",
]

JUDGEMENT_FIELDS = [
    "trajectory_id", "contamination_status", "disallowed_model_status",
    "judgement_source", "judgement_text", "judgement_parse_status",
]

METADATA_FIELDS = [
    "trajectory_id", "batch_id", "task_dir", "source_format", "scaffold_family",
    "agent_model", "effort_or_mode", "benchmark", "base_model", "run_replicate",
    "budget_hours", "total_duration", "behavior_status", "workspace_status",
    "metrics_status", "contamination_status", "disallowed_model_status",
    "trajectory_status", "duplicate_content_hash", "parser_confidence",
    "filter_reasons",
]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True,
                                     default=json_default) + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {field: row.get(field) for field in fieldnames}
            for key, value in clean.items():
                if isinstance(value, (dict, list)):
                    clean[key] = json.dumps(value, ensure_ascii=True, sort_keys=True)
            writer.writerow(clean)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
