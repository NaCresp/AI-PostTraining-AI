"""Run the reproducible PostTrainBench pilot normalization pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline.extract_events import (add_event_indices, normalize_claude,
                                                     normalize_codex, normalize_plain)
from analysis.pipeline.extract_experiments import extract_experiments
from analysis.pipeline.extract_judgements import read_judgements
from analysis.pipeline.extract_metrics import read_metrics
from analysis.pipeline.filter_cohorts import cohort_counts, cohort_membership, finalize_metadata
from analysis.pipeline.inventory import inventory_file_counts, scan_task_dirs
from analysis.pipeline.parse_claude import looks_like_claude, parse_claude_records
from analysis.pipeline.parse_codex import looks_like_codex, parse_codex_records
from analysis.pipeline.parse_trace import parse_plain_trace
from analysis.pipeline.schema import (EVENT_FIELDS, EXPERIMENT_FIELDS, JUDGEMENT_FIELDS,
                                             METADATA_FIELDS, METRIC_FIELDS, write_csv,
                                             write_jsonl)


PIPELINE_VERSION = "pilot-pipeline-v1"


def _read(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _error_log_failed(row: dict[str, Any]) -> bool:
    task_dir = Path(row["absolute_task_dir"])
    path = task_dir / "error.log"
    if not path.is_file():
        return False
    text = _read(path).lower()
    return bool(text.strip() and re.search(
        r"(?:traceback|exception|^\s*error\b|command failed|process exited|exit code [1-9])",
        text, re.M,
    ))


def parse_source(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str, dict[str, int]]:
    solve_path = Path(row["solve_out_path"]) if row.get("solve_out_path") else None
    trace_path = Path(row["trace_path"]) if row.get("trace_path") else None
    stats: dict[str, int] = {}
    if solve_path:
        text = _read(solve_path)
        records, stats = parse_claude_records(text)
        if looks_like_claude(records):
            events = normalize_claude(records)
            if events:
                return events, "claude_jsonl", "high", stats
        if looks_like_codex(records):
            events = normalize_codex(records)
            if events:
                return events, "codex_jsonl", "high", stats
        if records:
            # Keep machine-readable records even when a future harness has new event names.
            events = []
            for record in records:
                text_value = record.get("text") or record.get("message") or record.get("content")
                if text_value is not None:
                    events.append({"event_type": "assistant_text", "role": "unknown",
                                   "tool_name": None, "text": str(text_value), "command": None,
                                   "command_output": None, "file_path": None, "exit_code": None,
                                   "source_line_start": record.get("_source_line_start"),
                                   "source_line_end": record.get("_source_line_end"),
                                   "parser_confidence": "medium"})
            if events:
                return events, "jsonl_unknown", "medium", stats
    if trace_path:
        trace_records, trace_stats = parse_plain_trace(_read(trace_path))
        stats = {**stats, **{f"trace_{key}": value for key, value in trace_stats.items()}}
        events = normalize_plain(trace_records)
        if events:
            return events, "plain_trace", "low", stats
    return [], "unparsed", "low", stats


def process_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    events_raw, source_format, parser_confidence, parser_stats = parse_source(row)
    events = add_event_indices(row["trajectory_id"], events_raw)
    metric_rows, metrics_status, _metric_info = read_metrics(
        Path(row["metrics_path"]) if row.get("metrics_path") else None,
        row["trajectory_id"], row.get("benchmark"),
    )
    judgement_row, judgement_status = read_judgements(row)
    has_terminator = any(event.get("event_type") == "result" for event in events)
    # error.log is often a captured judge or server transcript rather than a
    # run-status file; only explicit normalized error events affect failure status.
    has_error = any(event.get("event_type") == "error" for event in events)
    if not events:
        behavior_status = "unavailable"
    elif has_terminator:
        behavior_status = "complete"
    else:
        behavior_status = "partial"
    if has_error:
        trajectory_status = "failed"
    elif has_terminator:
        trajectory_status = "complete"
    elif events:
        trajectory_status = "incomplete"
    else:
        trajectory_status = "unknown"
    metadata = finalize_metadata(row, source_format, parser_confidence, behavior_status,
                                 metrics_status, judgement_status, trajectory_status, events)
    experiments = extract_experiments(row["trajectory_id"], events)
    return metadata, events, experiments, metric_rows, judgement_row, parser_stats


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key)) if row.get(key) is not None else "unknown" for row in rows)
    return dict(sorted(counter.items()))


def _matrix(rows: list[dict[str, Any]], row_key: str, column_key: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = {}
    for row in rows:
        row_name = str(row.get(row_key)) if row.get(row_key) is not None else "unknown"
        column_name = str(row.get(column_key)) if row.get(column_key) is not None else "unknown"
        matrix.setdefault(row_name, Counter())[column_name] += 1
    return {name: dict(sorted(counter.items())) for name, counter in sorted(matrix.items())}


def _hash_config(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"


def _hash_scripts() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_pipeline(input_root: Path, output_root: Path, config_path: Path) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pipeline config: {config_path}") from exc
    matched_config = config.get("matched_cohort", {}) if isinstance(config, dict) else {}
    matched_models = set(matched_config.get("base_models", []))
    matched_benchmarks = set(matched_config.get("benchmarks", []))
    inventory = scan_task_dirs(input_root)
    metadata_rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_experiments: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_judgements: list[dict[str, Any]] = []
    parser_stats: Counter[str] = Counter()
    for row in inventory:
        metadata, events, experiments, metrics, judgement, stats = process_row(row)
        metadata_rows.append(metadata)
        all_events.extend(events)
        all_experiments.extend(experiments)
        all_metrics.extend(metrics)
        all_judgements.append(judgement)
        parser_stats.update({key: value for key, value in stats.items() if isinstance(value, int)})
    metadata_rows.sort(key=lambda row: row["trajectory_id"])
    all_events.sort(key=lambda row: (row["trajectory_id"], row["event_index"]))
    all_experiments.sort(key=lambda row: (row["trajectory_id"], row["experiment_index"]))
    all_metrics.sort(key=lambda row: (row["trajectory_id"], row["metric_name"]))
    all_judgements.sort(key=lambda row: row["trajectory_id"])

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "trajectory_metadata.jsonl", metadata_rows)
    write_jsonl(output_root / "events.jsonl", all_events)
    write_csv(output_root / "experiments.csv", all_experiments, EXPERIMENT_FIELDS)
    write_csv(output_root / "metrics.csv", all_metrics, METRIC_FIELDS)
    write_csv(output_root / "judgements.csv", all_judgements, JUDGEMENT_FIELDS)
    inventory_fields = METADATA_FIELDS + ["event_count"]
    write_csv(output_root / "inventory.csv", metadata_rows, inventory_fields)
    counts = cohort_counts(metadata_rows, matched_models, matched_benchmarks)
    write_csv(output_root / "cohort_counts.csv", counts, ["cohort", "trajectory_count"])
    memberships = [{"trajectory_id": row["trajectory_id"], "cohort": cohort}
                   for row in metadata_rows for cohort in cohort_membership(row, matched_models, matched_benchmarks)]
    memberships.sort(key=lambda row: (row["cohort"], row["trajectory_id"]))
    write_csv(output_root / "cohort_membership.csv", memberships, ["trajectory_id", "cohort"])
    for cohort in ("all_inventory", "behavior_cohort", "outcome_cohort", "strict_clean_cohort", "matched_cohort"):
        ids = {row["trajectory_id"] for row in memberships if row["cohort"] == cohort}
        write_jsonl(output_root / f"{cohort}.jsonl", (row for row in metadata_rows if row["trajectory_id"] in ids))

    experiment_summary = {
        "candidate_rows": len(all_experiments),
        "executed_rows": sum(row.get("executed_or_proposed") == "executed" for row in all_experiments),
        "proposed_rows": sum(row.get("executed_or_proposed") == "proposed" for row in all_experiments),
        "training_launch_rows": sum(bool(row.get("training_launched")) for row in all_experiments),
        "evaluation_launch_rows": sum(bool(row.get("evaluation_launched")) for row in all_experiments),
        "checkpoint_creation_rows": sum(bool(row.get("checkpoint_created")) for row in all_experiments),
        "trajectories_with_candidate_rows": len({row["trajectory_id"] for row in all_experiments}),
    }
    metric_schema_distribution = _counts(all_metrics, "metric_schema")
    metric_name_distribution = _counts(all_metrics, "metric_name")
    judgement_parse_status = _counts(all_judgements, "judgement_parse_status")
    cohort_count_map = {row["cohort"]: row["trajectory_count"] for row in counts}
    task_count = len(inventory)
    coverage_checks = {
        "benchmark_distribution_sum": sum(value for key, value in _counts(metadata_rows, "benchmark").items() if key != "unknown"),
        "base_model_distribution_sum": sum(value for key, value in _counts(metadata_rows, "base_model").items() if key != "unknown"),
        "source_format_distribution_sum": sum(_counts(metadata_rows, "source_format").values()),
        "metadata_rows_equal_task_directories": len(metadata_rows) == task_count,
        "benchmark_sum_equals_task_directories": sum(value for key, value in _counts(metadata_rows, "benchmark").items() if key != "unknown") == task_count,
        "base_model_sum_equals_task_directories": sum(value for key, value in _counts(metadata_rows, "base_model").items() if key != "unknown") == task_count,
        "behavior_subset_of_inventory": cohort_count_map.get("behavior_cohort", 0) <= cohort_count_map.get("all_inventory", 0),
        "outcome_subset_of_behavior_or_inventory": cohort_count_map.get("outcome_cohort", 0) <= cohort_count_map.get("behavior_cohort", 0),
        "strict_clean_subset_of_outcome": cohort_count_map.get("strict_clean_cohort", 0) <= cohort_count_map.get("outcome_cohort", 0),
        "matched_subset_of_inventory": cohort_count_map.get("matched_cohort", 0) <= cohort_count_map.get("all_inventory", 0),
    }
    run_replicate_values = [row.get("run_replicate") for row in metadata_rows if row.get("run_replicate")]

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "input_root": str(input_root),
        "top_level_run_batches": len({row["batch_id"] for row in inventory}),
        "task_level_directories": len(inventory),
        "benchmark_distribution": _counts(metadata_rows, "benchmark"),
        "base_model_distribution": _counts(metadata_rows, "base_model"),
        "scaffold_family_distribution": _counts(metadata_rows, "scaffold_family"),
        "agent_model_distribution": _counts(metadata_rows, "agent_model"),
        "effort_or_mode_distribution": _counts(metadata_rows, "effort_or_mode"),
        "run_replicate_distribution": _counts(metadata_rows, "run_replicate"),
        "run_replicate_summary": {
            "unique_nonmissing_ids": len(set(run_replicate_values)),
            "missing_ids": sum(not bool(row.get("run_replicate")) for row in metadata_rows),
            "interpretation": "Parsed numeric job_id from task directory names; not assumed to be an experimental replicate label.",
        },
        "file_availability": inventory_file_counts(inventory),
        "parser_coverage": _counts(metadata_rows, "source_format"),
        "parser_stats": dict(sorted(parser_stats.items())),
        "cohort_counts": counts,
        "filter_reason_counts": dict(sorted(Counter(reason for row in metadata_rows for reason in row.get("filter_reasons", [])).items())),
        "behavior_status": _counts(metadata_rows, "behavior_status"),
        "metrics_status": _counts(metadata_rows, "metrics_status"),
        "trajectory_status": _counts(metadata_rows, "trajectory_status"),
        "judgement_status": {
            "contamination": _counts(metadata_rows, "contamination_status"),
            "disallowed_model": _counts(metadata_rows, "disallowed_model_status"),
        },
        "judgement_parse_status": judgement_parse_status,
        "benchmark_base_model_matrix": _matrix(metadata_rows, "benchmark", "base_model"),
        "event_type_distribution": _counts(all_events, "event_type"),
        "event_confidence_distribution": _counts(all_events, "parser_confidence"),
        "truncated_event_rows": sum(bool(row.get("truncated_fields")) for row in all_events),
        "metric_schema_distribution": metric_schema_distribution,
        "metric_name_distribution": metric_name_distribution,
        "experiment_summary": experiment_summary,
        "coverage_checks": coverage_checks,
        "experiment_rows": len(all_experiments),
        "event_rows": len(all_events),
        "metric_rows": len(all_metrics),
        "assumptions": [
            "Only direct task directories under each top-level batch are canonical trajectories.",
            "Missing judgement is never treated as clean.",
            "Total duration is run-level; no per-event wall-clock inference is performed.",
            "Raw benchmark metrics are retained without cross-benchmark pooling.",
        ],
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_validation_markdown(output_root / "validation_report.md", report)
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "config_sha256": _hash_config(config_path),
        "scripts_sha256": _hash_scripts(),
        "input_root": str(input_root),
        "input_task_count": len(inventory),
        "input_source_hashes_sha256": hashlib.sha256("".join(sorted(row.get("duplicate_content_hash") or "" for row in inventory)).encode()).hexdigest(),
        "canonical_outputs": [
            "trajectory_metadata.jsonl", "events.jsonl", "experiments.csv", "metrics.csv",
            "judgements.csv", "inventory.csv", "cohort_counts.csv", "cohort_membership.csv",
            "all_inventory.jsonl", "behavior_cohort.jsonl", "outcome_cohort.jsonl",
            "strict_clean_cohort.jsonl", "matched_cohort.jsonl",
            "validation_report.json", "validation_report.md",
        ],
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def write_validation_markdown(path: Path, report: dict[str, Any]) -> None:
    total = report["task_level_directories"]

    def pct(value: int, denominator: int = total) -> str:
        return f"{100.0 * value / denominator:.1f}%" if denominator else "n/a"

    def cell(value: Any) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
        output = ["| " + " | ".join(headers) + " |",
                  "| " + " | ".join("---" for _ in headers) + " |"]
        output.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
        return output

    benchmark_descriptions = {
        "AIME 2025": "Math competition problems.",
        "ArenaHardWriting": "Long-form/creative writing evaluation.",
        "BFCL": "Function-calling evaluation.",
        "GPQA Main": "Graduate-level science question answering.",
        "GSM8K": "Grade-school mathematics question answering.",
        "HealthBench": "Medical question answering with grader-based metrics.",
        "HumanEval": "Code generation evaluation.",
        "unknown": "Task name did not match a known benchmark alias.",
    }
    model_descriptions = {
        "Qwen3-1.7B-Base": "Qwen3 1.7B base model.",
        "Qwen3-4B-Base": "Qwen3 4B base model.",
        "SmolLM3-3B-Base": "SmolLM3 3B base model.",
        "Gemma-3-4B-PT": "Gemma 3 4B pretrained model.",
        "unknown": "Base-model alias could not be resolved.",
    }
    source_descriptions = {
        "claude_jsonl": "Machine-readable Claude events parsed from solve_out.txt.",
        "codex_jsonl": "Machine-readable Codex thread/item events parsed from solve_out.txt.",
        "plain_trace": "trace.txt fallback; turns/text recovered with low confidence.",
        "jsonl_unknown": "JSONL parsed, but event vocabulary was not recognized.",
        "unparsed": "No usable machine event or trace event recovered.",
    }
    status_descriptions = {
        "complete": "A parsed behavior trace has a terminal result/turn marker.",
        "partial": "Some events were recovered, but no terminal marker was found.",
        "unavailable": "No usable behavior events were recovered.",
        "failed": "An explicit normalized error event was present.",
        "incomplete": "Events exist but the run has no terminal result marker.",
        "unknown": "Insufficient evidence to classify run status.",
        "valid": "metrics.json is a non-empty object with numeric metric fields.",
        "invalid": "metrics.json exists but is invalid, empty, or has no numeric fields.",
        "missing": "No top-level metrics.json was found.",
        "clean": "The judgement text explicitly indicates no contamination/disallowed use.",
        "flagged": "The judgement text explicitly indicates contamination/disallowed use.",
    }
    judgement_status_descriptions = {
        "clean": "The judgement text explicitly indicates no contamination/disallowed use.",
        "flagged": "The judgement text explicitly indicates contamination/disallowed use.",
        "unknown": "A judgement artifact is present but its result is ambiguous.",
        "missing": "No usable judgement result is available; this is not treated as clean.",
    }
    filter_descriptions = {
        "contamination_flagged": "Contamination judgement explicitly flagged the trajectory.",
        "contamination_unknown": "Contamination judgement is missing or ambiguous.",
        "invalid_metrics_json": "metrics.json could not be used as a numeric metric object.",
        "judgement_unknown": "Disallowed-model judgement is missing or ambiguous.",
        "missing_metrics": "No usable top-level metrics.json is present.",
        "missing_solve_and_trace": "Neither solve_out.txt nor trace.txt is present.",
        "partial_behavior_trace": "Behavior events exist without a terminal marker.",
        "trajectory_failed": "An explicit normalized error event was observed.",
        "unparseable_source": "A source file exists but no usable events were recovered.",
        "workspace_missing": "The expected task/ workspace directory is absent.",
        "unrecognized_benchmark": "Benchmark alias was not recognized.",
        "unrecognized_base_model": "Base-model alias was not recognized.",
    }

    lines = [
        "# Pilot Pipeline Validation Report", "",
        "This report describes the current public PostTrainBench trajectory snapshot "
        "after deterministic inventory, parsing, normalization, and layered filtering. "
        "Counts are task-level unless explicitly labelled otherwise.", "",
        "## 1. Snapshot", "",
    ]
    lines += table(["Metric", "Current value", "Definition / interpretation"], [
        ["Pipeline version", f"`{report['pipeline_version']}`", "Code version recorded in run_manifest.json."],
        ["Input root", f"`{report['input_root']}`", "Directory scanned for this run."],
        ["Top-level run batches", report["top_level_run_batches"], "Distinct non-hidden direct directories under the input root."],
        ["Canonical task directories", total, "Direct child directories of each run batch; one inventory row per directory."],
        ["Metadata rows", total, "Rows in trajectory_metadata.jsonl; should equal canonical task directories."],
    ])
    lines += ["", "The scanner does not count nested `task/`, `task/logs/`, or nested copies as new trajectories.", ""]

    lines += ["## 2. Corpus Composition", "", "### Benchmark distribution", ""]
    lines += table(["Benchmark", "Tasks", "Share", "Meaning"], [
        [name, count, pct(count), benchmark_descriptions.get(name, "Parsed benchmark alias.")]
        for name, count in report["benchmark_distribution"].items()
    ])
    lines += ["", "### Base-model distribution", ""]
    lines += table(["Base model", "Tasks", "Share", "Meaning"], [
        [name, count, pct(count), model_descriptions.get(name, "Parsed base-model alias.")]
        for name, count in report["base_model_distribution"].items()
    ])
    lines += ["", "### Scaffold / harness distribution", ""]
    lines += table(["Scaffold family", "Tasks", "Share", "Meaning"], [
        [name, count, pct(count), {
            "claude": "Claude Code-style harness.",
            "codex": "Codex-style harness.",
            "other": "Other observed harnesses, retained rather than discarded.",
            "unknown": "Harness family could not be inferred from batch name.",
        }.get(name, "Harness family label.")]
        for name, count in report["scaffold_family_distribution"].items()
    ])
    lines += ["", "### Agent model and effort/mode labels", ""]
    lines += table(["Field", "Value", "Tasks", "Meaning"], [
        ["agent_model", name, count, "Model alias inferred conservatively from the batch name; unknown means no reliable alias." ]
        for name, count in report["agent_model_distribution"].items()
    ] + [
        ["effort_or_mode", name, count, "Observed batch mode token such as high, xhigh, 1m; null means no recognized token."]
        for name, count in report["effort_or_mode_distribution"].items()
    ])
    lines += ["", "### Run-replicate field", "",
              "The task-directory suffix is stored as run_replicate when it is a long numeric job id. "
              "It is an identifier for grouping/audit, not evidence that the corpus provides repeated experimental replicates.", ""]
    lines += table(["Metric", "Value", "Definition"], [
        ["Unique non-missing IDs", report["run_replicate_summary"]["unique_nonmissing_ids"], "Distinct parsed numeric job IDs."],
        ["Missing IDs", report["run_replicate_summary"]["missing_ids"], "Canonical tasks whose directory name has no parsed job ID."],
    ])
    lines += ["", "### Benchmark x base-model matrix", "",
              "Each cell is the number of canonical task directories after alias parsing.", ""]
    matrix = report["benchmark_base_model_matrix"]
    matrix_columns = sorted({column for row in matrix.values() for column in row})
    lines += table(["Benchmark"] + matrix_columns + ["Row total"], [
        [name] + [matrix[name].get(column, 0) for column in matrix_columns] + [sum(matrix[name].values())]
        for name in matrix
    ])

    lines += ["", "## 3. Source and Parser Coverage", ""]
    lines += table(["Source format", "Tasks", "Share", "Definition"], [
        [name, count, pct(count), source_descriptions.get(name, "Source format classification.")]
        for name, count in report["parser_coverage"].items()
    ])
    lines += ["", "### File availability", "",
              "Availability counts the number of canonical task directories with the named direct artifact. "
              "It does not assert that the artifact is valid or complete.", ""]
    lines += table(["Artifact", "Tasks with file", "Share", "Definition"], [
        [name, count, pct(count), {
            "solve_out.txt": "Primary sanitized agent trace.",
            "trace.txt": "Human-readable fallback trace.",
            "metrics.json": "Top-level outcome/training metrics artifact.",
            "judge_output.json": "Anti-cheating judge output.",
            "judgement.log": "Alternative anti-cheating judge log.",
            "time_taken.txt": "Run-level duration; not per-event timestamps.",
        }.get(name, "Inventory artifact.")]
        for name, count in report["file_availability"].items()
    ])
    lines += ["", "### Parser counters", "",
              "These counters describe parser work, not additional trajectories. `json_lines` counts accepted JSON objects; "
              "`malformed_json_lines` counts lines that looked like JSON but failed to parse; warning counters are non-event preambles.", ""]
    parser_definitions = {
        "json_lines": "Accepted machine-readable JSON objects.",
        "malformed_json_lines": "JSON-looking lines that could not be decoded.",
        "warnings": "Non-empty non-JSON lines encountered while reading solve_out.txt.",
        "trace_lines": "Lines read by the plain-trace fallback parser.",
        "trace_turns": "Turn markers recovered from trace.txt.",
        "trace_tool_calls": "Tool-call markers recovered from trace.txt.",
        "trace_text_blocks": "Normalized fallback records recovered from trace.txt.",
        "lines": "Total lines processed by a machine-event parser.",
    }
    lines += table(["Counter", "Value", "Definition"], [
        [name, value, parser_definitions.get(name, "Parser diagnostic counter.")]
        for name, value in report["parser_stats"].items()
    ])

    lines += ["", "## 4. Behavior and Run Status", ""]
    lines += table(["Status", "Tasks", "Share", "Definition"], [
        [name, count, pct(count), status_descriptions.get(name, "Normalized status label.")]
        for name, count in report["behavior_status"].items()
    ])
    lines += ["", "### Trajectory status", ""]
    lines += table(["Status", "Tasks", "Share", "Definition"], [
        [name, count, pct(count), status_descriptions.get(name, "Normalized trajectory status.")]
        for name, count in report["trajectory_status"].items()
    ])

    lines += ["", "## 5. Metrics and Judgements", ""]
    lines += table(["Metrics status", "Tasks", "Share", "Definition"], [
        [name, count, pct(count), status_descriptions.get(name, "Metrics status label.")]
        for name, count in report["metrics_status"].items()
    ])
    lines += ["", "### Metric schemas", "",
              "Rows are emitted to metrics.csv without pooling raw scores across benchmarks.", ""]
    lines += table(["Schema", "Metric rows", "Meaning"], [
        [name, count, {
            "benchmark": "Benchmark-level scalar metrics such as accuracy.",
            "healthbench": "HealthBench accuracy plus by_axis/by_theme numeric fields.",
            "training": "Training loss, epoch, runtime, or related training diagnostics.",
        }.get(name, "Parsed numeric metric schema.")]
        for name, count in report["metric_schema_distribution"].items()
    ])
    lines += ["", "### Metric-name distribution", "",
              "Metric rows are the atomic records in metrics.csv. The same trajectory can contribute multiple rows, especially for HealthBench.", ""]
    lines += table(["Metric name", "Rows", "Meaning"], [
        [name, count,
         "Primary benchmark accuracy." if name == "accuracy" else
         "HealthBench per-axis accuracy." if name.startswith("by_axis.") else
         "HealthBench per-theme accuracy." if name.startswith("by_theme.") else
         "HealthBench grader-call count; not a performance score." if name == "total_grader_calls" else
         "Parsed numeric metric field."]
        for name, count in report["metric_name_distribution"].items()
    ])
    lines += ["", "### Judgement status", ""]
    judgement_rows = []
    for kind, values in report["judgement_status"].items():
        for name, count in values.items():
            judgement_rows.append([kind, name, count, pct(count), judgement_status_descriptions.get(name, "Judgement status label.")])
    lines += table(["Judgement", "Status", "Tasks", "Share", "Definition"], judgement_rows)
    lines += ["", "Judgement parse status counts", ""]
    lines += table(["Parse status", "Tasks", "Meaning"], [
        [name, count, "Judgement artifacts were read and normalized." if name == "parsed" else
         "Judgement text was read but capped at 20,000 characters." if name == "parsed_truncated" else
         "No judgement artifact was available." if name == "missing" else
         "Judgement artifact was present but ambiguous." if name == "unknown" else "Parser status."]
        for name, count in report["judgement_parse_status"].items()
    ])

    lines += ["", "## 6. Normalized Events and Experiments", ""]
    lines += table(["Event metric", "Value", "Definition"], [
        ["Event rows", report["event_rows"], "Rows in events.jsonl after harness normalization."],
        ["Truncated event rows", report["truncated_event_rows"], "Event rows where text/command/output exceeded the 4,000-character cap."],
        ["Experiment candidate rows", report["experiment_summary"]["candidate_rows"], "Deduplicated training/evaluation/checkpoint or script-write candidates."],
        ["Executed candidate rows", report["experiment_summary"]["executed_rows"], "Candidates supported by an execution command."],
        ["Proposed candidate rows", report["experiment_summary"]["proposed_rows"], "Script/config proposal without evidence of execution."],
        ["Training launch rows", report["experiment_summary"]["training_launch_rows"], "Executed candidates classified as training launches."],
        ["Evaluation launch rows", report["experiment_summary"]["evaluation_launch_rows"], "Executed candidates classified as evaluation launches."],
        ["Checkpoint creation rows", report["experiment_summary"]["checkpoint_creation_rows"], "Executed candidates with checkpoint/save/merge evidence."],
        ["Trajectories with candidate rows", report["experiment_summary"]["trajectories_with_candidate_rows"], "Unique trajectories represented in experiments.csv."],
    ])
    lines += ["", "### Event-type distribution", ""]
    lines += table(["Event type", "Rows", "Meaning"], [
        [name, count, "Normalized event vocabulary used by downstream strategy-state extraction."]
        for name, count in report["event_type_distribution"].items()
    ])
    lines += ["", "### Event parser confidence", ""]
    lines += table(["Confidence", "Rows", "Meaning"], [
        [name, count, {
            "high": "Structured Claude/Codex JSONL adapter output.",
            "medium": "Machine-readable JSON was recovered but the event vocabulary was only partially recognized.",
            "low": "Plain-text trace fallback output; exact fields may be unavailable.",
        }.get(name, "Parser confidence label.")]
        for name, count in report["event_confidence_distribution"].items()
    ])

    lines += ["", "## 7. Cohorts", "",
              "Cohorts are layered views over the same inventory; a trajectory can belong to multiple cohorts.", ""]
    cohort_definitions = {
        "all_inventory": "Every canonical task directory, including failures and unavailable traces.",
        "behavior_cohort": "behavior_status is complete or partial; suitable for event/trajectory analysis.",
        "outcome_cohort": "Behavior is not unavailable and metrics_status is valid; suitable for benchmark-internal outcome association.",
        "strict_clean_cohort": "Complete behavior, valid metrics, explicit clean judgements, and no explicit failure.",
        "matched_cohort": "Configured base models and benchmarks: Qwen3-1.7B/Qwen3-4B on GSM8K, HumanEval, and AIME 2025.",
    }
    lines += table(["Cohort", "Tasks", "Share of inventory", "Definition / use"], [
        [row["cohort"], row["trajectory_count"], pct(row["trajectory_count"]), cohort_definitions[row["cohort"]]]
        for row in report["cohort_counts"]
    ])

    lines += ["", "## 8. Filter Reasons", "",
              "Filter reasons are status annotations, not silent deletions. A trajectory remains in all_inventory even when it is excluded from a stricter cohort.", ""]
    lines += table(["Reason", "Tasks", "Definition"], [
        [name, count, filter_descriptions.get(name, "Machine-readable exclusion or downgrade reason.")]
        for name, count in report["filter_reason_counts"].items()
    ])

    lines += ["", "## 9. Validation Checks", ""]
    lines += table(["Check", "Value", "Interpretation"], [
        [name, value, "PASS" if isinstance(value, bool) and value else "Count or diagnostic value"]
        for name, value in report["coverage_checks"].items()
    ])
    lines += ["", "## 10. Output Row Counts", ""]
    lines += table(["Output", "Rows", "Definition"], [
        ["trajectory_metadata.jsonl", total, "One normalized metadata row per canonical task directory."],
        ["events.jsonl", report["event_rows"], "One normalized event row per recovered event."],
        ["experiments.csv", report["experiment_rows"], "Candidate experiment/action rows."],
        ["metrics.csv", report["metric_rows"], "Numeric metric rows; benchmark semantics remain schema-specific."],
        ["judgements.csv", total, "One judgement status row per canonical task directory."],
    ])
    lines += ["", "## 11. Scope and Caveats", ""]
    lines += [f"- {note}" for note in report["assumptions"]]
    lines += [
        "- `time_taken.txt` supplies run-level duration only; this report does not infer per-event wall-clock time or perform survival analysis.",
        "- Event and judgement free-text fields are truncated deterministically; the raw artifact remains addressable through task_dir and source-line spans.",
        "- Experiment extraction is conservative: writing a training script is proposed, while a launch command is executed evidence.",
        "- The public corpus is observational. This validation report does not annotate strategy quality, infer intent, or make causal claims.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="PostTrainBench trajectory root")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, required=True, help="JSON pipeline configuration")
    args = parser.parse_args(argv)
    report = run_pipeline(args.input.resolve(), args.output.resolve(), args.config.resolve())
    print(json.dumps({"task_level_directories": report["task_level_directories"],
                      "cohort_counts": report["cohort_counts"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
