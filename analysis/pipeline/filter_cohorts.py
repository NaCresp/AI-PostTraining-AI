"""Apply explicit, layered cohort rules while retaining the full inventory."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


MATCHED_MODELS = {"Qwen3-1.7B-Base", "Qwen3-4B-Base"}
MATCHED_BENCHMARKS = {"GSM8K", "HumanEval", "AIME 2025"}


def finalize_metadata(row: dict[str, Any], source_format: str, parser_confidence: str,
                      behavior_status: str, metrics_status: str,
                      judgement: dict[str, Any], trajectory_status: str,
                      events: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = list(row.get("inventory_reasons", []))
    if source_format == "unparsed":
        reasons.append("missing_solve_and_trace" if not row.get("source_path") else "unparseable_source")
    if metrics_status == "missing":
        reasons.append("missing_metrics")
    elif metrics_status == "invalid":
        reasons.append("invalid_metrics_json")
    if judgement.get("contamination_status") in {"unknown", "missing"}:
        reasons.append("contamination_unknown")
    elif judgement.get("contamination_status") == "flagged":
        reasons.append("contamination_flagged")
    if judgement.get("disallowed_model_status") in {"unknown", "missing"}:
        reasons.append("judgement_unknown")
    elif judgement.get("disallowed_model_status") == "flagged":
        reasons.append("disallowed_model_flagged")
    if row.get("workspace_status") == "missing":
        reasons.append("workspace_missing")
    if not row.get("benchmark"):
        reasons.append("unrecognized_benchmark")
    if not row.get("base_model"):
        reasons.append("unrecognized_base_model")
    if behavior_status == "partial":
        reasons.append("partial_behavior_trace")
    if trajectory_status == "failed":
        reasons.append("trajectory_failed")
    output = {
        "trajectory_id": row["trajectory_id"],
        "batch_id": row["batch_id"],
        "task_dir": row["task_dir"],
        "source_format": source_format,
        "scaffold_family": row.get("scaffold_family", "unknown"),
        "agent_model": row.get("agent_model"),
        "effort_or_mode": row.get("effort_or_mode"),
        "benchmark": row.get("benchmark"),
        "base_model": row.get("base_model"),
        "run_replicate": row.get("run_replicate"),
        "budget_hours": row.get("budget_hours"),
        "total_duration": row.get("total_duration"),
        "behavior_status": behavior_status,
        "workspace_status": row.get("workspace_status"),
        "metrics_status": metrics_status,
        "contamination_status": judgement.get("contamination_status", "missing"),
        "disallowed_model_status": judgement.get("disallowed_model_status", "missing"),
        "trajectory_status": trajectory_status,
        "duplicate_content_hash": row.get("duplicate_content_hash"),
        "parser_confidence": parser_confidence,
        "filter_reasons": sorted(set(reasons)),
        "event_count": len(events),
    }
    return output


def cohort_membership(row: dict[str, Any], matched_models: set[str] | None = None,
                      matched_benchmarks: set[str] | None = None) -> list[str]:
    matched_models = matched_models or MATCHED_MODELS
    matched_benchmarks = matched_benchmarks or MATCHED_BENCHMARKS
    cohorts = ["all_inventory"]
    if row.get("behavior_status") in {"complete", "partial"}:
        cohorts.append("behavior_cohort")
    if row.get("behavior_status") != "unavailable" and row.get("metrics_status") == "valid":
        cohorts.append("outcome_cohort")
    if (row.get("behavior_status") == "complete" and row.get("metrics_status") == "valid"
            and row.get("contamination_status") == "clean"
            and row.get("disallowed_model_status") == "clean"
            and row.get("trajectory_status") != "failed"):
        cohorts.append("strict_clean_cohort")
    if row.get("base_model") in matched_models and row.get("benchmark") in matched_benchmarks:
        cohorts.append("matched_cohort")
    return cohorts


def cohort_counts(rows: Iterable[dict[str, Any]], matched_models: set[str] | None = None,
                  matched_benchmarks: set[str] | None = None) -> list[dict[str, Any]]:
    counts = Counter()
    for row in rows:
        counts.update(cohort_membership(row, matched_models, matched_benchmarks))
    return [{"cohort": name, "trajectory_count": counts[name]}
            for name in ("all_inventory", "behavior_cohort", "outcome_cohort",
                         "strict_clean_cohort", "matched_cohort")]
