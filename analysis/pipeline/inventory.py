"""Canonical directory scan and metadata extraction."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_ALIASES = {
    "aime2025": "AIME 2025",
    "aime": "AIME 2025",
    "arenahardwriting": "ArenaHardWriting",
    "bfcl": "BFCL",
    "gpqamain": "GPQA Main",
    "gpqa": "GPQA Main",
    "gsm8k": "GSM8K",
    "healthbench": "HealthBench",
    "humaneval": "HumanEval",
}

BASE_MODEL_ALIASES = {
    "qwen3-1.7b-base": "Qwen3-1.7B-Base",
    "qwen_qwen3-1.7b-base": "Qwen3-1.7B-Base",
    "qwen3-4b-base": "Qwen3-4B-Base",
    "qwen_qwen3-4b-base": "Qwen3-4B-Base",
    "smollm3-3b-base": "SmolLM3-3B-Base",
    "huggingfacetb_smollm3-3b-base": "SmolLM3-3B-Base",
    "gemma-3-4b-pt": "Gemma-3-4B-PT",
    "google_gemma-3-4b-pt": "Gemma-3-4B-PT",
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _canonical_benchmark(raw: str) -> str | None:
    return BENCHMARK_ALIASES.get(raw.lower())


def _canonical_model(raw: str) -> str | None:
    normalized = raw.lower().replace(" ", "")
    if normalized in BASE_MODEL_ALIASES:
        return BASE_MODEL_ALIASES[normalized]
    for alias, value in BASE_MODEL_ALIASES.items():
        if alias in normalized:
            return value
    return None


def parse_task_name(name: str) -> dict[str, Any]:
    """Parse the observed ``benchmark_org_model_job`` convention conservatively."""
    pieces = name.split("_")
    raw_benchmark = pieces[0] if pieces else ""
    benchmark = _canonical_benchmark(raw_benchmark)
    model = _canonical_model(name)
    job_match = re.search(r"(?:^|_)(\d{5,})(?:$|_)", name)
    job_id = job_match.group(1) if job_match else None
    reasons: list[str] = []
    if benchmark is None:
        reasons.append("unrecognized_benchmark")
    if model is None:
        reasons.append("unrecognized_base_model")
    return {
        "benchmark": benchmark or raw_benchmark or None,
        "base_model": model,
        "run_replicate": job_id,
        "parse_reasons": reasons,
        "raw_task_name": name,
    }


def parse_batch_name(name: str) -> dict[str, Any]:
    lower = name.lower()
    if "claude" in lower:
        family = "claude"
    elif "codex" in lower:
        family = "codex"
    elif "opencode" in lower:
        family = "other"
    elif "glmx" in lower:
        family = "other"
    else:
        family = "unknown"

    agent_model = None
    for token in (
        "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
        "claude-sonnet-4-6", "gpt-5.5", "gpt-5.4", "gpt-5.3-codex",
        "gemini-3.1-pro", "glm-5.2-preview",
    ):
        if token in lower:
            agent_model = token
            break
    mode = None
    for token in ("xhigh", "high", "medium", "low", "1m"):
        if token in lower:
            mode = token
            break
    budget = None
    match = re.search(r"(?:^|_)(\d+)h(?:_|$)", lower)
    if match:
        budget = int(match.group(1))
    return {"scaffold_family": family, "agent_model": agent_model,
            "effort_or_mode": mode, "budget_hours": budget}


def _duration(task_dir: Path) -> str | None:
    path = task_dir / "time_taken.txt"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None


def scan_task_dirs(root: Path) -> list[dict[str, Any]]:
    """Return exactly one inventory row per direct child of each run batch."""
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for batch_dir in sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name):
        batch_info = parse_batch_name(batch_dir.name)
        for task_dir in sorted((p for p in batch_dir.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name):
            task_info = parse_task_name(task_dir.name)
            solve = task_dir / "solve_out.txt"
            trace = task_dir / "trace.txt"
            source = solve if solve.is_file() else trace if trace.is_file() else None
            content_hash = sha256_file(source) if source else None
            trajectory_id = hashlib.sha256(
                str(task_dir.relative_to(root)).encode("utf-8")
            ).hexdigest()[:20]
            rows.append({
                "trajectory_id": trajectory_id,
                "batch_id": batch_dir.name,
                "task_dir": str(task_dir.relative_to(root)),
                "absolute_task_dir": str(task_dir),
                "source_path": str(source) if source else None,
                "solve_out_path": str(solve) if solve.is_file() else None,
                "trace_path": str(trace) if trace.is_file() else None,
                "duplicate_content_hash": content_hash,
                "total_duration": _duration(task_dir),
                "workspace_status": "present" if (task_dir / "task").is_dir() else "missing",
                "metrics_path": str(task_dir / "metrics.json") if (task_dir / "metrics.json").is_file() else None,
                "judgement_log_path": str(task_dir / "judgement.log") if (task_dir / "judgement.log").is_file() else None,
                "judge_output_path": str(task_dir / "judge_output.json") if (task_dir / "judge_output.json").is_file() else None,
                "contamination_path": str(task_dir / "contamination_judgement.txt") if (task_dir / "contamination_judgement.txt").is_file() else None,
                "disallowed_path": str(task_dir / "disallowed_model_judgement.txt") if (task_dir / "disallowed_model_judgement.txt").is_file() else None,
                **batch_info,
                **{k: v for k, v in task_info.items() if k != "parse_reasons"},
                "inventory_reasons": list(task_info["parse_reasons"]),
            })
    return rows


def inventory_file_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    fields = {
        "solve_out.txt": "solve_out_path", "trace.txt": "trace_path",
        "metrics.json": "metrics_path", "judgement.log": "judgement_log_path",
        "judge_output.json": "judge_output_path", "time_taken.txt": "total_duration",
    }
    rows = list(rows)
    return {name: sum(1 for row in rows if row.get(key)) for name, key in fields.items()}
