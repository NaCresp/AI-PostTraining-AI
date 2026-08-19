#!/usr/bin/env python3
"""Run AIME pass@8 evaluations for selected checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = REPO_ROOT / "experiments" / "evaluator" / "aime" / "evaluate.py"

# The checkpoints the paper's pass@8 comparison was run over, given as they sat
# in each run's working directory. Model weights are not redistributed, so these
# paths do not resolve in a checkout -- pass --checkpoint NAME PATH to score
# your own.
DEFAULT_CHECKPOINTS = [
    ("human_v1_step20", "trajectories/human_aime2025_14ce9dfd/checkpoints/grpo_v1/global_step_20"),
    ("human_v3_step40", "trajectories/human_aime2025_14ce9dfd/checkpoints/grpo_v3/global_step_40"),
    ("human_v3_step50", "trajectories/human_aime2025_14ce9dfd/checkpoints/grpo_v3/global_step_50"),
    ("human_v3_step60", "trajectories/human_aime2025_14ce9dfd/checkpoints/grpo_v3/global_step_60"),
    ("experience_v2_step10", "trajectories/experience_aime2025_c247e78e/checkpoints/grpo_v2/global_step_10/merged_hf"),
    ("experience_v3_step20", "trajectories/experience_aime2025_c247e78e/checkpoints/grpo_v3/global_step_20/merged_hf"),
]


def load_evaluator():
    spec = importlib.util.spec_from_file_location("aime_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import evaluator from {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_records(name: str, checkpoint: str, records: list[dict]) -> dict:
    total = len(records)
    passed = [r for r in records if any(r.get("per_sample_acc", []))]
    sample_correct = sum(sum(1 for acc in r.get("per_sample_acc", []) if acc) for r in records)
    sample_total = sum(len(r.get("per_sample_acc", [])) for r in records)
    return {
        "name": name,
        "checkpoint": checkpoint,
        "pass_at_8": round(len(passed) / total, 4) if total else 0.0,
        "pass_at_8_correct": len(passed),
        "total_problems": total,
        "sample_accuracy": round(sample_correct / sample_total, 4) if sample_total else 0.0,
        "sample_correct": sample_correct,
        "sample_total": sample_total,
        "correct_ground_truths": [r.get("ground_truth") for r in passed],
        "per_problem": [
            {
                "index": r.get("index"),
                "ground_truth": r.get("ground_truth"),
                "num_correct_samples": sum(1 for acc in r.get("per_sample_acc", []) if acc),
                "pass": any(r.get("per_sample_acc", [])),
            }
            for r in records
        ],
    }


def run_single_in_child(name: str, checkpoint: str, args: argparse.Namespace) -> dict:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute() and not checkpoint.startswith(("Qwen/", "meta-llama/")):
        checkpoint_path = REPO_ROOT / checkpoint

    os.environ["EVAL_N_SAMPLES"] = str(args.samples)
    os.environ["EVAL_TEMPERATURE"] = str(args.temperature)
    os.environ["EVAL_TOP_P"] = str(args.top_p)
    os.environ["EVAL_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["EVAL_MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    evaluator = load_evaluator()
    evaluator.EVAL_N_SAMPLES = args.samples
    evaluator.EVAL_TEMPERATURE = args.temperature
    evaluator.EVAL_TOP_P = args.top_p
    evaluator.EVAL_MAX_TOKENS = args.max_tokens
    evaluator.EVAL_MAX_MODEL_LEN = args.max_model_len
    evaluator.MERGE_CACHE_DIR = str(REPO_ROOT / "experiments" / "human_guidance" / "pass8_merge_cache" / name)

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = evaluator.select_eval_gpus(args.tp_size)

    model_path = evaluator.resolve_checkpoint_path(str(checkpoint_path))
    df = evaluator.load_test_data()
    prompts = df["prompt"].tolist()
    prompts = [p.tolist() if hasattr(p, "tolist") else p for p in prompts]
    all_responses = evaluator.generate_responses(
        model_path,
        prompts,
        tp_size=args.tp_size,
        n_samples=args.samples,
    )

    records = []
    for i, (responses, row) in enumerate(zip(all_responses, df.itertuples())):
        gt = row.reward_model["ground_truth"]
        per_sample_acc = []
        sample_details = []
        for resp in responses:
            result = evaluator.score_response(resp, gt)
            per_sample_acc.append(bool(result["acc"]))
            sample_details.append({
                "pred": result["pred"],
                "acc": bool(result["acc"]),
                "response_preview": resp[-200:],
            })
        records.append({
            "index": i,
            "data_source": row.data_source,
            "ground_truth": gt,
            "per_sample_acc": per_sample_acc,
            "correct": any(per_sample_acc),
            "samples": sample_details,
        })

    summary = summarize_records(name, str(checkpoint_path), records)
    sample_level_results = evaluator.aggregate_results(records)
    summary["evaluator_sample_level_results"] = sample_level_results
    return summary


def run_one(name: str, checkpoint: str, args: argparse.Namespace) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-name",
        name,
        "--single-checkpoint",
        checkpoint,
        "--samples",
        str(args.samples),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--tp-size",
        str(args.tp_size),
    ]
    if args.gpu_ids:
        cmd.extend(["--gpu-ids", args.gpu_ids])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        return {
            "name": name,
            "checkpoint": checkpoint,
            "error": "evaluator_failed",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    result = json.loads(lines[-1])
    result["evaluator_stderr_tail"] = proc.stderr[-4000:]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AIME pass@8 evaluations.")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-ids", default="", help="Optional EVAL_GPU_IDS override, e.g. 4,5,6,7")
    parser.add_argument("--out", default="experiments/human_guidance/pass8_results.json")
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("NAME", "PATH"))
    parser.add_argument("--single-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--single-checkpoint", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single_name:
        print(json.dumps(run_single_in_child(args.single_name, args.single_checkpoint, args), ensure_ascii=False))
        return 0
    results = []
    for name, checkpoint in args.checkpoint or DEFAULT_CHECKPOINTS:
        print(f"[pass8] evaluating {name}: {checkpoint}", file=sys.stderr, flush=True)
        result = run_one(name, checkpoint, args)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    payload = {
        "samples": args.samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "results": results,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[pass8] wrote {out_path}", file=sys.stderr)
    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
