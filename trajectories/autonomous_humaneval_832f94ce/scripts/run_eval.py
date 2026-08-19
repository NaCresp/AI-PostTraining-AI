#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--max-connections", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    cmd = [
        "python",
        "evaluate.py",
        "--model-path",
        args.model_path,
        "--limit",
        str(args.limit),
        "--max-tokens",
        str(args.max_tokens),
        "--max-connections",
        str(args.max_connections),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--json-output-file",
        args.out_json,
    ]
    subprocess.check_call(cmd)
    with open(args.out_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["_wall_seconds"] = time.time() - t0
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

