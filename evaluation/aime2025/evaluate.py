"""
AIME Evaluator — load a checkpoint, generate on AIME 2025, score accuracy.

Usage:
    python3 experiments/evaluator/aime/evaluate.py --checkpoint_path <path>

The checkpoint_path can be:
  - A HuggingFace model directory (contains config.json + weights)
  - A verl FSDP checkpoint dir like global_step_N/ (auto-resolves actor/huggingface/)
  - A HuggingFace model ID (e.g. "Qwen/Qwen3-1.7B-Base")

Scoring: accuracy — generates 1 response per problem; score = fraction correct.

Output (stdout JSON):
{
  "aime_2025": {"accuracy": 0.233, "correct": 7, "total": 30},
  "overall":   {"accuracy": 0.233, "correct": 7, "total": 30}
}
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATOR_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(EVALUATOR_ROOT, "..", ".."))
VERL_ROOT = os.path.join(REPO_ROOT, "training")

sys.path.insert(0, VERL_ROOT)

DATA_PATH = os.path.join(REPO_ROOT, "benchmark", "aime", "test.parquet")
LOG_DIR = os.path.join(EVALUATOR_ROOT, "logs")
MERGE_CACHE_DIR = os.path.join(EVALUATOR_ROOT, ".merge_cache")

YEAR_KEYS = ["aime_2025"]

EVAL_MAX_MODEL_LEN = int(os.environ.get("EVAL_MAX_MODEL_LEN", "32768"))
EVAL_MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "32000"))
EVAL_TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0.6"))
EVAL_TOP_P = float(os.environ.get("EVAL_TOP_P", "0.95"))
EVAL_N_SAMPLES = int(os.environ.get("EVAL_N_SAMPLES", "1"))
EVAL_MIN_FREE_MIB = int(os.environ.get("EVAL_MIN_FREE_MIB", "20000"))

_THINKING_RE = re.compile(
    r"<think>[\s\S]*?</think>", re.IGNORECASE
)


def _hf_dir_has_weights(path: str) -> bool:
    """Check if a HuggingFace directory contains actual model weights."""
    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "config.json")):
        return False
    for f in os.listdir(path):
        if f.endswith((".safetensors", ".bin")):
            return True
    return False


def _merge_fsdp_shards(actor_dir: str, output_dir: str) -> str:
    """Merge FSDP sharded checkpoint into a single HF model directory.

    Handles DTensor shards saved by verl's FSDP checkpoint manager.
    All shards use Shard(dim=0), so we cat _local_tensor along dim 0.
    """
    import torch

    shard_pattern = os.path.join(actor_dir, "model_world_size_*_rank_*.pt")
    shard_files = sorted(glob.glob(shard_pattern))
    if not shard_files:
        raise FileNotFoundError(f"No FSDP shards found at {shard_pattern}")

    fsdp_config_path = os.path.join(actor_dir, "fsdp_config.json")
    with open(fsdp_config_path) as f:
        fsdp_cfg = json.load(f)
    world_size = fsdp_cfg["world_size"]
    print(f"[evaluator] Merging {world_size} FSDP shards from {actor_dir}", file=sys.stderr)

    shards = []
    for rank in range(world_size):
        shard_file = os.path.join(actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        shards.append(torch.load(shard_file, map_location="cpu", weights_only=False))

    merged_state_dict = {}
    for key in shards[0].keys():
        tensors = []
        for rank in range(world_size):
            t = shards[rank][key]
            if hasattr(t, "_local_tensor"):
                tensors.append(t._local_tensor)
            else:
                tensors.append(t)

        if len(tensors[0].shape) == 0:
            merged_state_dict[key] = tensors[0]
        else:
            merged_state_dict[key] = torch.cat(tensors, dim=0)

    del shards

    hf_src = os.path.join(actor_dir, "huggingface")
    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(hf_src):
        src = os.path.join(hf_src, f)
        dst = os.path.join(output_dir, f)
        if os.path.isfile(src):
            import shutil
            shutil.copy2(src, dst)

    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)

    merged_bf16 = {k: v.to(torch.bfloat16) for k, v in merged_state_dict.items()}
    del merged_state_dict
    model.save_pretrained(output_dir, state_dict=merged_bf16)
    del merged_bf16
    print(f"[evaluator] Merged HF model saved to {output_dir}", file=sys.stderr)
    return output_dir


def resolve_checkpoint_path(path: str) -> str:
    """Resolve checkpoint to a HF directory with weights, merging FSDP shards if needed."""
    if not os.path.isdir(path):
        return path

    if _hf_dir_has_weights(path):
        return path

    hf_sub = os.path.join(path, "actor", "huggingface")
    if _hf_dir_has_weights(hf_sub):
        return hf_sub

    for candidate in ["huggingface", "actor/huggingface"]:
        full = os.path.join(path, candidate)
        if _hf_dir_has_weights(full):
            return full

    actor_dir = os.path.join(path, "actor") if os.path.isdir(os.path.join(path, "actor")) else path
    fsdp_config = os.path.join(actor_dir, "fsdp_config.json")
    shard_pattern = os.path.join(actor_dir, "model_world_size_*_rank_0.pt")
    if os.path.isfile(fsdp_config) and glob.glob(shard_pattern):
        cache_key = os.path.basename(os.path.normpath(path))
        output_dir = os.path.join(MERGE_CACHE_DIR, cache_key)
        if _hf_dir_has_weights(output_dir):
            print(f"[evaluator] Using cached merged checkpoint: {output_dir}", file=sys.stderr)
            return output_dir
        return _merge_fsdp_shards(actor_dir, output_dir)

    if os.path.isfile(os.path.join(path, "config.json")):
        return path

    return path


def load_test_data():
    """Load AIME 2025 problems from benchmark/aime/test.parquet."""
    if not os.path.isfile(DATA_PATH):
        print(f"[evaluator] ERROR: Test data not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    import pandas as pd
    df = pd.read_parquet(DATA_PATH)
    df = df[df["data_source"] == "aime_2025"].reset_index(drop=True)
    if df.empty:
        print("[evaluator] ERROR: No aime_2025 problems found in test data", file=sys.stderr)
        sys.exit(1)
    return df


def generate_responses(model_path: str, prompts: list[list[dict]], tp_size: int = 1, n_samples: int = 1) -> list[list[str]]:
    """Generate responses for all prompts using vLLM.
    
    Returns a list of lists: for each prompt, n_samples responses.
    """
    from vllm import LLM, SamplingParams

    gpu_mem = float(os.environ.get("EVAL_GPU_MEM_UTIL", "0.9"))

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
        max_model_len=EVAL_MAX_MODEL_LEN,
    )

    tokenizer = llm.get_tokenizer()

    formatted_prompts = []
    for messages in prompts:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = messages[0]["content"]
        formatted_prompts.append(text)

    sampling_params = SamplingParams(
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        max_tokens=EVAL_MAX_TOKENS,
        n=n_samples,
    )

    outputs = llm.generate(formatted_prompts, sampling_params)
    all_responses = []
    for output in outputs:
        all_responses.append([o.text for o in output.outputs])
    return all_responses


def strip_thinking_blocks(text: str) -> str:
    """Remove Qwen3 thinking blocks before scoring (evaluator-only preprocessing)."""
    return _THINKING_RE.sub("", text)


def score_response(solution_str: str, ground_truth: str) -> dict:
    """Score a single response using math_dapo scorer."""
    from verl.utils.reward_score.math_dapo import compute_score
    solution_str = strip_thinking_blocks(solution_str)
    return compute_score(solution_str, ground_truth)


def aggregate_results(records: list[dict]) -> dict:
    """Aggregate per-problem results into accuracy scores."""
    year_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in records:
        ds = r["data_source"]
        for acc in r["per_sample_acc"]:
            year_stats[ds]["total"] += 1
            if acc:
                year_stats[ds]["correct"] += 1

    results = {}
    all_correct = 0
    all_total = 0
    for year in YEAR_KEYS:
        info = year_stats.get(year, {"correct": 0, "total": 0})
        rate = info["correct"] / info["total"] if info["total"] else 0.0
        results[year] = {
            "accuracy": round(rate, 4),
            "correct": info["correct"],
            "total": info["total"],
        }
        all_correct += info["correct"]
        all_total += info["total"]

    for ds in year_stats:
        if ds not in YEAR_KEYS:
            info = year_stats[ds]
            rate = info["correct"] / info["total"] if info["total"] else 0.0
            results[ds] = {
                "accuracy": round(rate, 4),
                "correct": info["correct"],
                "total": info["total"],
            }
            all_correct += info["correct"]
            all_total += info["total"]

    overall_rate = all_correct / all_total if all_total else 0.0
    results["overall"] = {
        "accuracy": round(overall_rate, 4),
        "correct": all_correct,
        "total": all_total,
    }
    return results


def _query_gpu_free_mib() -> list[tuple[int, int]]:
    """Return (gpu_index, free_mib) for all visible GPUs, sorted by free memory descending."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[evaluator] nvidia-smi failed: {e}", file=sys.stderr)
        return []

    gpus: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpus.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    gpus.sort(key=lambda x: x[1], reverse=True)
    return gpus


def select_eval_gpus(tp_size: int, min_free_mib=None) -> str:
    """Pick GPUs with enough free memory for vLLM."""
    override = (os.environ.get("EVAL_GPU_IDS") or "").strip()
    if override:
        print(f"[evaluator] Using EVAL_GPU_IDS={override}", file=sys.stderr)
        return override

    need = max(1, tp_size)
    threshold = min_free_mib if min_free_mib is not None else EVAL_MIN_FREE_MIB
    candidates = [(i, free) for i, free in _query_gpu_free_mib() if free >= threshold]

    if len(candidates) < need:
        all_gpus = _query_gpu_free_mib()
        detail = ", ".join(f"{i}:{free}MiB" for i, free in all_gpus) or "none"
        raise RuntimeError(
            f"Need {need} GPU(s) with >={threshold} MiB free for eval (tp={tp_size}), "
            f"found {len(candidates)}. All GPUs: {detail}"
        )

    picked = [str(i) for i, _ in candidates[:need]]
    picked_detail = ", ".join(f"{i}({free}MiB free)" for i, free in candidates[:need])
    print(
        f"[evaluator] Auto-selected CUDA_VISIBLE_DEVICES={','.join(picked)} ({picked_detail})",
        file=sys.stderr,
    )
    return ",".join(picked)


def _dir_is_writable(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    probe = os.path.join(path, f".eval_write_probe_{os.getpid()}")
    try:
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
        return True
    except OSError:
        return False


def _log_output_dir(checkpoint_path: str) -> tuple[str, str]:
    abs_path = os.path.abspath(checkpoint_path)
    name = os.path.basename(abs_path.rstrip(os.sep)) or "model"
    if _dir_is_writable(abs_path):
        return abs_path, name
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR, name


def save_log(checkpoint_path: str, records: list[dict], results: dict):
    """Save per-problem results under checkpoint dir when writable, else evaluator logs."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir, checkpoint_name = _log_output_dir(checkpoint_path)
    log_path = os.path.join(out_dir, f"eval_{checkpoint_name}_{ts}.jsonl")

    header = {
        "type": "meta",
        "checkpoint_path": checkpoint_path,
        "timestamp": ts,
        "results": results,
    }
    if out_dir == LOG_DIR:
        print(
            f"[evaluator] Checkpoint dir not writable; saving log to {log_path}",
            file=sys.stderr,
        )

    with open(log_path, "w") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[evaluator] Detailed log saved to {log_path}", file=sys.stderr)
    return log_path


def main():
    parser = argparse.ArgumentParser(description="AIME Evaluator (accuracy)")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint or HF model ID")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM")
    args = parser.parse_args()

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = select_eval_gpus(args.tp_size)

    model_path = resolve_checkpoint_path(args.checkpoint_path)
    print(f"[evaluator] Model path: {model_path}", file=sys.stderr)

    df = load_test_data()
    print(f"[evaluator] Loaded {len(df)} AIME 2025 problems", file=sys.stderr)
    print(f"[evaluator] Scoring mode: accuracy (n_samples={EVAL_N_SAMPLES})", file=sys.stderr)

    prompts = df["prompt"].tolist()
    prompts = [p.tolist() if hasattr(p, "tolist") else p for p in prompts]

    print(f"[evaluator] Generating {EVAL_N_SAMPLES} response(s) per problem with vLLM (tp={args.tp_size})...", file=sys.stderr)
    all_responses = generate_responses(model_path, prompts, tp_size=args.tp_size, n_samples=EVAL_N_SAMPLES)

    print(f"[evaluator] Scoring {len(all_responses)} problems × {EVAL_N_SAMPLES} sample(s)...", file=sys.stderr)
    records = []
    for i, (responses, row) in enumerate(zip(all_responses, df.itertuples())):
        gt = row.reward_model["ground_truth"]
        per_sample_acc = []
        sample_details = []
        for resp in responses:
            result = score_response(resp, gt)
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

    results = aggregate_results(records)
    save_log(args.checkpoint_path, records, results)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
