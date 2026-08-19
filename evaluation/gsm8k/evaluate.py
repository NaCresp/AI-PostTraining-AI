"""
GSM8K Evaluator — load a checkpoint, generate on GSM8K test set, score accuracy.

Usage:
    python3 experiments/evaluator/gsm8k/evaluate.py --checkpoint_path <path>

The checkpoint_path can be:
  - A HuggingFace model directory (contains config.json + weights)
  - A verl FSDP checkpoint dir like global_step_N/ (auto-resolves actor/huggingface/)
  - A HuggingFace model ID (e.g. "Qwen/Qwen3-1.7B-Base")

Scoring: accuracy — generates 1 response per problem; extracts "#### <number>".

Output (stdout JSON):
{
  "gsm8k": {"accuracy": 0.45, "correct": 593, "total": 1319},
  "overall": {"accuracy": 0.45, "correct": 593, "total": 1319}
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

DATA_PATH = os.path.join(REPO_ROOT, "benchmark", "gsm8k", "test.parquet")
LOG_DIR = os.path.join(EVALUATOR_ROOT, "logs")
MERGE_CACHE_DIR = os.path.join(EVALUATOR_ROOT, ".merge_cache")

GROUP_KEYS = ["gsm8k"]

EVAL_MAX_MODEL_LEN = int(os.environ.get("EVAL_MAX_MODEL_LEN", "8192"))
EVAL_MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "2048"))
EVAL_TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0.6"))
EVAL_TOP_P = float(os.environ.get("EVAL_TOP_P", "0.95"))
EVAL_N_SAMPLES = int(os.environ.get("EVAL_N_SAMPLES", "1"))
EVAL_MIN_FREE_MIB = int(os.environ.get("EVAL_MIN_FREE_MIB", "20000"))

_THINKING_RE = re.compile(
    r"<think>[\s\S]*?</think>", re.IGNORECASE
)

_SOLUTION_CLIP_CHARS = 8192


# ---------------------------------------------------------------------------
# Checkpoint resolution (shared across all evaluators)
# ---------------------------------------------------------------------------

def _hf_dir_has_weights(path: str) -> bool:
    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "config.json")):
        return False
    for f in os.listdir(path):
        if f.endswith((".safetensors", ".bin")):
            return True
    return False


def _merge_fsdp_shards(actor_dir: str, output_dir: str) -> str:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data():
    """Load GSM8K problems from benchmark/gsm8k/test.parquet."""
    if not os.path.isfile(DATA_PATH):
        print(f"[evaluator] ERROR: Test data not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    import pandas as pd
    df = pd.read_parquet(DATA_PATH)
    df = df[df["data_source"] == "gsm8k"].reset_index(drop=True)
    if df.empty:
        print("[evaluator] ERROR: No gsm8k problems found in test data", file=sys.stderr)
        sys.exit(1)
    return df


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_responses(model_path: str, prompts: list[list[dict]], tp_size: int = 1, n_samples: int = 1) -> list[list[str]]:
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


# ---------------------------------------------------------------------------
# GSM8K scoring
# ---------------------------------------------------------------------------

def strip_thinking_blocks(text: str) -> str:
    return _THINKING_RE.sub("", text)


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    """Extract numerical answer from #### pattern."""
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if not solutions:
            return None
        return solutions[-1].replace(",", "").replace("$", "")
    else:
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        for final_answer in reversed(answer):
            if final_answer not in ("", "."):
                return final_answer
        return None


def _norm_num(s) -> str | None:
    if s is None:
        return None
    return str(s).strip().replace(",", "").replace("$", "")


def score_response(solution_str: str, ground_truth: str) -> dict:
    """Score a single GSM8K response by extracting #### answer."""
    solution_str = strip_thinking_blocks(solution_str)
    pred = extract_solution(solution_str, method="strict")
    gt_n = _norm_num(ground_truth)
    pred_n = _norm_num(pred)

    if pred_n is not None and gt_n is not None and pred_n == gt_n:
        acc = 1
    else:
        acc = 0

    return {"acc": acc, "pred": pred if pred is not None else "[NO_ANSWER]"}


# ---------------------------------------------------------------------------
# Aggregation & logging
# ---------------------------------------------------------------------------

def aggregate_results(records: list[dict]) -> dict:
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in records:
        ds = r["data_source"]
        for acc in r["per_sample_acc"]:
            group_stats[ds]["total"] += 1
            if acc:
                group_stats[ds]["correct"] += 1

    results = {}
    all_correct = 0
    all_total = 0
    for key in GROUP_KEYS:
        info = group_stats.get(key, {"correct": 0, "total": 0})
        rate = info["correct"] / info["total"] if info["total"] else 0.0
        results[key] = {
            "accuracy": round(rate, 4),
            "correct": info["correct"],
            "total": info["total"],
        }
        all_correct += info["correct"]
        all_total += info["total"]

    for ds in group_stats:
        if ds not in GROUP_KEYS:
            info = group_stats[ds]
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
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[evaluator] nvidia-smi failed: {e}", file=sys.stderr)
        return []
    gpus: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                gpus.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    gpus.sort(key=lambda x: x[1], reverse=True)
    return gpus


def select_eval_gpus(tp_size: int, min_free_mib=None) -> str:
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
    print(f"[evaluator] Auto-selected CUDA_VISIBLE_DEVICES={','.join(picked)} ({picked_detail})", file=sys.stderr)
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
        print(f"[evaluator] Checkpoint dir not writable; saving log to {log_path}", file=sys.stderr)

    with open(log_path, "w") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[evaluator] Detailed log saved to {log_path}", file=sys.stderr)
    return log_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GSM8K Evaluator (accuracy)")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint or HF model ID")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM")
    args = parser.parse_args()

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = select_eval_gpus(args.tp_size)

    model_path = resolve_checkpoint_path(args.checkpoint_path)
    print(f"[evaluator] Model path: {model_path}", file=sys.stderr)

    df = load_test_data()
    print(f"[evaluator] Loaded {len(df)} GSM8K problems", file=sys.stderr)
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
