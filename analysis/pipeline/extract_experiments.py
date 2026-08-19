"""Extract candidate training/evaluation actions without inferring intent."""

from __future__ import annotations

import re
from typing import Any, Iterable


TRAIN_RE = re.compile(r"(?:train|finetun|fine[-_ ]?tune|sft|grpo|ppo|dpo|rlhf|preference|accelerate\s+launch|torchrun|deepspeed)", re.I)
EVAL_RE = re.compile(r"(?:evaluate|evaluation|eval\.py|run_eval|score|lm_eval)", re.I)
CHECKPOINT_RE = re.compile(r"(?:checkpoint|save_pretrained|save_model|merge_model|model_path|final_model)", re.I)
SCRIPT_RE = re.compile(r"(?:train|sft|grpo|dpo|rl|finetun|fine[-_ ]?tune|ppo)", re.I)


def _classify(command: str, script_path: str | None, event_type: str) -> dict[str, Any]:
    source = " ".join(x for x in (command, script_path) if x)
    launch = (not command) or bool(re.search(r"\b(?:python(?:\d*(?:\.\d+)?)?|accelerate|torchrun|deepspeed|bash|sh)\b|(?:^|\s)\./", command, re.I))
    training = bool(TRAIN_RE.search(source)) and launch
    evaluation = bool(EVAL_RE.search(source)) and launch and not training
    checkpoint = bool(CHECKPOINT_RE.search(source)) and launch
    lower = source.lower()
    if re.search(r"\b(grpo|ppo|rlhf|reinforce|reward|rl)\b", lower):
        method = "rl"
    elif re.search(r"\b(dpo|preference|orpo)\b", lower):
        method = "preference_optimization"
    elif re.search(r"\b(distill|distillation)\b", lower):
        method = "distillation"
    elif re.search(r"\b(sft|supervised|fine[-_ ]?tune|finetune)\b|(?:^|[ /_-])train(?:\.py)?(?:$|[ /_-])", lower):
        method = "sft"
    elif evaluation:
        method = "evaluation"
    elif training:
        method = "other_training"
    else:
        method = "other"
    if re.search(r"\b(lora|qlora|peft|adapter)\b", lower):
        update = "LoRA/PEFT"
    elif training:
        update = "full_or_unspecified"
    else:
        update = None
    if re.search(r"synthetic|self[-_ ]?generated|generate.*data", lower):
        data = "synthetic_or_self_generated"
    elif re.search(r"preference|chosen|rejected", lower):
        data = "preference_pairs"
    elif re.search(r"benchmark|gsm8k|humaneval|aime|gpqa|bfcl", lower):
        data = "benchmark_or_task"
    elif training:
        data = "unspecified"
    else:
        data = None
    if re.search(r"reward|grpo|ppo|rlhf|exact[_ -]?match|verifier", lower):
        objective = "reward_or_verifier"
    elif re.search(r"dpo|preference|chosen|rejected", lower):
        objective = "preference_objective"
    elif training:
        objective = "language_modeling_or_unspecified"
    else:
        objective = None
    parent_match = re.search(r"(?:resume[_ -]?from[_ -]?checkpoint|checkpoint[_ -]?path)\s*[= ]\s*([^\s]+)", source, re.I)
    return {
        "training_launched": training and event_type in {"command_execution", "tool_call"},
        "evaluation_launched": evaluation and event_type in {"command_execution", "tool_call"},
        "checkpoint_created": checkpoint and event_type in {"command_execution", "tool_call"},
        "checkpoint_parent": parent_match.group(1) if parent_match else None,
        "method_family": method,
        "update_mechanism": update,
        "data_regime": data,
        "objective_or_reward": objective,
    }


def extract_experiments(trajectory_id: str, events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for event in events:
        command = event.get("command")
        script_path = event.get("file_path")
        event_type = str(event.get("event_type") or "")
        tool = str(event.get("tool_name") or "")
        text = str(event.get("text") or "")
        if not command and event_type == "tool_call" and tool.lower() in {"bash", "shell", "command"}:
            command = text
        command = str(command) if command else None
        script_path = str(script_path) if script_path else None
        is_script_write = bool(script_path and SCRIPT_RE.search(script_path) and tool.lower() in {"write", "edit", "notebookedit"})
        candidate = bool(command and ((TRAIN_RE.search(command) or EVAL_RE.search(command) or CHECKPOINT_RE.search(command))
                                      and bool(re.search(r"\b(?:python(?:\d*(?:\.\d+)?)?|accelerate|torchrun|deepspeed|bash|sh)\b|(?:^|\s)\./", command, re.I)))) or is_script_write
        if not candidate:
            continue
        kind = "proposed" if is_script_write and not command else "executed"
        key = (kind, command or "", script_path or "")
        if key in seen:
            continue
        seen.add(key)
        flags = _classify(command or script_path or text, script_path, event_type if kind == "executed" else "proposal")
        rows.append({
            "trajectory_id": trajectory_id,
            "experiment_index": len(rows),
            "executed_or_proposed": kind,
            "command": command,
            "script_path": script_path,
            "training_launched": bool(flags["training_launched"] and kind == "executed"),
            "evaluation_launched": bool(flags["evaluation_launched"] and kind == "executed"),
            "checkpoint_created": bool(flags["checkpoint_created"] and kind == "executed"),
            "checkpoint_parent": flags["checkpoint_parent"],
            "method_family": flags["method_family"],
            "update_mechanism": flags["update_mechanism"],
            "data_regime": flags["data_regime"],
            "objective_or_reward": flags["objective_or_reward"],
            "local_config": command or text or script_path,
            "result_reference": event.get("command_output"),
            "extraction_confidence": event.get("parser_confidence", "medium"),
        })
    return rows
