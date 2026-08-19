#!/usr/bin/env python3
"""Broad-criterion strategy-change measurement.

Extends the objective-level (paradigm) transition analysis with the two
remaining dimensions of the paper's strategy definition (Preliminaries §2.2):

  1. data-source type: curated vs. self-generated training data.  Judged only
     for supervised-likelihood and preference episodes; on-policy rollouts are
     inherent to reward optimization and are not an independent data-source
     dimension.  A pair contributes a data-source change only when BOTH sides
     are supervised/preference episodes with a recognized data-source label
     and the labels differ.
  2. stage structure: candidate stage additions/removals are flagged with
     evidence for author review (initialization source switches between the
     base model and an in-run checkpoint, or explicit warm-up/stage language
     around the launch).  Candidates are reported separately and are NOT
     added to the mechanical union count.

Outputs (analysis/broad_criterion/output/):
  episode_labels.jsonl        per-episode paradigm + data-source + evidence
  broad_transitions.csv       per-pair decomposition (paradigm/data/stage)
  stage_candidates_review.csv flagged stage-structure candidates + evidence
  summary.json                corpus / per-harness decomposition counts
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1] / "pilot_study"
OUT = HERE / "output"

SUPERVISED_FORMS = {"supervised_likelihood", "preference_optimization"}

# --- data-source evidence -------------------------------------------------
SELF_GEN = re.compile(
    r"(?:rejection[_\- ]?(?:sampl|filter)|best[_\- ]?of[_\- ]?n"
    r"|self[_\- ]?(?:instruct|gen|generated|distill)"
    r"|synthetic[_\- ]?(?:data|examples|dataset)"
    r"|model[_\- ]?generated|generated[_\- ]?(?:data|dataset|solutions|responses|samples|completions)"
    r"|teacher[_\- ]?(?:gen|generated|output|response|data)"
    r"|distill(?:ation)?[_\- ]?(?:data|dataset)"
    r"|rft[_\- ]?data|star[_\- ]?data"
    r"|sampled[_\- ]?(?:solutions|responses|completions|answers)"
    r"|correct[_\- ]?(?:samples|generations|solutions)\.(?:jsonl|json|parquet))",
    re.I,
)
GEN_INDICATOR = re.compile(
    r"(?:vllm|sglang|\.generate\s*\(|generate_(?:data|solutions|responses|samples)"
    r"|sample_(?:solutions|responses|completions)|do_sample\s*=\s*True|best_of)",
    re.I,
)
DATA_FILE = re.compile(r"[\w./-]{3,120}\.(?:jsonl|parquet|json|csv)\b", re.I)
NON_DATA_JSON = re.compile(
    r"(?:config|tokenizer|generation_config|adapter|special_tokens|vocab|merges"
    r"|metrics|results?|eval[\w-]*|manifest|package|settings)[\w.-]*\.json$",
    re.I,
)


def data_files(text: str) -> set[str]:
    return {f for f in DATA_FILE.findall(text) if not NON_DATA_JSON.search(f)}
CURATED = re.compile(
    r"(?:load_dataset\s*\(\s*['\"](?:openai/)?(?:gsm8k|mbpp|codesearchnet|humaneval"
    r"|open-?r1|openmathinstruct|metamath|math|nvidia|allenai|huggingfaceh4|tulu|ultrafeedback"
    r"|codeparrot|bigcode|sahil2801|iamtarun|nickrosh|evol|alpaca|dolly|oasst)"
    r"|--dataset[_\- ]?(?:name|path)?\s+\S*(?:gsm8k|mbpp|math|alpaca|codealpaca|evol|oasst)"
    r"|datasets\.load_dataset|hf[_\- ]?hub[_\- ]?download|huggingface\.co/datasets"
    r"|wget\s+\S+\.(?:jsonl|json|parquet|zip|tar)|snapshot_download)",
    re.I,
)

# --- stage-structure evidence --------------------------------------------
BASE_MODEL_PAT = re.compile(
    r"(?:qwen[\w./-]*(?:1\.7b|4b)[\w.-]*(?:base)?|smollm3[\w.-]*|gemma-3-4b[\w.-]*)",
    re.I,
)
CKPT_PATH = re.compile(
    r"(?:--model(?:_name_or_path|_path)?|--base_model|from_pretrained\s*\(|--resume_from_checkpoint|resume_from_checkpoint\s*=)"
    r"\s*[\(\'\"=]*\s*([\w./~-]+)",
    re.I,
)
CKPT_HINT = re.compile(r"(?:output|checkpoint|ckpt|merged|final|save|_v\d|stage|run\d|global_step|epoch)", re.I)
STAGE_KEYWORD = re.compile(r"(?:stage[_\- ]?\d|phase[_\- ]?\d|curriculum|two[_\- ]?stage|second[_\- ]?stage)", re.I)


def launched_script_names(command: str | None) -> list[str]:
    if not command:
        return []
    names = re.findall(
        r"(?:python(?:\d(?:\.\d+)?)?|bash|sh)\s+(?:-[a-z]+\s+)*([\w./-]+\.(?:py|sh))",
        command.lower(),
    )
    return [Path(n).name.lower() for n in names]


def stream_grouped(path: Path, key: str = "trajectory_id"):
    """Yield (trajectory_id, rows) from a jsonl file sorted by trajectory_id."""
    current, rows = None, []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row.get(key)
            if tid != current:
                if current is not None:
                    yield current, rows
                current, rows = tid, []
            rows.append(row)
    if current is not None:
        yield current, rows


def classify_init_source(text: str) -> tuple[str, str]:
    """Return (init_source, evidence) where init_source in {base, checkpoint, unknown}."""
    hits = CKPT_PATH.findall(text)
    for raw in hits:
        path = raw.strip().strip("'\")(,")
        if not path or path.startswith("--"):
            continue
        if BASE_MODEL_PAT.search(path) and not CKPT_HINT.search(path):
            return "base", path[:200]
        if CKPT_HINT.search(path):
            return "checkpoint", path[:200]
    if hits:
        return "unknown", str(hits[0])[:200]
    return "unknown", ""


def classify_data_source(objective_form: str, script_context: str, provenance: dict) -> tuple[str, str]:
    """Data-source label from the training-data pathway only.

    Self-generated evidence must come from (a) data-file names referenced by
    the training script/command that match generation patterns, or (b) a
    window command that both matches a generation indicator and mentions a
    data file consumed by the training script.  Bare vllm/eval sampling in the
    window is NOT evidence."""
    if objective_form == "reward_optimization":
        return "on_policy_inherent", "reward optimization trains on rollouts"
    if objective_form == "on_policy_distillation":
        return "on_policy_inherent", "distillation trains on student samples"
    if objective_form not in SUPERVISED_FORMS:
        return "not_applicable", ""
    train_files = data_files(script_context)
    self_evidence = ""
    for name in train_files:
        if SELF_GEN.search(name):
            self_evidence = f"training data file: {name}"
            break
    if not self_evidence and SELF_GEN.search(script_context):
        self_evidence = f"in training script: {SELF_GEN.search(script_context).group(0)}"
    prov_curated = ""
    for name in train_files:
        mark = provenance.get(Path(name).name.lower())
        if not mark:
            continue
        kind, ev = mark
        if kind == "self" and not self_evidence:
            self_evidence = f"file provenance: {name} <- {ev}"
        elif kind == "curated" and not prov_curated:
            prov_curated = f"file provenance: {name} <- {ev}"
    curated_hit = CURATED.search(script_context)
    if not curated_hit and prov_curated:
        curated_hit = None  # handled below via prov_curated
    curated_evidence = curated_hit.group(0)[:150] if curated_hit else prov_curated[:150]
    if self_evidence and curated_evidence:
        return "mixed", f"{self_evidence[:120]} | curated: {curated_evidence[:80]}"
    if self_evidence:
        return "self_generated", self_evidence[:200]
    if curated_evidence:
        return "curated", curated_evidence[:200]
    return "unknown", ""


def episode_context(events: list[dict], launch_index: int, prev_launch_index: int, command: str) -> tuple[str, str]:
    targets = set(launched_script_names(command))
    script_chunks: list[str] = [command or ""]
    window_chunks: list[str] = []
    for ev in events:
        idx = ev.get("event_index")
        if idx is None or idx > launch_index:
            continue
        source = ev.get("file_content") or ev.get("command") or ""
        if not source:
            continue
        name = Path(str(ev.get("file_path") or "")).name.lower()
        source_str = str(source)
        if (name and name in targets) or (targets and any(t in source_str.lower() for t in targets)):
            script_chunks.append(source_str)
        elif idx > prev_launch_index:
            window_chunks.append(source_str)
    return "\n".join(script_chunks[-12:])[-300000:], "\n".join(window_chunks[-40:])[-300000:]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes_by_tid: dict[str, list[dict]] = defaultdict(list)
    for row in map(json.loads, (PILOT / "intermediate/experiment_episodes.jsonl").open()):
        episodes_by_tid[row["trajectory_id"]].append(row)
    for eps in episodes_by_tid.values():
        eps.sort(key=lambda r: r["experiment_index"])

    harness_by_tid: dict[str, str] = {}
    with (PILOT / "tables/trajectory_analysis.csv").open() as handle:
        for row in csv.DictReader(handle):
            harness_by_tid[row["trajectory_id"]] = row.get("harness_family", "Unknown")

    labeled: dict[tuple[str, int], dict] = {}
    n_done = 0
    for tid, events in stream_grouped(PILOT / "intermediate/events_enriched.jsonl"):
        eps = episodes_by_tid.get(tid)
        if not eps:
            continue
        by_index = {ev.get("event_index"): ev for ev in events}
        # File provenance: walk events once, marking data files written/mentioned
        # by generation commands (self) or hub-download/convert commands (curated).
        prov_marks: list[tuple[int, str, str, str]] = []
        for ev in events:
            idx0 = ev.get("event_index")
            src0 = str(ev.get("file_content") or ev.get("command") or "")
            if not src0 or idx0 is None:
                continue
            files0 = data_files(src0)
            if not files0:
                continue
            if GEN_INDICATOR.search(src0) or SELF_GEN.search(src0):
                kind0, ev0 = "self", (GEN_INDICATOR.search(src0) or SELF_GEN.search(src0)).group(0)[:60]
            elif CURATED.search(src0):
                kind0, ev0 = "curated", CURATED.search(src0).group(0)[:60]
            else:
                continue
            for f0 in files0:
                prov_marks.append((idx0, Path(f0).name.lower(), kind0, ev0))
        prev_launch = -1
        for ep in eps:
            ref = str(ep.get("launch_reference") or ":")
            try:
                launch_index = int(ref.rsplit(":", 1)[1])
            except (ValueError, IndexError):
                launch_index = -1
            launch_event = by_index.get(launch_index, {})
            command = str(launch_event.get("command") or "")
            script_ctx, window_ctx = episode_context(events, launch_index, prev_launch, command)
            provenance = {}
            for idx0, fname, kind0, ev0 in prov_marks:
                if idx0 <= launch_index:
                    provenance[fname] = (kind0, ev0)  # latest mark before launch wins
            data_source, data_evidence = classify_data_source(ep.get("objective_form"), script_ctx, provenance)
            init_source, init_evidence = classify_init_source(script_ctx)
            stage_kw = STAGE_KEYWORD.search(script_ctx)
            labeled[(tid, ep["experiment_index"])] = {
                "trajectory_id": tid,
                "experiment_index": ep["experiment_index"],
                "harness_family": harness_by_tid.get(tid, "Unknown"),
                "objective_form": ep.get("objective_form"),
                "objective_signature": ep.get("objective_signature"),
                "data_source": data_source,
                "data_source_evidence": data_evidence,
                "init_source": init_source,
                "init_source_evidence": init_evidence,
                "stage_keyword": stage_kw.group(0)[:80] if stage_kw else "",
                "launch_reference": ep.get("launch_reference"),
            }
            prev_launch = launch_index
        n_done += 1
        if n_done % 200 == 0:
            print(f"labeled {n_done} trajectories", file=sys.stderr)

    with (OUT / "episode_labels.jsonl").open("w") as handle:
        for key in sorted(labeled):
            handle.write(json.dumps(labeled[key], ensure_ascii=True, sort_keys=True) + "\n")

    # Pair-level decomposition over the SAME 3,557-pair denominator.
    pair_rows: list[dict] = []
    stage_review: list[dict] = []
    with (PILOT / "intermediate/objective_transitions.csv").open() as handle:
        for row in csv.DictReader(handle):
            tid = row["trajectory_id"]
            left = labeled.get((tid, int(row["from_experiment"])))
            right = labeled.get((tid, int(row["to_experiment"])))
            if left is None or right is None:
                continue
            paradigm_change = row["transition_type"] == "objective_change"
            both_supervised = (
                left["objective_form"] in SUPERVISED_FORMS
                and right["objective_form"] in SUPERVISED_FORMS
            )
            data_known = (
                both_supervised
                and left["data_source"] in {"curated", "self_generated", "mixed"}
                and right["data_source"] in {"curated", "self_generated", "mixed"}
            )
            binary = lambda label: "self_involved" if label in {"self_generated", "mixed"} else label
            data_change = data_known and binary(left["data_source"]) != binary(right["data_source"])
            stage_candidate = (
                not paradigm_change
                and left["init_source"] != "unknown"
                and right["init_source"] != "unknown"
                and left["init_source"] != right["init_source"]
            )
            pair_rows.append({
                "trajectory_id": tid,
                "harness_family": left["harness_family"],
                "from_experiment": row["from_experiment"],
                "to_experiment": row["to_experiment"],
                "paradigm_change": int(paradigm_change),
                "both_supervised": int(both_supervised),
                "data_known": int(data_known),
                "data_change": int(data_change),
                "from_data_source": left["data_source"],
                "to_data_source": right["data_source"],
                "stage_candidate": int(stage_candidate),
                "broad_change_mechanical": int(paradigm_change or data_change),
            })
            if data_change or stage_candidate:
                stage_review.append({
                    "trajectory_id": tid,
                    "harness_family": left["harness_family"],
                    "pair": f"{row['from_experiment']}->{row['to_experiment']}",
                    "kind": "data_change" if data_change else "stage_candidate",
                    "from_data_source": left["data_source"],
                    "to_data_source": right["data_source"],
                    "from_init": left["init_source"],
                    "to_init": right["init_source"],
                    "stage_keyword": right["stage_keyword"],
                    "data_evidence_from": left["data_source_evidence"],
                    "data_evidence_to": right["data_source_evidence"],
                    "init_evidence_to": right["init_source_evidence"],
                    "launch_reference": right["launch_reference"],
                })

    fields = list(pair_rows[0].keys()) if pair_rows else []
    with (OUT / "broad_transitions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    if stage_review:
        with (OUT / "stage_candidates_review.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stage_review[0].keys()))
            writer.writeheader()
            writer.writerows(stage_review)

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        traj = lambda flag: len({r["trajectory_id"] for r in rows if r[flag]})
        return {
            "pairs": n,
            "paradigm_changes": sum(r["paradigm_change"] for r in rows),
            "supervised_pairs": sum(r["both_supervised"] for r in rows),
            "data_known_pairs": sum(r["data_known"] for r in rows),
            "data_changes": sum(r["data_change"] for r in rows),
            "stage_candidates": sum(r["stage_candidate"] for r in rows),
            "broad_changes_mechanical": sum(r["broad_change_mechanical"] for r in rows),
            "broad_change_rate": round(sum(r["broad_change_mechanical"] for r in rows) / n, 5) if n else None,
            "trajectories_with_paradigm_change": traj("paradigm_change"),
            "trajectories_with_broad_change": traj("broad_change_mechanical"),
        }

    summary = {"corpus": summarize(pair_rows)}
    for harness in sorted({r["harness_family"] for r in pair_rows}):
        summary[harness] = summarize([r for r in pair_rows if r["harness_family"] == harness])
    # episode-level label coverage for the supervised episodes
    sup = [r for r in labeled.values() if r["objective_form"] in SUPERVISED_FORMS]
    summary["episode_label_coverage"] = {
        "supervised_episodes": len(sup),
        "curated": sum(r["data_source"] == "curated" for r in sup),
        "self_generated": sum(r["data_source"] == "self_generated" for r in sup),
        "mixed": sum(r["data_source"] == "mixed" for r in sup),
        "unknown": sum(r["data_source"] == "unknown" for r in sup),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
