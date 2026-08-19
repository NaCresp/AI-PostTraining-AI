import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_pilot import (action_category, classify_records, extract_json_line,
                       agent_from_batch, classify_objective, normalize_codex, normalize_opencode,
                       objective_context, objective_transitions)


class ParserTests(unittest.TestCase):
    def test_agent_model_from_batch(self):
        self.assertEqual(
            agent_from_batch("opencode_anthropic_claude-opus-4-5_10h"),
            "Claude Opus 4.5",
        )

    def test_warning_prefix_json(self):
        self.assertIsNone(extract_json_line("WARNING: cuda"))
        self.assertEqual(extract_json_line("prefix {\"type\": \"result\"}"), {"type": "result"})

    def test_codex_lifecycle_is_structured(self):
        records = [
            {"type": "item.started", "item": {"id": "i1", "type": "command_execution", "command": "python train.py", "status": "in_progress"}},
            {"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": "python train.py", "aggregated_output": "done", "exit_code": 0, "status": "completed"}},
            {"type": "turn.completed"},
        ]
        self.assertEqual(classify_records(records), "codex_jsonl")
        events = normalize_codex(records, "tid")
        self.assertEqual(sum(e["event_type"] == "command_execution" for e in events), 2)
        self.assertTrue(any(e["event_type"] == "result" for e in events))

    def test_opencode_tool_schema(self):
        records = [{"type": "tool_use", "part": {"id": "p1", "callID": "c1", "type": "tool", "tool": "bash", "state": {"status": "completed", "input": {"command": "python evaluate.py"}, "output": "accuracy: 0.5", "exit": 0}}}]
        self.assertEqual(classify_records(records), "opencode_jsonl")
        events = normalize_opencode(records, "tid")
        self.assertEqual(events[0]["command"], "python evaluate.py")
        self.assertEqual(events[0]["exit_code"], 0)

    def test_opencode_file_content_is_available_for_objective_annotation(self):
        records = [
            {"type": "tool_use", "part": {"id": "p1", "callID": "c1", "type": "tool", "tool": "write", "state": {
                "status": "completed", "input": {"filePath": "train_sft.py", "content": "from trl import SFTTrainer\ntrainer=SFTTrainer(...)"}, "exit": 0}}},
            {"type": "tool_use", "part": {"id": "p2", "callID": "c2", "type": "tool", "tool": "bash", "state": {
                "status": "completed", "input": {"command": "python train_sft.py"}, "exit": 0}}},
        ]
        events = normalize_opencode(records, "tid")
        self.assertIn("SFTTrainer", events[0]["file_content"])
        context = objective_context(events, {"command": "python train_sft.py", "event_index": 1})
        self.assertEqual(classify_objective("python train_sft.py", context)[0], "supervised_likelihood")


class ExtractionTests(unittest.TestCase):
    def test_install_and_dataset_inspection_are_not_training(self):
        self.assertEqual(action_category("python -m pip install trl peft accelerate")[0], "environment_setup")
        self.assertNotEqual(action_category("python -c \"from datasets import load_dataset; ds=load_dataset('x')\"")[0], "training")
        self.assertEqual(action_category("python train_sft.py --epochs 1")[0], "training")
        self.assertEqual(action_category("python scripts/merge_lora.py --adapter a")[0], "checkpoint")
        self.assertNotEqual(action_category("pkill -f 'python train_grpo.py'")[0], "training")
        self.assertNotEqual(action_category("/bin/bash -lc \"pkill -f 'python train_grpo.py'\"")[0], "training")
        self.assertNotEqual(action_category("python build_sft_dataset.py")[0], "training")
        self.assertNotEqual(action_category("python test_train.py")[0], "training")

    def test_script_write_is_not_execution(self):
        self.assertNotEqual(action_category("cat > train.py <<'PY'\npython train.py\nPY")[0], "training")

    def test_objective_forms_ignore_parameterization_and_data(self):
        full = classify_objective("python train_sft.py", "from trl import SFTTrainer\ntrainer=SFTTrainer(...)")
        lora = classify_objective("python train_sft.py --use-lora --data dpo_focus.jsonl", "from trl import SFTTrainer\ntrainer=SFTTrainer(...)")
        self.assertEqual(full[0], "supervised_likelihood")
        self.assertEqual(lora[0], "supervised_likelihood")
        self.assertEqual(full[1], lora[1])

    def test_objective_forms(self):
        self.assertEqual(classify_objective("python train_grpo.py", "from trl import GRPOTrainer\ntrainer=GRPOTrainer(...)")[0], "reward_optimization")
        self.assertEqual(classify_objective("python train_dpo.py", "from trl import DPOTrainer\ntrainer=DPOTrainer(...)")[0], "preference_optimization")
        self.assertEqual(classify_objective("python train_gkd.py", "teacher_logits=teacher(...)\nstudent_logits=student(...)\nloss=kl_div(student_logits, teacher_logits)")[0], "on_policy_distillation")
        self.assertEqual(classify_objective("python train.py", "")[0], "objective_unknown")
        self.assertEqual(classify_objective("python train_sft.py", "# reinforce stopping behavior\nfrom trl import SFTTrainer\ntrainer=SFTTrainer(...)")[0], "supervised_likelihood")

    def test_unknown_objective_pairs_are_excluded(self):
        episodes = [
            {"experiment_index": 0, "objective_form": "supervised_likelihood", "objective_signature": "supervised_likelihood", "normalized_progress": 0.0},
            {"experiment_index": 1, "objective_form": "objective_unknown", "objective_signature": "objective_unknown", "normalized_progress": 0.5},
            {"experiment_index": 2, "objective_form": "reward_optimization", "objective_signature": "reward_grpo", "normalized_progress": 1.0},
        ]
        self.assertEqual(objective_transitions(episodes, "tid"), [])

    def test_same_supervised_objective_is_not_a_change(self):
        episodes = [
            {"experiment_index": 0, "objective_form": "supervised_likelihood", "objective_signature": "supervised_likelihood", "normalized_progress": 0.0},
            {"experiment_index": 1, "objective_form": "supervised_likelihood", "objective_signature": "supervised_likelihood", "normalized_progress": 1.0},
        ]
        rows = objective_transitions(episodes, "tid")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transition_type"], "same_objective")


class CorpusOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prefer the released annotations; fall back to a locally regenerated run.
        local = Path(__file__).resolve().parents[1]
        released = local.parent / "annotations" / "objective_level"
        cls.root = released if (released / "tables/trajectory_analysis.csv").exists() else local

    def test_full_inventory_and_28_cells(self):
        with (self.root / "tables/trajectory_analysis.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1338)
        cells = {(r["benchmark"], r["base_model"]) for r in rows}
        self.assertEqual(len(cells), 28)

    def test_no_hash_deduplication(self):
        with (self.root / "tables/trajectory_analysis.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        dup = [r for r in rows if int(r.get("duplicate_group_size") or 1) > 1]
        self.assertGreater(len(dup), 0)
        self.assertEqual(len(rows), len({r["trajectory_id"] for r in rows}))

    def test_evaluation_points_have_trace_reference(self):
        path = self.root / "intermediate/evaluation_points.jsonl"
        if not path.exists():
            self.skipTest(
                "intermediate/ is not redistributed; regenerate it with run_pilot.py"
            )
        with path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertGreater(len(rows), 0)
        self.assertTrue(all(r.get("source_reference") for r in rows))


if __name__ == "__main__":
    unittest.main()
