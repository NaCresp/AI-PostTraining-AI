import unittest

from analysis.pipeline.filter_cohorts import cohort_membership, finalize_metadata
from analysis.pipeline.extract_judgements import _status


class FilterTests(unittest.TestCase):
    def test_layered_cohorts(self):
        row = {"trajectory_id": "t", "batch_id": "b", "task_dir": "x", "scaffold_family": "codex",
               "benchmark": "GSM8K", "base_model": "Qwen3-1.7B-Base", "workspace_status": "present",
               "inventory_reasons": []}
        metadata = finalize_metadata(row, "codex_jsonl", "high", "complete", "valid",
                                     {"contamination_status": "clean", "disallowed_model_status": "clean"},
                                     "complete", [{"event_type": "result"}])
        self.assertEqual(cohort_membership(metadata), ["all_inventory", "behavior_cohort", "outcome_cohort", "strict_clean_cohort", "matched_cohort"])

    def test_missing_judgement_not_clean(self):
        row = {"trajectory_id": "t", "batch_id": "b", "task_dir": "x", "scaffold_family": "codex",
               "benchmark": "GSM8K", "base_model": "Qwen3-1.7B-Base", "workspace_status": "present",
               "inventory_reasons": []}
        metadata = finalize_metadata(row, "unparsed", "low", "unavailable", "missing",
                                     {"contamination_status": "missing", "disallowed_model_status": "missing"},
                                     "unknown", [])
        self.assertNotIn("strict_clean_cohort", cohort_membership(metadata))
        self.assertIn("missing_metrics", metadata["filter_reasons"])

    def test_unproduced_judgement_is_unknown(self):
        self.assertEqual(_status("No contamination_judgement.txt produced.", "contamination"), "unknown")
        self.assertEqual(_status("no contamination detected", "contamination"), "clean")
        self.assertEqual(_status("only allowed use detected", "disallowed"), "clean")


if __name__ == "__main__":
    unittest.main()
