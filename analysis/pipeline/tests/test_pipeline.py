import hashlib
import tempfile
import unittest
from pathlib import Path

from analysis.pipeline.run_pipeline import run_pipeline


class PipelineReproducibilityTests(unittest.TestCase):
    def test_fixture_rerun_has_identical_canonical_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "codex_demo_10h_run1" / "gsm8k_Qwen_Qwen3-1.7B-Base_12345"
            task.mkdir(parents=True)
            (task / "solve_out.txt").write_text(
                '{"type":"thread.started"}\n'
                '{"type":"item.completed","item":{"type":"command_execution","command":"python train.py","aggregated_output":"ok","exit_code":0,"status":"completed"}}\n'
                '{"type":"turn.completed"}\n', encoding="utf-8"
            )
            (task / "metrics.json").write_text('{"accuracy": 0.5, "stderr": 0.1}\n', encoding="utf-8")
            (task / "contamination_judgement.txt").write_text("no contamination detected\n", encoding="utf-8")
            (task / "disallowed_model_judgement.txt").write_text("only allowed use detected\n", encoding="utf-8")
            config = Path("analysis/pipeline/config.json").resolve()
            out1, out2 = root / "out1", root / "out2"
            run_pipeline(root, out1, config)
            run_pipeline(root, out2, config)
            names = ["trajectory_metadata.jsonl", "events.jsonl", "experiments.csv", "metrics.csv",
                     "judgements.csv", "cohort_counts.csv", "cohort_membership.csv", "validation_report.json"]
            for name in names:
                self.assertEqual(hashlib.sha256((out1 / name).read_bytes()).digest(),
                                 hashlib.sha256((out2 / name).read_bytes()).digest(), name)


if __name__ == "__main__":
    unittest.main()
