import unittest

from analysis.pipeline.extract_experiments import extract_experiments


class ExperimentTests(unittest.TestCase):
    def test_proposed_is_not_executed(self):
        events = [
            {"event_type": "tool_call", "tool_name": "Write", "file_path": "train_rl.py", "text": "script", "parser_confidence": "high"},
            {"event_type": "command_execution", "tool_name": "command_execution", "command": "python train_rl.py --lr 1e-5", "command_output": "ok", "parser_confidence": "high"},
            {"event_type": "command_execution", "tool_name": "command_execution", "command": "python evaluate.py", "command_output": "0.4", "parser_confidence": "high"},
        ]
        rows = extract_experiments("t1", events)
        self.assertEqual(rows[0]["executed_or_proposed"], "proposed")
        self.assertFalse(rows[0]["training_launched"])
        self.assertEqual(rows[1]["executed_or_proposed"], "executed")
        self.assertTrue(rows[1]["training_launched"])
        self.assertTrue(rows[2]["evaluation_launched"])

    def test_local_change_is_preserved(self):
        rows = extract_experiments("t1", [{"event_type": "command_execution", "command": "python train.py --lr 1e-5 --epochs 2", "parser_confidence": "high"}])
        self.assertEqual(rows[0]["method_family"], "sft")
        self.assertIn("--lr", rows[0]["local_config"])


if __name__ == "__main__":
    unittest.main()
