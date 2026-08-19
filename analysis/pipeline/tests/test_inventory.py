import tempfile
import unittest
from pathlib import Path

from analysis.pipeline.inventory import scan_task_dirs


class InventoryTests(unittest.TestCase):
    def test_nested_task_is_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "codex_demo_10h_run1" / "gsm8k_Qwen_Qwen3-1.7B-Base_12345"
            (task / "task").mkdir(parents=True)
            (task / "solve_out.txt").write_text('{"type":"thread.started"}\n', encoding="utf-8")
            rows = scan_task_dirs(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["benchmark"], "GSM8K")
            self.assertEqual(rows[0]["base_model"], "Qwen3-1.7B-Base")


if __name__ == "__main__":
    unittest.main()
