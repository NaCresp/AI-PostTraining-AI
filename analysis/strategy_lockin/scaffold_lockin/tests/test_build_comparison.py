import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "build_comparison.py"
SPEC = importlib.util.spec_from_file_location("build_comparison", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ScaffoldComparisonTest(unittest.TestCase):
    def test_summary_uses_identified_denominators_and_objective_switches(self):
        trajectories = [
            {
                "trajectory_id": "a",
                "training_experiment_count": "2",
                "initial_strategy_family": "full_sft",
                "valid_objective_pair_count": "1",
                "objective_change_count": "0",
                "final_metric": "0.5",
                "best_metric": "0.6",
                "base_model": "m1",
                "budget_hours": "10",
                "evaluation_count": "3",
            },
            {
                "trajectory_id": "b",
                "training_experiment_count": "3",
                "initial_strategy_family": "other_unknown",
                "valid_objective_pair_count": "2",
                "objective_change_count": "1",
                "final_metric": "",
                "best_metric": "",
                "base_model": "m1",
                "budget_hours": "10",
                "evaluation_count": "2",
            },
        ]
        episodes = {
            "a": [
                {
                    "executed": True,
                    "action_categories": ["training"],
                    "method_family": "full_sft",
                },
                {
                    "executed": True,
                    "action_categories": ["evaluation"],
                    "method_family": "peft_sft",
                },
            ],
            "b": [
                {
                    "executed": True,
                    "action_categories": ["training"],
                    "method_family": "other_unknown",
                }
            ],
        }

        result = MODULE.summarize_group(
            "Claude", "GSM8K", trajectories, episodes
        )

        self.assertEqual(result["n_trained_trajectories"], 2)
        self.assertEqual(result["n_initial_method_identified"], 1)
        self.assertEqual(result["dominant_initial_method_share_among_identified"], 1.0)
        self.assertEqual(result["verified_training_commands"], 5)
        self.assertEqual(result["n_training_commands_method_identified"], 1)
        self.assertEqual(result["objective_switches"], 1)
        self.assertEqual(result["valid_objective_pairs"], 3)
        self.assertEqual(result["mean_final_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
