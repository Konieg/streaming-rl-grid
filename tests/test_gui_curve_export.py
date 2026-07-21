import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from matplotlib.figure import Figure

from stream_rl_grid.gui import TrainingPanel


class FakeStatusVariable:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


class GuiCurveExportTests(unittest.TestCase):
    def test_both_curve_panels_are_saved_with_timestamp(self):
        with tempfile.TemporaryDirectory() as folder:
            panel = TrainingPanel.__new__(TrainingPanel)
            panel.base_dir = Path(folder)
            panel.last_snapshot = {"step": 123}
            panel.status_var = FakeStatusVariable()
            panel.trainer = SimpleNamespace(
                run_id="test-run",
                config=SimpleNamespace(training=SimpleNamespace(log_dir="runs", metric_window=1000)),
            )
            panel.last_snapshot.update(
                {
                    "algorithm": "tidbd",
                    "curves": {
                        "steps": [10, 20],
                        "average_reward": [-1.0, -0.5],
                        "reward_rate": [-0.9, -0.6],
                        "abs_td_error": [1.0, 0.8],
                        "alpha_mean": [0.1, 0.1],
                        "epsilon": [0.1, 0.1],
                        "td_error_magnitude": [float("nan"), float("nan")],
                        "goal_count_window": [1.0, 2.0],
                        "mean_inter_goal_time": [float("nan"), 8.0],
                        "invalid_action_rate": [0.2, 0.1],
                        "average_reward_estimation_bias": [0.1, -0.1],
                        "average_reward_estimation_error": [0.1, 0.1],
                    },
                    "adaptation_events": [
                        {
                            "event_step": 10,
                            "events": ["season:1"],
                            "status": "recovered",
                            "baseline_reward": -1.0,
                            "recovery_threshold": -1.1,
                            "end_step": 20,
                            "delay": 10,
                        }
                    ],
                }
            )
            panel.figure = Figure(figsize=(6, 2))
            panel.figure.add_subplot(121).plot([0, 1], [0.0, 1.0])
            panel.figure.add_subplot(122).plot([0, 1], [1.0, 0.0])

            panel.save_curves()

            output = panel.base_dir / "runs" / "test-run" / "figures"
            files = list(output.glob("learning_curves_*.png"))
            self.assertEqual(len(files), 1)
            self.assertRegex(
                files[0].name,
                re.compile(r"^learning_curves_\d{8}-\d{6}-\d{6}\.png$"),
            )
            contents = files[0].read_bytes()
            self.assertIn(b"Creation Time", contents)
            self.assertIn(b"Saved at", contents)
            self.assertEqual(panel.figure.texts, [])
            self.assertEqual(len(list(output.glob("performance_metrics_*.png"))), 1)
            self.assertEqual(len(list(output.glob("performance_metrics_*.csv"))), 1)
            self.assertEqual(len(list(output.glob("adaptation_events_*.csv"))), 1)
            self.assertEqual(len(list(output.glob("performance_summary_*.json"))), 1)
            self.assertIn(str(output), panel.status_var.value)


if __name__ == "__main__":
    unittest.main()
