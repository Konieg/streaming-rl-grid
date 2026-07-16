import tempfile
import unittest
from pathlib import Path

from experiments.runner import ExperimentConfig, run_experiment


class FormalRunnerTests(unittest.TestCase):
    def test_supervised_runner_writes_reproducible_experiment_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "e1"
            path = run_experiment(
                ExperimentConfig(
                    experiment="e1",
                    steps=20,
                    seeds=(0, 1),
                    noise_features=2,
                    report_every=5,
                    output=str(output),
                )
            )
            self.assertEqual(path, output.resolve())
            for name in ("config.json", "seed-000.csv", "seed-001.csv", "aggregate.csv", "learning_curve.png"):
                self.assertTrue((path / name).is_file(), name)

    def test_td_tidbd_runner_completes_with_finite_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "e5"
            path = run_experiment(
                ExperimentConfig(
                    experiment="e5",
                    method="tidbd",
                    steps=25,
                    seeds=(0,),
                    report_every=5,
                    output=str(output),
                )
            )
            content = (path / "aggregate.csv").read_text(encoding="utf-8")
            self.assertIn("mean_window_loss", content)
            self.assertNotIn("nan", content.lower())


if __name__ == "__main__":
    unittest.main()
