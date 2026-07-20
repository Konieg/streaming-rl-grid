import csv
import tempfile
import unittest
from pathlib import Path

from stream_rl_grid.metrics import MetricsTracker
from stream_rl_grid.environment import ContinualWindyGridWorld
from stream_rl_grid.phase1_plot import _event_steps, _selection_eligible
from stream_rl_grid.phase1_sweep import (
    METHOD_LABELS, _app_config, build_jobs, make_manifest, parameter_configurations,
)
from stream_rl_grid.dyna_q_plus_sweep import (
    make_manifest as make_dyna_plus_manifest,
    parameter_configurations as dyna_plus_parameter_configurations,
)
from stream_rl_grid.eight_algorithm_comparison import (
    METHOD_ORDER as EIGHT_METHODS,
    make_manifest as make_eight_algorithm_manifest,
)
from stream_rl_grid.eight_algorithm_tile_comparison import (
    make_manifest as make_tile_comparison_manifest,
)
from stream_rl_grid.features import create_feature_representation
from stream_rl_grid.tile_coding_sweep import (
    make_manifest as make_tile_sweep_manifest,
    parameter_configurations as tile_sweep_parameter_configurations,
)


class PhaseOneExperimentTests(unittest.TestCase):
    def test_parameter_grid_has_expected_63_configurations(self):
        configurations = parameter_configurations()
        self.assertEqual(len(configurations), 63)
        counts = {}
        for config in configurations:
            counts[config["method"]] = counts.get(config["method"], 0) + 1
        self.assertEqual(counts, {
            "sarsa": 3,
            "sarsa_lambda": 9,
            "tidbd": 3,
            "q_learning": 3,
            "q_lambda": 9,
            "dyna_q": 9,
            "dyna_q_lambda": 27,
        })

    def test_default_manifest_expands_to_945_jobs_and_d55(self):
        manifest = make_manifest(60_000, [0, 1, 2, 3, 4])
        self.assertEqual(manifest["expected_runs"], 945)
        self.assertEqual(len(build_jobs(manifest)), 945)
        self.assertEqual(manifest["feature_representation"], "handcrafted_lfa")
        self.assertEqual(_event_steps(manifest, "wind")[0], 5_500)
        self.assertEqual(_event_steps(manifest, "wind")[-1], 59_500)
        self.assertEqual(len(_event_steps(manifest, "wind")), 10)
        self.assertEqual(len(_event_steps(manifest, "goal")), 9)
        self.assertEqual(len(_event_steps(manifest, "obstacles")), 9)
        self.assertEqual(len(_event_steps(manifest, "reward")), 9)

    def test_dyna_q_plus_sweep_has_405_runs(self):
        source = make_manifest(60_000, [0, 1, 2, 3, 4])
        manifest = make_dyna_plus_manifest(source, Path("phase1_manifest.json"))
        self.assertEqual(len(dyna_plus_parameter_configurations()), 27)
        self.assertEqual(manifest["expected_runs"], 405)
        self.assertEqual(len(build_jobs(manifest)), 405)

    def test_all_algorithm_tile_sweep_has_1350_runs(self):
        source = make_manifest(60_000, [0, 1, 2, 3, 4])
        manifest = make_tile_sweep_manifest(source, Path("phase1_manifest.json"))
        configurations = tile_sweep_parameter_configurations()
        self.assertEqual(len(configurations), 90)
        self.assertEqual(manifest["expected_runs"], 1_350)
        self.assertEqual(len(build_jobs(manifest)), 1_350)
        self.assertEqual(manifest["feature_representation"], "tile_coding")
        self.assertEqual(
            {config["method"] for config in configurations}, set(EIGHT_METHODS)
        )
        counts = {}
        for config in configurations:
            counts[config["method"]] = counts.get(config["method"], 0) + 1
        self.assertEqual(counts["dyna_q_plus"], 27)
        first_config = _app_config(manifest, build_jobs(manifest)[0])
        coder = create_feature_representation(
            first_config.environment, first_config.agent
        )
        self.assertEqual(coder.size, 65_536)
        self.assertEqual(coder.nominal_active_count, 17)

    def test_final_comparison_uses_eight_winners_in_five_settings(self):
        source = make_manifest(60_000, [0, 1, 2, 3, 4])
        dyna_manifest = make_dyna_plus_manifest(source, Path("phase1_manifest.json"))
        old_winners = {}
        for config in source["parameter_configurations"]:
            old_winners.setdefault(config["method"], config)
        plus_winner = dyna_manifest["parameter_configurations"][0]
        with tempfile.TemporaryDirectory() as folder:
            old_selected = Path(folder) / "phase1_selected.csv"
            plus_selected = Path(folder) / "plus_selected.csv"
            with old_selected.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("setting", "method", "config_id")
                )
                writer.writeheader()
                for setting in ("transition_shift", "reward_shift", "combined"):
                    for method in METHOD_LABELS:
                        writer.writerow({
                            "setting": setting, "method": method,
                            "config_id": old_winners[method]["config_id"],
                        })
            with plus_selected.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("setting", "method", "config_id")
                )
                writer.writeheader()
                for setting in ("transition_shift", "reward_shift", "combined"):
                    writer.writerow({
                        "setting": setting, "method": "dyna_q_plus",
                        "config_id": plus_winner["config_id"],
                    })
            manifest = make_eight_algorithm_manifest(
                source, old_selected, dyna_manifest, plus_selected
            )
            tile_manifest = make_tile_comparison_manifest(
                source, old_selected, dyna_manifest, plus_selected
            )

        self.assertEqual(manifest["expected_runs"], 200)
        self.assertEqual(len(build_jobs(manifest)), 200)
        self.assertEqual(set(manifest["settings"]), {
            "wind_only", "goal_only", "obstacles_only", "reward_only", "combined",
        })
        self.assertEqual(tuple(manifest["method_order"]), EIGHT_METHODS)
        self.assertEqual(len(manifest["parameter_configurations"]), 40)
        self.assertEqual(
            {job["parameters"]["method"] for job in build_jobs(manifest)},
            set(EIGHT_METHODS),
        )
        for job in build_jobs(manifest):
            config = _app_config(manifest, job)
            config.validate()
            ContinualWindyGridWorld(config.environment)
            if job["parameters"]["method"] == "dyna_q_plus":
                self.assertEqual(config.agent.algorithm, "dyna_q_plus")
                self.assertEqual(
                    config.agent.dyna_plus_kappa,
                    job["parameters"]["dyna_plus_kappa"],
                )
        self.assertEqual(tile_manifest["expected_runs"], 200)
        self.assertEqual(tile_manifest["feature_representation"], "tile_coding")
        tile_job = build_jobs(tile_manifest)[0]
        tile_config = _app_config(tile_manifest, tile_job)
        tile_coder = create_feature_representation(
            tile_config.environment, tile_config.agent
        )
        self.assertEqual(tile_coder.size, 65_536)
        self.assertEqual(tile_coder.nominal_active_count, 17)

    def test_exact_auc_stream_average_postchange_and_recovery(self):
        tracker = MetricsTracker(
            window=4,
            chart_points=20,
            sample_interval=2,
            record_step_metrics=True,
            post_change_window=3,
            recovery_smoothing=2,
            recovery_tolerance=0.10,
            recovery_horizon=6,
        )
        rewards = [1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0]
        for step, reward in enumerate(rewards, start=1):
            events = ["wind_phase:1"] if step == 4 else []
            tracker.update(
                step, reward, 0.0,
                {"events": events, "goal_reached": False, "collision": False},
                reward_rate=0.0, alpha_mean=0.1,
            )
        summary = tracker.summary(len(rewards))
        self.assertEqual(summary["reward_auc"], 5.0)
        self.assertEqual(summary["stream_average_reward"], 0.625)
        rows = tracker.change_metric_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prechange_mean_reward"], 1.0)
        self.assertEqual(rows[0]["postchange_reward_auc"], 0.0)
        self.assertEqual(rows[0]["postchange_mean_reward"], 0.0)
        self.assertEqual(rows[0]["recovery_steps"], 4)

    def test_recovery_tolerance_uses_half_point_minimum_scale(self):
        tracker = MetricsTracker(
            window=2, chart_points=10, sample_interval=1,
            record_step_metrics=True, post_change_window=2,
            recovery_smoothing=2, recovery_tolerance=0.10,
            recovery_horizon=4,
        )
        # Baseline 0.2 gives a 0.05 tolerance and threshold 0.15. With the old
        # minimum scale 1.0, the first post-change window (mean 0.12) recovered.
        rewards = [0.2, 0.2, 0.12, 0.12, 0.16, 0.16]
        for step, reward in enumerate(rewards, start=1):
            tracker.update(
                step, reward, 0.0,
                {"events": ["wind_phase:1"] if step == 2 else []},
                reward_rate=0.0, alpha_mean=0.1,
            )
        row = tracker.change_metric_rows()[0]
        self.assertAlmostEqual(row["prechange_mean_reward"], 0.2)
        self.assertEqual(row["recovery_steps"], 4)
        self.assertEqual(row["postchange_window_250"], 4)

    def test_parameter_selection_requires_all_seeds_to_finish_successfully(self):
        self.assertTrue(_selection_eligible(5, 5, 0, 0))
        self.assertFalse(_selection_eligible(4, 5, 1, 0))
        self.assertFalse(_selection_eligible(4, 5, 0, 1))


if __name__ == "__main__":
    unittest.main()
