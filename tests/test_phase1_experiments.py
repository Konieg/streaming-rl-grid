import unittest

from stream_rl_grid.metrics import MetricsTracker
from stream_rl_grid.phase1_plot import _event_steps
from stream_rl_grid.phase1_sweep import build_jobs, make_manifest, parameter_configurations


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


if __name__ == "__main__":
    unittest.main()
