import unittest

import numpy as np

from experiments.features import GridFeatureEncoder, NuisanceFeatureEncoder
from experiments.learners import LinearDifferentialLearner
from experiments.lfa_runner import run_one
from experiments.phase0.protocol import condition_by_name
from experiments.phase3.p3_2_ablation import _masks


class CurrentPhaseTests(unittest.TestCase):
    def test_d55_features_are_unit_norm_and_ignore_previous_action(self) -> None:
        encoder = GridFeatureEncoder(5, 5)
        first = encoder.encode((1, 2, 4, 0, 0), 3)
        second = encoder.encode((1, 2, 4, 0, 4), 3)
        self.assertEqual(first.shape, (55,))
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0)
        self.assertLessEqual(int(np.count_nonzero(first)), 11)
        self.assertGreater(int(np.count_nonzero(first)), 0)
        np.testing.assert_array_equal(first, second)

    def test_d71_features_are_unit_norm_with_one_nuisance(self) -> None:
        encoder = NuisanceFeatureEncoder(5, 5)
        features = encoder.encode((1, 2, 4, 0, 0), 3, 7)
        self.assertEqual(features.shape, (71,))
        self.assertAlmostEqual(float(np.linalg.norm(features)), 1.0)
        self.assertLessEqual(int(np.count_nonzero(features)), 12)
        self.assertGreater(int(np.count_nonzero(features)), 1)
        self.assertEqual(encoder.groups[55 + 7], "nuisance")

    def test_fixed_and_adaptive_linear_updates(self) -> None:
        features = np.zeros(55)
        features[0] = 1.0
        successor = np.zeros(55)
        groups = np.full(55, "test")
        fixed = LinearDifferentialLearner(55, groups, fixed_alpha=0.1)
        result = fixed.update(features, 2.0, successor)
        self.assertEqual(result["delta"], 2.0)
        self.assertEqual(fixed.weights[0], 0.2)
        self.assertEqual(fixed.average_reward, 0.02)

        adaptive = LinearDifferentialLearner(55, groups, adaptive=True)
        adaptive.update(features, 2.0, successor)
        self.assertAlmostEqual(adaptive.weights[0], 0.1)
        self.assertAlmostEqual(adaptive.h[0], 0.1)

    def test_prediction_stream_is_paired_across_lfa_step_sizes(self) -> None:
        condition = condition_by_name("seasonal_wind")
        _, slow, _, _ = run_one(
            "prediction", condition, "slow", {"fixed_alpha": 0.01}, 2, 40
        )
        _, fast, _, _ = run_one(
            "prediction", condition, "fast", {"fixed_alpha": 0.10}, 2, 40
        )
        np.testing.assert_array_equal(slow["action"], fast["action"])
        np.testing.assert_array_equal(slow["observation"], fast["observation"])

    def test_phase3_ablation_groups_delete_equal_counts(self) -> None:
        alphas = np.linspace(0.001, 0.1, 71)
        groups = np.asarray(["main"] * 55 + ["nuisance"] * 16)
        masks = _masks(alphas, groups)
        self.assertEqual(int(np.count_nonzero(masks["unablated"] == 0)), 0)
        for name in ("nuisance", "low_alpha", "middle_alpha", "high_alpha"):
            self.assertEqual(int(np.count_nonzero(masks[name] == 0)), 16)

    def test_lfa_runner_records_change_and_recurrence(self) -> None:
        summary, _, _, _ = run_one(
            "prediction", condition_by_name("hidden_context"), "adaptive",
            {"adaptive": True}, 0, 1_100,
        )
        self.assertEqual(summary["change_steps"], [500, 1_000])
        self.assertEqual(len(summary["recurrences"]), 1)


if __name__ == "__main__":
    unittest.main()
