import threading
import unittest

from stream_rl_grid.algo import ALGORITHM_LABELS
from stream_rl_grid.config import GOAL_REACHED_BEHAVIOR_LABELS
from stream_rl_grid.gui import TrainingPanel


class FakeVariable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeCombo:
    def __init__(self):
        self.state = None

    def configure(self, **changes):
        self.state = changes.get("state", self.state)


class GuiSnapshotDeliveryTests(unittest.TestCase):
    def test_only_latest_unrendered_snapshot_is_retained(self):
        panel = TrainingPanel.__new__(TrainingPanel)
        panel._snapshot_lock = threading.Lock()
        panel._pending_snapshot = None

        for step in range(10_000):
            panel._publish_snapshot({"step": step})

        self.assertEqual(panel._take_pending_snapshot(), {"step": 9_999})
        self.assertIsNone(panel._take_pending_snapshot())

    def test_algorithm_labels_expose_lambda_variants_clearly(self):
        self.assertIn("SARSA(λ)", ALGORITHM_LABELS["sarsa"])
        self.assertIn("Q(λ)", ALGORITHM_LABELS["q_lambda"])

    def test_agent_fields_are_specific_to_selected_algorithm(self):
        common = set(TrainingPanel.COMMON_AGENT_FIELDS)
        tile_fields = set(
            TrainingPanel.REPRESENTATION_CONFIG_FIELDS["tile_coding"]
        )
        expected_extras = {
            "q_learning": set(),
            "q_lambda": {"lambda_"},
            "sarsa": {"lambda_"},
            "dyna_q": {"planning_steps"},
            "dyna_q_plus": {"planning_steps", "dyna_plus_kappa"},
            "dyna_q_lambda": {"lambda_", "planning_steps"},
            "tidbd": {"lambda_", "theta", "beta_min", "beta_max"},
        }
        for algorithm, extras in expected_extras.items():
            with self.subTest(algorithm=algorithm):
                self.assertEqual(
                    set(TrainingPanel.agent_fields_for_algorithm(algorithm)),
                    common | tile_fields | extras,
                )
                self.assertEqual(
                    set(TrainingPanel.agent_fields_for_algorithm(
                        algorithm, "handcrafted_lfa"
                    )),
                    common | extras,
                )
                self.assertEqual(
                    set(TrainingPanel.agent_fields_for_algorithm(
                        algorithm, "handcrafted_lfa_nuisance"
                    )),
                    common | extras,
                )

    def test_goal_reached_gui_label_maps_to_config_key(self):
        panel = TrainingPanel.__new__(TrainingPanel)
        panel.variables = {
            "goal_reached_behavior": FakeVariable(
                GOAL_REACHED_BEHAVIOR_LABELS["relocate_target"]
            )
        }
        self.assertEqual(
            panel._selected_goal_reached_behavior(), "relocate_target"
        )

    def test_goal_moves_forces_and_disables_random_restart_choice(self):
        panel = TrainingPanel.__new__(TrainingPanel)
        panel.variables = {
            "goal_moves": FakeVariable(True),
            "goal_reached_behavior": FakeVariable(
                GOAL_REACHED_BEHAVIOR_LABELS["relocate_target"]
            ),
        }
        panel.goal_behavior_combo = FakeCombo()

        panel._refresh_goal_behavior_availability()

        self.assertEqual(
            panel.variables["goal_reached_behavior"].get(),
            GOAL_REACHED_BEHAVIOR_LABELS["random_agent_restart"],
        )
        self.assertEqual(panel.goal_behavior_combo.state, "disabled")


if __name__ == "__main__":
    unittest.main()
