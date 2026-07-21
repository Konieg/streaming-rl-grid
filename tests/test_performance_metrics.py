import unittest

from stream_rl_grid.config import EnvironmentConfig
from stream_rl_grid.environment import ContinualWindyGridWorld
from stream_rl_grid.metrics import MetricsTracker


class PerformanceMetricsTests(unittest.TestCase):
    @staticmethod
    def update(tracker, step, reward, reward_rate, goal=False, invalid=False, events=()):
        tracker.update(
            step=step,
            reward=reward,
            delta=reward,
            info={
                "goal_reached": goal,
                "collision": invalid,
                "invalid_action": invalid,
                "events": list(events),
            },
            reward_rate=reward_rate,
            alpha_mean=0.1,
        )

    def test_window_metrics_and_reward_estimation_error(self):
        tracker = MetricsTracker(window=4, chart_points=20, sample_interval=1)
        self.update(tracker, 1, 1.0, 0.0, goal=True)
        self.update(tracker, 2, 2.0, 0.0, invalid=True)
        self.update(tracker, 3, 3.0, 0.0, goal=True)
        self.update(tracker, 4, 4.0, 3.0, invalid=True)
        self.update(tracker, 5, 5.0, 4.0, goal=True)

        summary = tracker.summary(5)
        self.assertEqual(summary["average_reward"], 3.5)
        self.assertEqual(summary["goal_count_window"], 2.0)
        self.assertEqual(summary["mean_steps_between_goals"], 2.0)
        self.assertEqual(summary["invalid_action_rate"], 0.5)
        self.assertEqual(summary["average_reward_estimation_bias"], 0.5)
        self.assertEqual(summary["average_reward_estimation_error"], 0.5)
        self.assertEqual(summary["average_reward_estimation_mean_bias"], 0.5)
        self.assertEqual(summary["average_reward_estimation_mae"], 0.5)
        self.assertEqual(summary["average_reward_estimation_rmse"], 0.5)

    def test_mean_inter_goal_time_is_nan_without_a_complete_interval(self):
        tracker = MetricsTracker(window=5, chart_points=5, sample_interval=1)
        self.update(tracker, 1, -1.0, 0.0, goal=True)
        self.assertNotEqual(tracker.summary(1)["mean_steps_between_goals"],
                            tracker.summary(1)["mean_steps_between_goals"])

    def test_adaptation_delay_recovers_after_post_event_window_and_sustain(self):
        tracker = MetricsTracker(
            window=3,
            chart_points=20,
            sample_interval=1,
            recovery_ratio=0.9,
            recovery_window=2,
            recovery_sustain=2,
            baseline_floor=1.0,
        )
        self.update(tracker, 1, 2.0, 2.0)
        self.update(tracker, 2, 2.0, 2.0)
        self.update(tracker, 3, 2.0, 2.0, events=("season:1", "goal_moved"))
        self.update(tracker, 4, 1.0, 2.0)
        self.update(tracker, 5, 2.0, 2.0)
        self.update(tracker, 6, 2.0, 2.0)
        self.update(tracker, 7, 2.0, 2.0)

        events = tracker.adaptation_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["events"], ["season:1", "goal_moved"])
        self.assertEqual(events[0]["baseline_reward"], 2.0)
        self.assertAlmostEqual(events[0]["recovery_threshold"], 1.8)
        self.assertEqual(events[0]["status"], "recovered")
        self.assertEqual(events[0]["delay"], 4)
        self.assertEqual(tracker.summary(7)["adaptation_delay_median"], 4.0)

    def test_new_event_censors_pending_and_early_event_is_unavailable(self):
        tracker = MetricsTracker(
            window=3, chart_points=20, sample_interval=1,
            recovery_window=2, recovery_sustain=2,
        )
        self.update(tracker, 1, 1.0, 1.0, events=("season:1",))
        self.update(tracker, 2, 1.0, 1.0)
        self.update(tracker, 3, 1.0, 1.0, events=("context:1",))
        self.update(tracker, 4, -5.0, 1.0)
        self.update(tracker, 5, -5.0, 1.0, events=("goal_moved",))

        events = tracker.adaptation_events()
        self.assertEqual(events[0]["status"], "unavailable")
        self.assertEqual(events[1]["status"], "censored")
        self.assertEqual(events[2]["status"], "pending")

    def test_pending_adaptation_round_trips_through_state(self):
        tracker = MetricsTracker(
            window=2, chart_points=10, sample_interval=1,
            recovery_window=2, recovery_sustain=2,
        )
        self.update(tracker, 1, 1.0, 1.0)
        self.update(tracker, 2, 1.0, 1.0, events=("season:1",))
        self.update(tracker, 3, -1.0, 1.0)
        state = tracker.state_dict()

        restored = MetricsTracker(
            window=2, chart_points=10, sample_interval=1,
            recovery_window=2, recovery_sustain=2,
        )
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)

    def test_wind_collision_is_not_an_invalid_agent_action(self):
        config = EnvironmentConfig(
            width=5,
            height=5,
            obstacle_count=0,
            profile="stationary",
            w_strength=1.0,
            manual_wind_direction="right",
            context_maps=[[]],
            start_position=[4, 2],
            goal_position=[0, 0],
        )
        environment = ContinualWindyGridWorld(config)
        _, _, _, _, wind_info = environment.step(4)
        self.assertTrue(wind_info["collision"])
        self.assertFalse(wind_info["invalid_action"])

        environment.config.w_strength = 0.0
        _, _, _, _, action_info = environment.step(1)
        self.assertTrue(action_info["collision"])
        self.assertTrue(action_info["invalid_action"])


if __name__ == "__main__":
    unittest.main()
