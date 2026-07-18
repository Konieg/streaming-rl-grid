"""Continuing-task metrics that do not rely on episode boundaries."""

from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np


class MetricsTracker:
    def __init__(
        self,
        window: int,
        chart_points: int,
        sample_interval: int,
        record_step_metrics: bool = False,
        post_change_window: int = 1_000,
        recovery_smoothing: int = 250,
        recovery_tolerance: float = 0.10,
        recovery_horizon: int = 5_000,
    ):
        self.window = int(window)
        self.chart_points = int(chart_points)
        self.sample_interval = int(sample_interval)
        self.record_step_metrics = bool(record_step_metrics)
        self.post_change_window = int(post_change_window)
        self.recovery_smoothing = int(recovery_smoothing)
        self.recovery_tolerance = float(recovery_tolerance)
        self.recovery_horizon = int(recovery_horizon)
        self.rewards: Deque[float] = deque(maxlen=self.window)
        self.abs_deltas: Deque[float] = deque(maxlen=self.window)
        self.collisions: Deque[int] = deque(maxlen=self.window)
        self.goals: Deque[int] = deque(maxlen=self.window)
        self.total_reward = 0.0
        self.total_goals = 0
        self.total_collisions = 0
        self.last_goal_step = 0
        self.goal_intervals: Deque[int] = deque(maxlen=self.window)
        self.curve_steps: Deque[int] = deque(maxlen=self.chart_points)
        self.curve_reward: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_reward_rate: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_abs_delta: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_alpha_mean: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_interval_reward: Deque[float] = deque(maxlen=self.chart_points)
        self.reward_trace: List[float] = []
        self.change_events: List[Dict[str, Any]] = []
        self.interval_reward_sum = 0.0
        self.interval_count = 0
        self.last_interval_reward_sum = 0.0
        self.last_interval_count = 0

    def update(
        self,
        step: int,
        reward: float,
        delta: float,
        info: Dict[str, Any],
        reward_rate: float,
        alpha_mean: float,
    ) -> None:
        goal = int(bool(info.get("goal_reached", False)))
        collision = int(bool(info.get("collision", False)))
        self.rewards.append(float(reward))
        self.abs_deltas.append(abs(float(delta)))
        self.collisions.append(collision)
        self.goals.append(goal)
        self.total_reward += float(reward)
        self.interval_reward_sum += float(reward)
        self.interval_count += 1
        self.total_goals += goal
        self.total_collisions += collision
        if self.record_step_metrics:
            self.reward_trace.append(float(reward))
            for event in info.get("events", []):
                event_type = self._external_event_type(str(event))
                if event_type is not None:
                    self.change_events.append({
                        "step": int(step),
                        "event_type": event_type,
                        "event": str(event),
                    })
        if goal:
            if self.last_goal_step > 0:
                self.goal_intervals.append(step - self.last_goal_step)
            self.last_goal_step = step
        if step % self.sample_interval == 0:
            self.last_interval_reward_sum = self.interval_reward_sum
            self.last_interval_count = self.interval_count
            self.curve_steps.append(step)
            self.curve_reward.append(self.window_average_reward)
            self.curve_reward_rate.append(float(reward_rate))
            self.curve_abs_delta.append(self.window_abs_delta)
            self.curve_alpha_mean.append(float(alpha_mean))
            self.curve_interval_reward.append(self.interval_average_reward)
            self.interval_reward_sum = 0.0
            self.interval_count = 0

    @staticmethod
    def _external_event_type(event: str):
        if event.startswith("wind_phase:"):
            return "wind"
        if event.startswith("reward_phase:"):
            return "reward"
        if event == "goal_moved":
            return "goal"
        if event.startswith("context:"):
            return "obstacles"
        return None

    @property
    def interval_average_reward(self) -> float:
        return (
            float(self.last_interval_reward_sum / self.last_interval_count)
            if self.last_interval_count else 0.0
        )

    @property
    def window_average_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def window_abs_delta(self) -> float:
        return float(np.mean(self.abs_deltas)) if self.abs_deltas else 0.0

    @property
    def window_goal_rate(self) -> float:
        return float(np.mean(self.goals)) if self.goals else 0.0

    @property
    def window_collision_rate(self) -> float:
        return float(np.mean(self.collisions)) if self.collisions else 0.0

    def summary(self, step: int) -> Dict[str, float]:
        stream_average = self.total_reward / step if step > 0 else 0.0
        return {
            "step": float(step),
            "average_reward": self.window_average_reward,
            "abs_td_error": self.window_abs_delta,
            "goals_per_1000_steps": self.window_goal_rate * 1000.0,
            "collision_rate": self.window_collision_rate,
            "mean_steps_between_goals": float(np.mean(self.goal_intervals)) if self.goal_intervals else 0.0,
            "total_goals": float(self.total_goals),
            "total_collisions": float(self.total_collisions),
            "reward_auc": float(self.total_reward),
            "stream_average_reward": float(stream_average),
            "interval_reward_sum": float(self.last_interval_reward_sum),
            "interval_reward_count": float(self.last_interval_count),
            "interval_average_reward": self.interval_average_reward,
        }

    def change_metric_rows(self) -> List[Dict[str, Any]]:
        """Return exact event-aligned reward and recovery statistics.

        A schedule changes after the reward at its boundary step, so an event at
        step e uses rewards e+1 onward (zero-based trace index e) as post-change
        data. Recovery is deliberately censored at the next external event.
        """
        if not self.record_step_metrics:
            return []
        rewards = np.asarray(self.reward_trace, dtype=np.float64)
        rows: List[Dict[str, Any]] = []
        for index, event in enumerate(self.change_events):
            step = int(event["step"])
            next_step = (
                int(self.change_events[index + 1]["step"])
                if index + 1 < len(self.change_events) else len(rewards)
            )
            pre_start = max(0, step - self.window)
            pre = rewards[pre_start:step]
            post_end = min(
                len(rewards), next_step, step + self.post_change_window
            )
            post = rewards[step:post_end]
            baseline = float(np.mean(pre)) if pre.size else float("nan")
            post_sum = float(np.sum(post))
            post_mean = float(np.mean(post)) if post.size else float("nan")
            recovery_end = min(
                len(rewards), next_step, step + self.recovery_horizon
            )
            recovery_signal = rewards[step:recovery_end]
            recovery_steps = None
            if (
                np.isfinite(baseline)
                and recovery_signal.size >= self.recovery_smoothing
            ):
                kernel = np.full(
                    self.recovery_smoothing,
                    1.0 / self.recovery_smoothing,
                    dtype=np.float64,
                )
                smoothed = np.convolve(recovery_signal, kernel, mode="valid")
                tolerance = self.recovery_tolerance * max(1.0, abs(baseline))
                recovered = np.flatnonzero(smoothed >= baseline - tolerance)
                if recovered.size:
                    recovery_steps = int(recovered[0] + self.recovery_smoothing)
            rows.append({
                "step": step,
                "event_type": event["event_type"],
                "event": event["event"],
                "prechange_window": int(pre.size),
                "prechange_mean_reward": baseline,
                "postchange_window": int(post.size),
                "postchange_reward_auc": post_sum,
                "postchange_mean_reward": post_mean,
                "recovery_steps": recovery_steps,
                "recovered_before_next_change": recovery_steps is not None,
                "next_change_step": int(next_step),
            })
        return rows

    def curves(self) -> Dict[str, List[float]]:
        return {
            "steps": list(self.curve_steps),
            "average_reward": list(self.curve_reward),
            "reward_rate": list(self.curve_reward_rate),
            "abs_td_error": list(self.curve_abs_delta),
            "alpha_mean": list(self.curve_alpha_mean),
            "interval_average_reward": list(self.curve_interval_reward),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "chart_points": self.chart_points,
            "sample_interval": self.sample_interval,
            "record_step_metrics": self.record_step_metrics,
            "post_change_window": self.post_change_window,
            "recovery_smoothing": self.recovery_smoothing,
            "recovery_tolerance": self.recovery_tolerance,
            "recovery_horizon": self.recovery_horizon,
            "rewards": list(self.rewards),
            "abs_deltas": list(self.abs_deltas),
            "collisions": list(self.collisions),
            "goals": list(self.goals),
            "total_reward": self.total_reward,
            "total_goals": self.total_goals,
            "total_collisions": self.total_collisions,
            "last_goal_step": self.last_goal_step,
            "goal_intervals": list(self.goal_intervals),
            "reward_trace": self.reward_trace,
            "change_events": self.change_events,
            "interval_reward_sum": self.interval_reward_sum,
            "interval_count": self.interval_count,
            "last_interval_reward_sum": self.last_interval_reward_sum,
            "last_interval_count": self.last_interval_count,
            "curves": self.curves(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state["window"]) != self.window:
            raise ValueError("Checkpoint metric window is incompatible.")
        self.rewards = deque(state["rewards"], maxlen=self.window)
        self.abs_deltas = deque(state["abs_deltas"], maxlen=self.window)
        self.collisions = deque(state["collisions"], maxlen=self.window)
        self.goals = deque(state["goals"], maxlen=self.window)
        self.total_reward = float(state["total_reward"])
        self.total_goals = int(state["total_goals"])
        self.total_collisions = int(state["total_collisions"])
        self.last_goal_step = int(state["last_goal_step"])
        self.goal_intervals = deque(state["goal_intervals"], maxlen=self.window)
        self.reward_trace = [float(value) for value in state.get("reward_trace", [])]
        self.change_events = [dict(event) for event in state.get("change_events", [])]
        self.interval_reward_sum = float(state.get("interval_reward_sum", 0.0))
        self.interval_count = int(state.get("interval_count", 0))
        self.last_interval_reward_sum = float(state.get("last_interval_reward_sum", 0.0))
        self.last_interval_count = int(state.get("last_interval_count", 0))
        curves = state["curves"]
        self.curve_steps = deque(curves["steps"], maxlen=self.chart_points)
        self.curve_reward = deque(curves["average_reward"], maxlen=self.chart_points)
        self.curve_reward_rate = deque(curves["reward_rate"], maxlen=self.chart_points)
        self.curve_abs_delta = deque(curves["abs_td_error"], maxlen=self.chart_points)
        self.curve_alpha_mean = deque(curves["alpha_mean"], maxlen=self.chart_points)
        self.curve_interval_reward = deque(
            curves.get("interval_average_reward", []), maxlen=self.chart_points
        )
