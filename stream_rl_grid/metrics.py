"""Continuing-task metrics that do not rely on episode boundaries."""

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np


class MetricsTracker:
    ENVIRONMENT_EVENT_PREFIXES = ("season:", "context:")
    ENVIRONMENT_EVENT_NAMES = ("goal_moved",)

    def __init__(
        self,
        window: int,
        chart_points: int,
        sample_interval: int,
        recovery_ratio: float = 0.9,
        recovery_window: int = 100,
        recovery_sustain: int = 100,
        baseline_floor: float = 1.0,
    ):
        self.window = int(window)
        self.chart_points = int(chart_points)
        self.sample_interval = int(sample_interval)
        self.recovery_ratio = float(recovery_ratio)
        self.recovery_window = int(recovery_window)
        self.recovery_sustain = int(recovery_sustain)
        self.baseline_floor = float(baseline_floor)

        self.rewards: Deque[float] = deque(maxlen=self.window)
        self.abs_deltas: Deque[float] = deque(maxlen=self.window)
        self.collisions: Deque[int] = deque(maxlen=self.window)
        self.invalid_actions: Deque[int] = deque(maxlen=self.window)
        self.goals: Deque[int] = deque(maxlen=self.window)
        self.total_reward = 0.0
        self.total_goals = 0
        self.total_collisions = 0
        self.total_invalid_actions = 0
        self.last_goal_step = 0
        self.goal_interval_records: Deque[Dict[str, int]] = deque(maxlen=self.window)

        self.reward_error_count = 0
        self.reward_error_sum = 0.0
        self.reward_error_abs_sum = 0.0
        self.reward_error_squared_sum = 0.0
        self.current_reward_estimation_bias = float("nan")

        self.adaptation_records: List[Dict[str, Any]] = []
        self.pending_adaptation: Optional[Dict[str, Any]] = None

        self.curve_steps: Deque[int] = deque(maxlen=self.chart_points)
        self.curve_reward: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_reward_rate: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_abs_delta: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_alpha_mean: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_epsilon: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_td_error_magnitude: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_goal_count: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_mean_inter_goal: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_invalid_action_rate: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_reward_estimation_bias: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_reward_estimation_abs_error: Deque[float] = deque(maxlen=self.chart_points)

    def update(
        self,
        step: int,
        reward: float,
        delta: float,
        info: Dict[str, Any],
        reward_rate: float,
        alpha_mean: float,
        epsilon: float = float("nan"),
        td_error_magnitude: float = float("nan"),
    ) -> None:
        goal = int(bool(info.get("goal_reached", False)))
        collision = int(bool(info.get("collision", False)))
        invalid_action = int(bool(info.get("invalid_action", False)))
        self.rewards.append(float(reward))
        self.abs_deltas.append(abs(float(delta)))
        self.collisions.append(collision)
        self.invalid_actions.append(invalid_action)
        self.goals.append(goal)
        self.total_reward += float(reward)
        self.total_goals += goal
        self.total_collisions += collision
        self.total_invalid_actions += invalid_action

        if goal:
            if self.last_goal_step > 0:
                self.goal_interval_records.append(
                    {"end_step": int(step), "interval": int(step - self.last_goal_step)}
                )
            self.last_goal_step = int(step)

        if len(self.rewards) == self.window:
            self.current_reward_estimation_bias = float(reward_rate) - self.window_average_reward
            self.reward_error_count += 1
            self.reward_error_sum += self.current_reward_estimation_bias
            self.reward_error_abs_sum += abs(self.current_reward_estimation_bias)
            self.reward_error_squared_sum += self.current_reward_estimation_bias ** 2
        else:
            self.current_reward_estimation_bias = float("nan")

        self._advance_pending_adaptation(int(step), float(reward))
        environment_events = self._environment_events(info.get("events", []))
        if environment_events:
            self._start_adaptation_event(int(step), environment_events)

        if step % self.sample_interval == 0:
            self.curve_steps.append(step)
            self.curve_reward.append(self.window_average_reward)
            self.curve_reward_rate.append(float(reward_rate))
            self.curve_abs_delta.append(self.window_abs_delta)
            self.curve_alpha_mean.append(float(alpha_mean))
            self.curve_epsilon.append(float(epsilon))
            self.curve_td_error_magnitude.append(float(td_error_magnitude))
            self.curve_goal_count.append(float(self.window_goal_count))
            self.curve_mean_inter_goal.append(self.window_mean_inter_goal_time(step))
            self.curve_invalid_action_rate.append(self.window_invalid_action_rate)
            self.curve_reward_estimation_bias.append(self.current_reward_estimation_bias)
            self.curve_reward_estimation_abs_error.append(
                abs(self.current_reward_estimation_bias)
                if np.isfinite(self.current_reward_estimation_bias) else float("nan")
            )

    @classmethod
    def _environment_events(cls, raw_events) -> List[str]:
        return [
            str(event)
            for event in raw_events
            if str(event) in cls.ENVIRONMENT_EVENT_NAMES
            or str(event).startswith(cls.ENVIRONMENT_EVENT_PREFIXES)
        ]

    def _start_adaptation_event(self, step: int, events: List[str]) -> None:
        if self.pending_adaptation is not None:
            censored = self._public_adaptation_record(self.pending_adaptation)
            censored.update({"status": "censored", "end_step": step, "delay": None})
            self.adaptation_records.append(censored)
            self.pending_adaptation = None

        if len(self.rewards) < self.window:
            self.adaptation_records.append(
                {
                    "event_step": step,
                    "events": list(events),
                    "status": "unavailable",
                    "baseline_reward": None,
                    "recovery_threshold": None,
                    "end_step": step,
                    "delay": None,
                }
            )
            return

        baseline = self.window_average_reward
        tolerance = (1.0 - self.recovery_ratio) * max(abs(baseline), self.baseline_floor)
        self.pending_adaptation = {
            "event_step": step,
            "events": list(events),
            "status": "pending",
            "baseline_reward": baseline,
            "recovery_threshold": baseline - tolerance,
            "post_rewards": [],
            "consecutive_recovered": 0,
        }

    def _advance_pending_adaptation(self, step: int, reward: float) -> None:
        pending = self.pending_adaptation
        if pending is None or step <= int(pending["event_step"]):
            return
        post_rewards = pending["post_rewards"]
        post_rewards.append(float(reward))
        if len(post_rewards) > self.recovery_window:
            del post_rewards[0]
        if len(post_rewards) < self.recovery_window:
            return
        recovery_mean = float(np.mean(post_rewards))
        if recovery_mean >= float(pending["recovery_threshold"]):
            pending["consecutive_recovered"] = int(pending["consecutive_recovered"]) + 1
        else:
            pending["consecutive_recovered"] = 0
        if int(pending["consecutive_recovered"]) >= self.recovery_sustain:
            recovered = self._public_adaptation_record(pending)
            recovered.update(
                {
                    "status": "recovered",
                    "end_step": step,
                    "delay": step - int(pending["event_step"]),
                }
            )
            self.adaptation_records.append(recovered)
            self.pending_adaptation = None

    @staticmethod
    def _public_adaptation_record(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event_step": int(record["event_step"]),
            "events": list(record["events"]),
            "status": str(record["status"]),
            "baseline_reward": record.get("baseline_reward"),
            "recovery_threshold": record.get("recovery_threshold"),
        }

    def adaptation_events(self) -> List[Dict[str, Any]]:
        records = [dict(record) for record in self.adaptation_records]
        if self.pending_adaptation is not None:
            pending = self._public_adaptation_record(self.pending_adaptation)
            pending.update({"end_step": None, "delay": None})
            records.append(pending)
        return records

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
    def window_goal_count(self) -> int:
        return int(sum(self.goals))

    @property
    def window_collision_rate(self) -> float:
        return float(np.mean(self.collisions)) if self.collisions else 0.0

    @property
    def window_invalid_action_rate(self) -> float:
        return float(np.mean(self.invalid_actions)) if self.invalid_actions else 0.0

    def window_mean_inter_goal_time(self, step: int) -> float:
        lower = int(step) - self.window + 1
        intervals = [
            int(record["interval"])
            for record in self.goal_interval_records
            if int(record["end_step"]) >= lower
        ]
        return float(np.mean(intervals)) if intervals else float("nan")

    def _reward_error_summary(self) -> Dict[str, float]:
        if self.reward_error_count == 0:
            return {
                "average_reward_estimation_mean_bias": float("nan"),
                "average_reward_estimation_mae": float("nan"),
                "average_reward_estimation_rmse": float("nan"),
            }
        count = float(self.reward_error_count)
        return {
            "average_reward_estimation_mean_bias": self.reward_error_sum / count,
            "average_reward_estimation_mae": self.reward_error_abs_sum / count,
            "average_reward_estimation_rmse": float(
                np.sqrt(self.reward_error_squared_sum / count)
            ),
        }

    def _adaptation_summary(self) -> Dict[str, float]:
        records = self.adaptation_events()
        delays = [float(record["delay"]) for record in records if record["status"] == "recovered"]
        result = {
            "adaptation_delay_mean": float(np.mean(delays)) if delays else float("nan"),
            "adaptation_delay_median": float(np.median(delays)) if delays else float("nan"),
            "adaptation_recovered_count": float(sum(r["status"] == "recovered" for r in records)),
            "adaptation_censored_count": float(sum(r["status"] == "censored" for r in records)),
            "adaptation_pending_count": float(sum(r["status"] == "pending" for r in records)),
            "adaptation_unavailable_count": float(sum(r["status"] == "unavailable" for r in records)),
        }
        return result

    def summary(self, step: int) -> Dict[str, float]:
        current_bias = self.current_reward_estimation_bias
        result = {
            "step": float(step),
            "average_reward": self.window_average_reward,
            "abs_td_error": self.window_abs_delta,
            "goal_count_window": float(self.window_goal_count),
            "goals_per_1000_steps": self.window_goal_rate * 1000.0,
            "collision_rate": self.window_collision_rate,
            "invalid_action_rate": self.window_invalid_action_rate,
            "mean_steps_between_goals": self.window_mean_inter_goal_time(step),
            "average_reward_estimation_bias": current_bias,
            "average_reward_estimation_error": (
                abs(current_bias) if np.isfinite(current_bias) else float("nan")
            ),
            "total_goals": float(self.total_goals),
            "total_collisions": float(self.total_collisions),
            "total_invalid_actions": float(self.total_invalid_actions),
        }
        result.update(self._reward_error_summary())
        result.update(self._adaptation_summary())
        return result

    def curves(self) -> Dict[str, List[float]]:
        return {
            "steps": list(self.curve_steps),
            "average_reward": list(self.curve_reward),
            "reward_rate": list(self.curve_reward_rate),
            "abs_td_error": list(self.curve_abs_delta),
            "alpha_mean": list(self.curve_alpha_mean),
            "epsilon": list(self.curve_epsilon),
            "td_error_magnitude": list(self.curve_td_error_magnitude),
            "goal_count_window": list(self.curve_goal_count),
            "mean_inter_goal_time": list(self.curve_mean_inter_goal),
            "invalid_action_rate": list(self.curve_invalid_action_rate),
            "average_reward_estimation_bias": list(self.curve_reward_estimation_bias),
            "average_reward_estimation_error": list(self.curve_reward_estimation_abs_error),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "chart_points": self.chart_points,
            "sample_interval": self.sample_interval,
            "recovery_ratio": self.recovery_ratio,
            "recovery_window": self.recovery_window,
            "recovery_sustain": self.recovery_sustain,
            "baseline_floor": self.baseline_floor,
            "rewards": list(self.rewards),
            "abs_deltas": list(self.abs_deltas),
            "collisions": list(self.collisions),
            "invalid_actions": list(self.invalid_actions),
            "goals": list(self.goals),
            "total_reward": self.total_reward,
            "total_goals": self.total_goals,
            "total_collisions": self.total_collisions,
            "total_invalid_actions": self.total_invalid_actions,
            "last_goal_step": self.last_goal_step,
            "goal_interval_records": list(self.goal_interval_records),
            "reward_error_count": self.reward_error_count,
            "reward_error_sum": self.reward_error_sum,
            "reward_error_abs_sum": self.reward_error_abs_sum,
            "reward_error_squared_sum": self.reward_error_squared_sum,
            "current_reward_estimation_bias": self.current_reward_estimation_bias,
            "adaptation_records": self.adaptation_records,
            "pending_adaptation": self.pending_adaptation,
            "curves": self.curves(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state["window"]) != self.window:
            raise ValueError("Checkpoint metric window is incompatible.")
        self.rewards = deque(state["rewards"], maxlen=self.window)
        self.abs_deltas = deque(state["abs_deltas"], maxlen=self.window)
        self.collisions = deque(state["collisions"], maxlen=self.window)
        self.invalid_actions = deque(
            state.get("invalid_actions", state["collisions"]), maxlen=self.window
        )
        self.goals = deque(state["goals"], maxlen=self.window)
        self.total_reward = float(state["total_reward"])
        self.total_goals = int(state["total_goals"])
        self.total_collisions = int(state["total_collisions"])
        self.total_invalid_actions = int(state.get("total_invalid_actions", 0))
        self.last_goal_step = int(state["last_goal_step"])
        self.goal_interval_records = deque(
            (dict(record) for record in state.get("goal_interval_records", [])),
            maxlen=self.window,
        )
        self.reward_error_count = int(state.get("reward_error_count", 0))
        self.reward_error_sum = float(state.get("reward_error_sum", 0.0))
        self.reward_error_abs_sum = float(state.get("reward_error_abs_sum", 0.0))
        self.reward_error_squared_sum = float(state.get("reward_error_squared_sum", 0.0))
        self.current_reward_estimation_bias = float(
            state.get("current_reward_estimation_bias", float("nan"))
        )
        self.adaptation_records = [dict(record) for record in state.get("adaptation_records", [])]
        pending = state.get("pending_adaptation")
        self.pending_adaptation = None if pending is None else dict(pending)

        curves = state["curves"]
        self.curve_steps = deque(curves["steps"], maxlen=self.chart_points)
        self.curve_reward = deque(curves["average_reward"], maxlen=self.chart_points)
        self.curve_reward_rate = deque(curves["reward_rate"], maxlen=self.chart_points)
        self.curve_abs_delta = deque(curves["abs_td_error"], maxlen=self.chart_points)
        self.curve_alpha_mean = deque(curves["alpha_mean"], maxlen=self.chart_points)
        missing = [float("nan")] * len(self.curve_steps)
        self.curve_epsilon = deque(curves.get("epsilon", missing), maxlen=self.chart_points)
        self.curve_td_error_magnitude = deque(
            curves.get("td_error_magnitude", missing), maxlen=self.chart_points
        )
        self.curve_goal_count = deque(
            curves.get("goal_count_window", missing), maxlen=self.chart_points
        )
        self.curve_mean_inter_goal = deque(
            curves.get("mean_inter_goal_time", missing), maxlen=self.chart_points
        )
        self.curve_invalid_action_rate = deque(
            curves.get("invalid_action_rate", missing), maxlen=self.chart_points
        )
        self.curve_reward_estimation_bias = deque(
            curves.get("average_reward_estimation_bias", missing), maxlen=self.chart_points
        )
        self.curve_reward_estimation_abs_error = deque(
            curves.get("average_reward_estimation_error", missing), maxlen=self.chart_points
        )
