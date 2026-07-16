"""Runnable E1–E8 streaming experiments on the shared GridWorld."""

import csv
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stream_rl_grid.config import EnvironmentConfig
from stream_rl_grid.environment import ContinualWindyGridWorld

from .features import GridFeatureEncoder
from .learners import LinearOnlineLearner


PRESETS = {
    "e1": ("stationary", "supervised", "reward"),
    "e2": ("moving_goal", "supervised", "reward"),
    "e3": ("hidden_context", "supervised", "collision"),
    "e4": ("hidden_context", "supervised", "collision"),
    "e5": ("combined", "td", "reward"),
    "e7": ("moving_goal", "td", "reward"),
    "e8": ("stationary", "supervised", "reward"),
}


@dataclass
class ExperimentConfig:
    experiment: str = "e1"
    steps: int = 20_000
    seeds: Tuple[int, ...] = (0,)
    method: str = "scalar"
    profile: Optional[str] = None
    task: Optional[str] = None
    target: Optional[str] = None
    initial_alpha: float = 0.05
    theta: float = 0.01
    reward_rate_step: float = 0.01
    width: int = 5
    height: int = 5
    obstacles: int = 3
    num_contexts: int = 2
    wind_period: int = 400
    target_move_interval: int = 300
    context_switch_interval: int = 500
    noise_features: int = 0
    oracle_context: bool = False
    ablate_groups: Tuple[str, ...] = ()
    report_every: int = 50
    window: int = 100
    alignment_window: int = 200
    output: Optional[str] = None

    def resolved(self) -> "ExperimentConfig":
        if self.experiment not in PRESETS:
            raise ValueError("experiment must be one of %s" % ", ".join(sorted(PRESETS)))
        profile, task, target = PRESETS[self.experiment]
        result = ExperimentConfig(**asdict(self))
        result.profile = result.profile or profile
        result.task = result.task or task
        result.target = result.target or target
        if result.task not in ("supervised", "td"):
            raise ValueError("task must be supervised or td.")
        if result.target not in ("reward", "collision", "goal"):
            raise ValueError("target must be reward, collision, or goal.")
        if result.steps <= 0 or result.report_every <= 0 or result.window <= 0:
            raise ValueError("steps, report_every, and window must be positive.")
        if result.experiment == "e7" and result.method == "scalar":
            result.method = "tidbd"
        return result


def _environment(config: ExperimentConfig, seed: int) -> ContinualWindyGridWorld:
    env_config = EnvironmentConfig(
        width=config.width,
        height=config.height,
        obstacle_count=config.obstacles,
        num_contexts=config.num_contexts,
        profile=str(config.profile),
        seed=int(seed),
        max_wind_strength=1,
        wind_period=config.wind_period,
        target_move_interval=config.target_move_interval,
        context_switch_interval=config.context_switch_interval,
        manual_wind_direction="auto" if config.profile in ("seasonal_wind", "combined") else "none",
    )
    return ContinualWindyGridWorld(env_config)


def _target(name: str, reward: float, info: Dict[str, object]) -> float:
    if name == "reward":
        return float(reward)
    if name == "collision":
        return float(bool(info["collision"]))
    return float(bool(info["goal_reached"]))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_one(config: ExperimentConfig, seed: int) -> List[Dict[str, object]]:
    environment = _environment(config, seed)
    encoder = GridFeatureEncoder(
        width=environment.width,
        height=environment.height,
        noise_features=config.noise_features,
        include_context=config.oracle_context,
        num_contexts=environment.config.num_contexts,
        ablate_groups=config.ablate_groups,
        seed=10_000 + seed,
    )
    learner = LinearOnlineLearner(
        size=encoder.size,
        group_by_index=encoder.group_by_index,
        method=config.method,
        initial_alpha=config.initial_alpha,
        theta=config.theta,
    )
    policy_rng = np.random.default_rng(20_000 + seed)
    observation = environment.observation()
    action = int(policy_rng.integers(5))
    reward_rate = 0.0
    losses = deque(maxlen=config.window)
    rows: List[Dict[str, object]] = []

    for step in range(1, config.steps + 1):
        context_before = environment.context_index
        features = encoder.encode(observation, action, context_before)
        prediction = learner.predict(features)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        if terminated or truncated:
            raise RuntimeError("The experiment environment must be continuing.")
        next_action = int(policy_rng.integers(5))
        if config.task == "supervised":
            target = _target(str(config.target), reward, info)
            error = target - prediction
            learner.update(features, error)
            loss = error * error
        else:
            next_features = encoder.encode(next_observation, next_action, environment.context_index)
            next_value = learner.predict(next_features)
            error = float(reward - reward_rate + next_value - prediction)
            learner.update(features, error)
            reward_rate += config.reward_rate_step * error
            target = prediction + error
            loss = error * error
        losses.append(float(loss))
        alpha = learner.active_alpha_summary(features)
        if step % config.report_every == 0 or info["events"]:
            row: Dict[str, object] = {
                "seed": seed,
                "step": step,
                "experiment": config.experiment,
                "profile": config.profile,
                "task": config.task,
                "target_name": config.target,
                "method": config.method,
                "target": target,
                "prediction": prediction,
                "error": error,
                "loss": loss,
                "window_loss": float(np.mean(losses)),
                "reward": reward,
                "reward_rate": reward_rate,
                "collision": int(bool(info["collision"])),
                "goal_reached": int(bool(info["goal_reached"])),
                "context": context_before,
                "context_after": environment.context_index,
                "goal_x": environment.goal[0],
                "goal_y": environment.goal[1],
                "wind_x": info["wind"][0],
                "wind_y": info["wind"][1],
                "events": "|".join(info["events"]),
            }
            row.update(alpha)
            rows.append(row)
        observation, action = next_observation, next_action
    return rows


def _aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)
    result = []
    for step in sorted(grouped):
        records = grouped[step]
        numeric = ("window_loss", "loss", "error", "reward", "reward_rate", "alpha_active_mean")
        row: Dict[str, object] = {"step": step, "runs": len(records)}
        for name in numeric:
            values = [float(record[name]) for record in records if name in record]
            if values:
                row["mean_%s" % name] = float(np.mean(values))
                row["stderr_%s" % name] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        result.append(row)
    return result


def _event_aligned(rows: Sequence[Dict[str, object]], window: int) -> List[Dict[str, object]]:
    by_seed: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    collected: Dict[int, List[float]] = defaultdict(list)
    for seed_rows in by_seed.values():
        ordered = sorted(seed_rows, key=lambda row: int(row["step"]))
        by_step = {int(row["step"]): row for row in ordered}
        for event_row in ordered:
            if not event_row["events"]:
                continue
            event_step = int(event_row["step"])
            for offset in range(0, window + 1):
                later = by_step.get(event_step + offset)
                if later is not None:
                    collected[offset].append(float(later["loss"]))
    return [
        {"relative_step": offset, "mean_loss": float(np.mean(values)), "samples": len(values)}
        for offset, values in sorted(collected.items())
    ]


def _plot(output: Path, aggregate: Sequence[Dict[str, object]], config: ExperimentConfig) -> None:
    if not aggregate:
        return
    steps = [int(row["step"]) for row in aggregate]
    losses = [float(row["mean_window_loss"]) for row in aggregate]
    alphas = [float(row["mean_alpha_active_mean"]) for row in aggregate]
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(steps, losses, label="window loss")
    axes[0].set_ylabel("mean loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(steps, alphas, label="active alpha", color="tab:orange")
    axes[1].set_xlabel("stream step")
    axes[1].set_ylabel("mean active alpha")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("%s | %s | %s" % (config.experiment, config.profile, config.method))
    figure.tight_layout()
    figure.savefig(output / "learning_curve.png", dpi=150)
    plt.close(figure)


def run_experiment(config: ExperimentConfig) -> Path:
    config = config.resolved()
    root = Path(config.output) if config.output else Path("experiment_results") / (
        "%s-%s" % (config.experiment, datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    all_rows: List[Dict[str, object]] = []
    for seed in config.seeds:
        rows = _run_one(config, int(seed))
        _write_csv(root / ("seed-%03d.csv" % seed), rows)
        all_rows.extend(rows)
    aggregate = _aggregate(all_rows)
    aligned = _event_aligned(all_rows, config.alignment_window)
    _write_csv(root / "aggregate.csv", aggregate)
    _write_csv(root / "event_aligned.csv", aligned)
    _plot(root, aggregate, config)
    with (root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    return root
