"""Multi-seed comparison of TIDBD and fixed-step differential Sarsa."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .config import AppConfig, FEATURE_REPRESENTATIONS
from .trainer import Trainer


def run_benchmark(
    seeds: List[int], steps: int, output: Path,
    feature_representation: str = "tile_coding",
    wind_changes: bool = False,
    goal_moves: bool = False,
    obstacle_switches: bool = False,
    reward_changes: bool = False,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    curves = {}
    enabled = [
        name for name, active in (
            ("wind", wind_changes), ("goal", goal_moves),
            ("obstacles", obstacle_switches), ("reward", reward_changes),
        ) if active
    ]
    setting = "+".join(enabled) if enabled else "stationary"
    for algorithm in ("tidbd", "sarsa"):
        label = "TIDBD" if algorithm == "tidbd" else "Sarsa"
        method_curves = []
        for seed in seeds:
            config = AppConfig()
            config.environment.seed = seed
            config.environment.wind_changes = wind_changes
            config.environment.goal_moves = goal_moves
            config.environment.obstacle_switches = obstacle_switches
            config.environment.reward_changes = reward_changes
            config.agent.algorithm = algorithm
            config.agent.feature_representation = feature_representation
            config.training.auto_checkpoint_steps = steps + 1
            trainer = Trainer(config, base_dir=output)
            snapshot = trainer.run_steps(steps)
            rows.append(
                {
                    "setting": setting,
                    "method": label,
                    "seed": seed,
                    "steps": steps,
                    "average_reward": snapshot["average_reward"],
                    "reward_rate": snapshot["reward_rate"],
                    "goals_per_1000_steps": snapshot["goals_per_1000_steps"],
                    "collision_rate": snapshot["collision_rate"],
                    "abs_td_error": snapshot["abs_td_error"],
                }
            )
            method_curves.append(snapshot["curves"])
        curves[label] = method_curves

    csv_path = output / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(1, 1, figsize=(10, 4))
    for label in ("TIDBD", "Sarsa"):
        entries = curves[label]
        minimum = min(len(entry["steps"]) for entry in entries)
        x = np.asarray(entries[0]["steps"][:minimum])
        y = np.asarray([entry["average_reward"][:minimum] for entry in entries], dtype=float)
        mean = y.mean(axis=0)
        stderr = y.std(axis=0, ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.zeros_like(mean)
        axis.plot(x, mean, label=label)
        axis.fill_between(x, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.2)
    axis.set_title(setting)
    axis.set_xlabel("stream step")
    axis.set_ylabel("window average reward")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=150)
    plt.close(figure)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TIDBD against a fixed-step baseline")
    parser.add_argument("--wind-changes", action="store_true")
    parser.add_argument("--goal-moves", action="store_true")
    parser.add_argument("--obstacle-switches", action="store_true")
    parser.add_argument("--reward-changes", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument(
        "--features", choices=FEATURE_REPRESENTATIONS, default="tile_coding"
    )
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output) if args.output else root / "benchmark_results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = run_benchmark(
        args.seeds, args.steps, output.resolve(), args.features,
        args.wind_changes, args.goal_moves, args.obstacle_switches,
        args.reward_changes,
    )
    print("Benchmark written to %s" % csv_path.parent)


if __name__ == "__main__":
    main()
