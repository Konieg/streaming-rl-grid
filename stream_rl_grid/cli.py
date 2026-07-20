"""Headless streaming trainer; Ctrl+C is a supported manual stopping mechanism."""

import argparse
from pathlib import Path

from .config import (
    ALGORITHMS,
    FEATURE_REPRESENTATIONS,
    AppConfig,
    GOAL_REACHED_BEHAVIORS,
    WIND_CHOICES,
)
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming differential TD-control algorithms")
    parser.add_argument("--resume", type=str, help="Checkpoint to continue exactly")
    parser.add_argument("--wind-changes", action="store_true")
    parser.add_argument("--goal-moves", action="store_true")
    parser.add_argument("--obstacle-switches", action="store_true")
    parser.add_argument("--reward-changes", action="store_true")
    parser.add_argument("--wind-period", type=int, default=2_000)
    parser.add_argument("--reward-period", type=int, default=2_000)
    parser.add_argument("--target-move-interval", type=int, default=500)
    parser.add_argument("--context-switch-interval", type=int, default=3_000)
    parser.add_argument("--wind-start-step", type=int)
    parser.add_argument("--reward-start-step", type=int)
    parser.add_argument("--target-move-start-step", type=int)
    parser.add_argument("--context-switch-start-step", type=int)
    parser.add_argument("--num-contexts", type=int, default=3)
    parser.add_argument("--wind-direction", choices=WIND_CHOICES, default="none")
    parser.add_argument("--wind-strength", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl+C")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--obstacles", type=int, default=8)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="tidbd")
    parser.add_argument(
        "--features",
        choices=FEATURE_REPRESENTATIONS,
        default="tile_coding",
        help="Action-value feature representation",
    )
    parser.add_argument(
        "--fixed-alpha", action="store_true",
        help="Deprecated alias for --algorithm sarsa",
    )
    parser.add_argument("--planning-steps", type=int, default=5)
    parser.add_argument(
        "--dyna-plus-kappa", type=float, default=0.001,
        help="Dyna-Q+ time-bonus coefficient kappa",
    )
    parser.add_argument(
        "--goal-reached-behavior",
        choices=GOAL_REACHED_BEHAVIORS,
        default="random_agent_restart",
    )
    parser.add_argument("--report-every", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    if args.resume:
        trainer = Trainer.from_checkpoint(args.resume, base_dir=base_dir)
        print("Loaded exact continuation at step %d" % trainer.step_count)
    else:
        config = AppConfig()
        config.environment.wind_changes = args.wind_changes
        config.environment.goal_moves = args.goal_moves
        config.environment.obstacle_switches = args.obstacle_switches
        config.environment.reward_changes = args.reward_changes
        config.environment.wind_period = args.wind_period
        config.environment.reward_period = args.reward_period
        config.environment.target_move_interval = args.target_move_interval
        config.environment.context_switch_interval = args.context_switch_interval
        config.environment.wind_start_step = args.wind_start_step
        config.environment.reward_start_step = args.reward_start_step
        config.environment.target_move_start_step = args.target_move_start_step
        config.environment.context_switch_start_step = args.context_switch_start_step
        config.environment.num_contexts = args.num_contexts
        config.environment.manual_wind_direction = args.wind_direction
        config.environment.w_strength = args.wind_strength
        config.environment.seed = args.seed
        config.environment.width = args.width
        config.environment.height = args.height
        config.environment.obstacle_count = args.obstacles
        config.environment.obstacle_coordinates = None
        config.environment.goal_reached_behavior = args.goal_reached_behavior
        config.agent.algorithm = "sarsa" if args.fixed_alpha else args.algorithm
        config.agent.feature_representation = args.features
        config.agent.planning_steps = args.planning_steps
        config.agent.dyna_plus_kappa = args.dyna_plus_kappa
        trainer = Trainer(config, base_dir=base_dir)

    target = None if args.steps == 0 else trainer.step_count + args.steps
    try:
        while target is None or trainer.step_count < target:
            trainer.run_steps(min(args.report_every, (target - trainer.step_count) if target is not None else args.report_every))
            snapshot = trainer.snapshot()
            print(
                "step={step:.0f} avg_reward={average_reward:.4f} Rbar={reward_rate:.4f} "
                "goals/1k={goals_per_1000_steps:.2f} collision={collision_rate:.3f} alpha={alpha_mean:.3g}".format(
                    **snapshot
                )
            )
    except KeyboardInterrupt:
        print("\nManual stop requested.")
    finally:
        path = trainer.save()
        print("Exact-continuation checkpoint saved to %s" % path)


if __name__ == "__main__":
    main()
