"""Headless streaming trainer; Ctrl+C is a supported manual stopping mechanism."""

import argparse
from pathlib import Path

from .config import ALGORITHMS, REPRESENTATIONS, AppConfig, PROFILES
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming differential Sarsa control algorithms")
    parser.add_argument("--resume", type=str, help="Checkpoint to continue exactly")
    parser.add_argument("--profile", choices=PROFILES, default="combined")
    parser.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl+C")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--obstacles", type=int, default=8)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="tidbd")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="tabular-one-hot")
    parser.add_argument(
        "--fixed-alpha", action="store_true", help="Force the fixed-step Sarsa baseline"
    )
    parser.add_argument("--epsilon-kappa", type=float, default=0.01)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-max", type=float, default=0.30)
    parser.add_argument("--epsilon-scale", type=float, default=0.10)
    parser.add_argument("--epsilon-u-ref", type=float, default=1.0)
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
        config.environment.profile = args.profile
        config.environment.seed = args.seed
        config.environment.width = args.width
        config.environment.height = args.height
        config.environment.obstacle_count = args.obstacles
        config.environment.obstacle_coordinates = None
        config.agent.algorithm = "sarsa" if args.fixed_alpha else args.algorithm
        config.agent.representation = args.representation
        config.agent.use_tidbd = config.agent.algorithm in ("tidbd", "expected_sarsa_tidbd")
        config.agent.adaptive_epsilon_kappa = args.epsilon_kappa
        config.agent.adaptive_epsilon_min = args.epsilon_min
        config.agent.adaptive_epsilon_max = args.epsilon_max
        config.agent.adaptive_epsilon_scale = args.epsilon_scale
        config.agent.adaptive_epsilon_u_ref = args.epsilon_u_ref
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
