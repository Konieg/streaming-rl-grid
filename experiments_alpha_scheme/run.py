"""Command-line entry point for the formal E1–E8 experiment runner."""

import argparse
from typing import Tuple

from .runner import ExperimentConfig, PRESETS, run_experiment


def _groups(value: str) -> Tuple[str, ...]:
    return tuple(group.strip() for group in value.split(",") if group.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run streaming step-size experiments on GridWorld")
    parser.add_argument("--experiment", choices=sorted(PRESETS), default="e1")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--method", choices=("scalar", "grouped", "tidbd"), default="scalar")
    parser.add_argument("--profile", type=str)
    parser.add_argument("--task", choices=("supervised", "td"))
    parser.add_argument("--target", choices=("reward", "collision", "goal"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--obstacles", type=int, default=3)
    parser.add_argument("--num-contexts", type=int, default=2)
    parser.add_argument("--wind-period", type=int, default=400)
    parser.add_argument("--target-move-interval", type=int, default=300)
    parser.add_argument("--context-switch-interval", type=int, default=500)
    parser.add_argument("--noise-features", type=int, default=0)
    parser.add_argument("--oracle-context", action="store_true")
    parser.add_argument("--ablate-groups", type=_groups, default=())
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--output", type=str)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        experiment=args.experiment,
        steps=args.steps,
        seeds=tuple(args.seeds),
        method=args.method,
        profile=args.profile,
        task=args.task,
        target=args.target,
        initial_alpha=args.alpha,
        theta=args.theta,
        width=args.width,
        height=args.height,
        obstacles=args.obstacles,
        num_contexts=args.num_contexts,
        wind_period=args.wind_period,
        target_move_interval=args.target_move_interval,
        context_switch_interval=args.context_switch_interval,
        noise_features=args.noise_features,
        oracle_context=args.oracle_context,
        ablate_groups=args.ablate_groups,
        report_every=args.report_every,
        output=args.output,
    )
    output = run_experiment(config)
    print("Experiment written to %s" % output)


if __name__ == "__main__":
    main()
