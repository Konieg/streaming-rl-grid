import argparse

from stream_rl_grid.gui import main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continual Windy Grid GUI")
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Run exactly this many steps, then stop automatically; 0 runs until stopped.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    main(fixed_steps=args.steps)
