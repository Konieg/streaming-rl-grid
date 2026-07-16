"""P0.2: tabular differential Sarsa(0) control."""

import argparse
from pathlib import Path

from .protocol import DEFAULT_SEEDS, DEFAULT_STEPS
from .runner import run_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "p0_2_control",
    )
    args = parser.parse_args()
    run_matrix("control", args.steps, args.seeds, args.output)


if __name__ == "__main__":
    main()

