"""P1.1: D=55 differential TD(0) prediction with fixed alpha."""

import argparse
from pathlib import Path

from experiments.lfa_runner import fixed_mechanisms, run_matrix
from experiments.phase0.protocol import DEFAULT_SEEDS, DEFAULT_STEPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "p1_1_fixed")
    args = parser.parse_args()
    run_matrix("phase1", "p1_1_fixed_prediction", "prediction", fixed_mechanisms(), args.steps, args.seeds, args.output)


if __name__ == "__main__":
    main()

