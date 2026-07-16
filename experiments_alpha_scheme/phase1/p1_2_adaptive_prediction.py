"""P1.2: D=55 differential TD(0) prediction with per-feature TIDBD."""

import argparse
from pathlib import Path

from experiments.lfa_runner import adaptive_mechanism, run_matrix
from experiments.phase0.protocol import DEFAULT_SEEDS, DEFAULT_STEPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "p1_2_adaptive")
    args = parser.parse_args()
    run_matrix("phase1", "p1_2_adaptive_prediction", "prediction", adaptive_mechanism(), args.steps, args.seeds, args.output)


if __name__ == "__main__":
    main()

