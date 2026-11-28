"""Command-line runner for the reproducible Phase 3A validation study."""

from __future__ import annotations

import argparse
from pathlib import Path

from research.hierarchical_demand import run_reproducible_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="phase3a_study_results.json",
        help="JSON output path; a CSV summary is written alongside it.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(1, 2, 3, 4, 5),
        help="Independent synthetic data seeds.",
    )
    parser.add_argument(
        "--history-lengths",
        nargs="+",
        type=int,
        default=(30, 90),
        help="Training-history lengths in days.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Chronological fraction used for training.",
    )
    arguments = parser.parse_args()

    records = run_reproducible_study(
        output_path=arguments.output,
        seeds=arguments.seeds,
        history_lengths=arguments.history_lengths,
        train_fraction=arguments.train_fraction,
    )
    output = Path(arguments.output)
    print(f"Completed {len(records)} runs.")
    print(f"JSON: {output}")
    print(f"CSV:  {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
