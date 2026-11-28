"""Command-line synthetic calibration report runner."""

from __future__ import annotations

import argparse

from research.calibration import save_calibration_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="calibration-report.json")
    parser.add_argument("--history-lengths", nargs="+", type=int, default=(30, 90))
    parser.add_argument("--seeds", nargs="+", type=int, default=(7, 19, 42))
    args = parser.parse_args()

    reports = save_calibration_report(
        args.output,
        tuple(args.history_lengths),
        tuple(args.seeds),
    )
    for report in reports:
        print(
            f"{report.history_length}-day: "
            f"predictive coverage={report.mean_predictive_coverage}, "
            f"hierarchical MAE={report.mean_hierarchical_mae}, "
            f"sales MAE={report.mean_sales_mae}"
        )
        print(report.warning)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
