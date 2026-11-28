"""Synthetic interval-calibration and model-monitoring report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from research.hierarchical_demand import (
    run_hierarchical_validation_matrix,
)


@dataclass(frozen=True)
class CalibrationReport:
    """Coverage and MAE summary from synthetic validation runs."""

    history_length: int
    seed_count: int
    mean_predictive_coverage: float | None
    mean_hierarchical_mae: float | None
    mean_sales_mae: float | None
    warning: str


def build_calibration_report(
    history_lengths: tuple[int, ...] = (30, 90),
    seeds: tuple[int, ...] = (7, 19, 42),
) -> tuple[CalibrationReport, ...]:
    """Build a reproducible calibration report, not a real-merchant claim."""
    if not history_lengths or not seeds:
        raise ValueError("history_lengths and seeds must be non-empty")
    matrix = run_hierarchical_validation_matrix(history_lengths, seeds)
    reports: list[CalibrationReport] = []
    for history_length in history_lengths:
        runs = [
            run
            for name, run in matrix.items()
            if name.startswith(f"{history_length}d_seed")
        ]
        coverages = [
            value
            for run in runs
            if (value := run.predictive_interval_coverage) is not None
        ]
        hierarchical_maes = [
            value for run in runs if (value := run.hierarchical_mae) is not None
        ]
        sales_maes = [
            value
            for run in runs
            if (value := run.baseline_maes.get("sales_mean")) is not None
        ]
        mean_coverage = mean(coverages) if coverages else None
        warning = (
            "Synthetic coverage only; no real-merchant calibration claim."
            if mean_coverage is None or abs(mean_coverage - 0.90) <= 0.05
            else "Synthetic predictive coverage is not calibrated at the 90% target."
        )
        reports.append(
            CalibrationReport(
                history_length=history_length,
                seed_count=len(runs),
                mean_predictive_coverage=mean_coverage,
                mean_hierarchical_mae=(
                    mean(hierarchical_maes) if hierarchical_maes else None
                ),
                mean_sales_mae=mean(sales_maes) if sales_maes else None,
                warning=warning,
            )
        )
    return tuple(reports)


def save_calibration_report(
    output_path: str,
    history_lengths: tuple[int, ...] = (30, 90),
    seeds: tuple[int, ...] = (7, 19, 42),
) -> tuple[CalibrationReport, ...]:
    reports = build_calibration_report(history_lengths, seeds)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(report) for report in reports], indent=2),
        encoding="utf-8",
    )
    return reports
