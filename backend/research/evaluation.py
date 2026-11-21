"""Reproducible evaluation metrics for demand baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.baselines import Baseline, CaseBaseline, ForecastCase


@dataclass(frozen=True)
class EvaluationSummary:
    """Paired evaluation metrics for one forecasting method."""

    name: str
    trials: int
    successful_trials: int
    failure_rate: float
    mae: float | None
    bootstrap_mae_interval_90: tuple[float, float] | None
    interval_coverage: float | None = None


@dataclass(frozen=True)
class SuiteEvaluation:
    """Evaluation summaries for every baseline on identical cases."""

    summaries: dict[str, EvaluationSummary]


def _bootstrap_interval(
    errors: Sequence[float],
    seed: int,
    samples: int = 1000,
) -> tuple[float, float] | None:
    if not errors:
        return None
    import random

    rng = random.Random(seed)
    means = [
        sum(rng.choice(errors) for _ in errors) / len(errors) for _ in range(samples)
    ]
    means.sort()
    return (means[int(0.05 * samples)], means[int(0.95 * samples) - 1])


def evaluate_baseline(
    baseline: Baseline,
    histories: Sequence[Sequence[int]],
    true_means: Sequence[float],
    seed: int = 42,
) -> EvaluationSummary:
    """Evaluate one baseline while reporting failures separately from accuracy."""
    if len(histories) != len(true_means):
        raise ValueError("Histories and true means must have equal lengths")
    if not histories:
        raise ValueError("At least one history is required")

    errors: list[float] = []
    name = "unknown"
    for history, true_mean in zip(histories, true_means, strict=True):
        result = baseline(history)
        name = result.name
        if result.status == "estimated" and result.estimate is not None:
            errors.append(abs(result.estimate - true_mean))

    trials = len(histories)
    successful = len(errors)
    return EvaluationSummary(
        name=name,
        trials=trials,
        successful_trials=successful,
        failure_rate=(trials - successful) / trials,
        mae=sum(errors) / successful if successful else None,
        bootstrap_mae_interval_90=_bootstrap_interval(errors, seed),
    )


def evaluate_suite(
    baselines: dict[str, CaseBaseline],
    cases: Sequence[ForecastCase],
    future_means: Sequence[float],
    seed: int = 42,
    parameter_means: Sequence[float] | None = None,
) -> SuiteEvaluation:
    """Evaluate all baselines with identical trials and failure accounting."""
    if len(cases) != len(future_means):
        raise ValueError("Cases and future means must have equal lengths")
    if parameter_means is not None and len(parameter_means) != len(cases):
        raise ValueError("Cases and parameter means must have equal lengths")
    if not cases:
        raise ValueError("At least one case is required")

    summaries: dict[str, EvaluationSummary] = {}
    for offset, (name, baseline) in enumerate(baselines.items()):
        errors: list[float] = []
        results = [baseline(case) for case in cases]
        for result, true_mean in zip(results, future_means, strict=True):
            if result.status == "estimated" and result.estimate is not None:
                errors.append(abs(result.estimate - true_mean))
        trials = len(cases)
        interval_hits = [
            result.interval[0] <= true_mean <= result.interval[1]
            for result, true_mean in zip(
                results,
                parameter_means if parameter_means is not None else future_means,
                strict=True,
            )
            if result.interval is not None
        ]
        summaries[name] = EvaluationSummary(
            name=name,
            trials=trials,
            successful_trials=len(errors),
            failure_rate=(trials - len(errors)) / trials,
            mae=sum(errors) / len(errors) if errors else None,
            bootstrap_mae_interval_90=_bootstrap_interval(errors, seed + offset),
            interval_coverage=(
                sum(interval_hits) / len(interval_hits) if interval_hits else None
            ),
        )
    return SuiteEvaluation(summaries)


def paired_mae_difference(
    first: CaseBaseline,
    second: CaseBaseline,
    cases: Sequence[ForecastCase],
    true_means: Sequence[float],
    seed: int = 42,
) -> tuple[float | None, tuple[float, float] | None]:
    """Compare two models on paired cases with a bootstrap interval."""
    if len(cases) != len(true_means):
        raise ValueError("Cases and true means must have equal lengths")
    differences: list[float] = []
    for case, true_mean in zip(cases, true_means, strict=True):
        first_result = first(case)
        second_result = second(case)
        if (
            first_result.status == "estimated"
            and second_result.status == "estimated"
            and first_result.estimate is not None
            and second_result.estimate is not None
        ):
            differences.append(
                abs(first_result.estimate - true_mean)
                - abs(second_result.estimate - true_mean)
            )
    return (
        sum(differences) / len(differences) if differences else None,
        _bootstrap_interval(differences, seed),
    )
