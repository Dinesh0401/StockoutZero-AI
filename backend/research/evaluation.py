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
    true_means: Sequence[float],
    seed: int = 42,
) -> SuiteEvaluation:
    """Evaluate all baselines with identical trials and failure accounting."""
    if len(cases) != len(true_means):
        raise ValueError("Cases and true means must have equal lengths")
    if not cases:
        raise ValueError("At least one case is required")

    summaries: dict[str, EvaluationSummary] = {}
    for offset, (name, baseline) in enumerate(baselines.items()):
        errors: list[float] = []
        for case, true_mean in zip(cases, true_means, strict=True):
            result = baseline(case)
            if result.status == "estimated" and result.estimate is not None:
                errors.append(abs(result.estimate - true_mean))
        trials = len(cases)
        summaries[name] = EvaluationSummary(
            name=name,
            trials=trials,
            successful_trials=len(errors),
            failure_rate=(trials - len(errors)) / trials,
            mae=sum(errors) / len(errors) if errors else None,
            bootstrap_mae_interval_90=_bootstrap_interval(errors, seed + offset),
        )
    return SuiteEvaluation(summaries)
