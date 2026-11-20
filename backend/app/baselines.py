"""Comparable demand-forecasting baselines for intermittent sales."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isnan

from .forecast import estimate_demand


@dataclass(frozen=True)
class BaselineEstimate:
    """A point estimate and explicit failure state for one baseline."""

    name: str
    estimate: float | None
    status: str = "estimated"


Baseline = Callable[[Sequence[int]], BaselineEstimate]


@dataclass(frozen=True)
class ForecastCase:
    """One evaluation case containing sales and optional stock observations."""

    sales: tuple[int, ...]
    observations: tuple[dict[str, int | None], ...]
    future_demand: tuple[int, ...] = ()


CaseBaseline = Callable[[ForecastCase], BaselineEstimate]


def sales_mean(sales: Sequence[int]) -> BaselineEstimate:
    """Estimate demand with the mean of all recorded sales."""
    if not sales:
        return BaselineEstimate("sales_mean", None, "insufficient_information")
    return BaselineEstimate("sales_mean", sum(sales) / len(sales))


def moving_average(sales: Sequence[int], window: int = 7) -> BaselineEstimate:
    """Estimate demand with the mean of the most recent ``window`` days."""
    if window < 1:
        raise ValueError("Moving-average window must be at least 1")
    recent = sales[-window:]
    if not recent:
        return BaselineEstimate("moving_average", None, "insufficient_information")
    return BaselineEstimate("moving_average", sum(recent) / len(recent))


def croston(sales: Sequence[int], alpha: float = 0.1) -> BaselineEstimate:
    """Estimate intermittent demand using the classic Croston method."""
    if not 0 < alpha <= 1:
        raise ValueError("Croston alpha must be between 0 and 1")
    nonzero = [(index, value) for index, value in enumerate(sales) if value > 0]
    if not nonzero:
        return BaselineEstimate("croston", 0.0)

    demand = float(nonzero[0][1])
    interval = float(nonzero[0][0] + 1)
    last_index = nonzero[0][0]
    for index, value in nonzero[1:]:
        gap = index - last_index
        demand += alpha * (value - demand)
        interval += alpha * (gap - interval)
        last_index = index
    return BaselineEstimate("croston", demand / interval)


def tsb(
    sales: Sequence[int], alpha: float = 0.1, beta: float = 0.1
) -> BaselineEstimate:
    """Estimate intermittent demand using the Teunter-Syntetos-Babai method."""
    if not 0 < alpha <= 1 or not 0 < beta <= 1:
        raise ValueError("TSB alpha and beta must be between 0 and 1")
    if not sales:
        return BaselineEstimate("tsb", None, "insufficient_information")

    probability = 1.0 if sales[0] > 0 else 0.0
    demand = float(sales[0])
    for value in sales[1:]:
        occurrence = 1.0 if value > 0 else 0.0
        probability += beta * (occurrence - probability)
        if value > 0:
            demand += alpha * (value - demand)
    estimate = probability * demand
    return BaselineEstimate("tsb", estimate)


def poisson_uncensored(sales: Sequence[int]) -> BaselineEstimate:
    """Estimate a Poisson mean while ignoring stockout censoring."""
    result = sales_mean(sales)
    return BaselineEstimate("poisson_uncensored", result.estimate, result.status)


def poisson_censored(
    observations: Sequence[dict[str, int | None]],
) -> BaselineEstimate:
    """Estimate a Poisson mean from sales and known stock availability."""
    result = estimate_demand(observations)
    if result["status"] != "estimated":
        return BaselineEstimate("poisson_censored", None, result["status"])
    estimate = result["estimated_daily_demand"]
    if estimate is None or isnan(float(estimate)):
        return BaselineEstimate("poisson_censored", None, "insufficient_information")
    return BaselineEstimate("poisson_censored", float(estimate))


def default_baselines(sales: Sequence[int]) -> list[BaselineEstimate]:
    """Run all sales-only baselines on the same history."""
    return [
        sales_mean(sales),
        moving_average(sales),
        croston(sales),
        tsb(sales),
        poisson_uncensored(sales),
    ]


def baseline_suite() -> dict[str, CaseBaseline]:
    """Return all six baselines behind one case-based forecasting interface."""
    return {
        "sales_mean": lambda case: sales_mean(case.sales),
        "moving_average": lambda case: moving_average(case.sales),
        "croston": lambda case: croston(case.sales),
        "tsb": lambda case: tsb(case.sales),
        "poisson_uncensored": lambda case: poisson_uncensored(case.sales),
        "poisson_censored": lambda case: poisson_censored(case.observations),
    }
