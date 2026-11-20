from __future__ import annotations

import math
from typing import Iterable, Mapping

from scipy.optimize import minimize_scalar
from scipy.stats import poisson


def poisson_quantile(mean: float, probability: float) -> int:
    if mean <= 0:
        return 0
    return int(poisson.ppf(probability, mean))


def insufficient_result(
    censored_count: int = 0,
    usable_observations: int = 0,
    warning: str = "Insufficient information.",
) -> dict:
    return {
        "status": "insufficient_information",
        "estimated_daily_demand": None,
        "p10": None,
        "p50": None,
        "p90": None,
        "mean_interval_90": None,
        "censored_observations": censored_count,
        "usable_observations": usable_observations,
        "warning": warning,
    }


def estimate_demand(observations: Iterable[Mapping[str, object]]) -> dict:
    rows = list(observations)
    if not rows:
        raise ValueError("At least one observation is required")

    usable = []
    censored_count = 0
    informative_count = 0

    for row in rows:
        sales = row.get("sales", row.get("sold", row.get("demand")))
        available = row.get("available_stock", row.get("available"))

        try:
            sales_value = float(sales)
        except (TypeError, ValueError):
            raise ValueError("Sales must be a non-negative whole number")
        if not math.isfinite(sales_value) or sales_value < 0 or not sales_value.is_integer():
            raise ValueError("Sales must be a non-negative whole number")

        if available is None:
            continue

        try:
            available_value = float(available)
        except (TypeError, ValueError):
            raise ValueError("Available stock must be a non-negative whole number or null")
        if (
            not math.isfinite(available_value)
            or available_value < 0
            or not available_value.is_integer()
        ):
            raise ValueError("Available stock must be a non-negative whole number or null")
        if sales_value > available_value:
            raise ValueError("Recorded sales cannot exceed available stock")

        # A known zero-availability record gives no demand information.
        if available_value == 0:
            continue

        usable.append((int(sales_value), int(available_value)))
        if sales_value == available_value and sales_value > 0:
            censored_count += 1
        elif sales_value < available_value:
            informative_count += 1

    if not usable:
        return insufficient_result(
            warning="Insufficient information: no usable positive-stock observations were supplied.",
        )

    if informative_count == 0:
        return insufficient_result(
            censored_count=censored_count,
            usable_observations=len(usable),
            warning="Insufficient information: all usable observations are censored, so latent demand is not identifiable.",
        )

    if all(sales == 0 for sales, _ in usable):
        return insufficient_result(
            usable_observations=len(usable),
            warning="Insufficient information: zero sales alone do not support a confident positive demand estimate.",
        )

    max_sale = max(sales for sales, _ in usable)
    upper_bound = max(10.0, max_sale * 4.0)

    def log_likelihood(mean: float) -> float:
        mean = max(mean, 1e-9)
        total = 0.0
        for sales, available in usable:
            if sales < available:
                total += float(poisson.logpmf(sales, mean))
            else:
                total += float(poisson.logsf(available - 1, mean))
        return total

    result = minimize_scalar(
        lambda mean: -log_likelihood(mean),
        bounds=(1e-6, upper_bound),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError("Demand likelihood optimization failed")

    best_mean = max(float(result.x), 1e-6)

    # Approximate parameter uncertainty; this is not a demand prediction interval.
    observed_information = max(len(usable) / max(best_mean, 1e-6), 1.0)
    standard_error = math.sqrt(1.0 / observed_information)
    mean_low = max(0.0, best_mean - 1.645 * standard_error)
    mean_high = best_mean + 1.645 * standard_error

    censor_share = censored_count / len(usable)
    warning = None
    if len(usable) < 7:
        warning = "Sparse history: estimate has high parameter uncertainty."
    elif censor_share >= 0.5:
        warning = "Many observations are stockout-censored; verify stock availability."

    return {
        "status": "estimated",
        "estimated_daily_demand": round(best_mean, 2),
        "p10": poisson_quantile(best_mean, 0.10),
        "p50": poisson_quantile(best_mean, 0.50),
        "p90": poisson_quantile(best_mean, 0.90),
        "mean_interval_90": [round(mean_low, 2), round(mean_high, 2)],
        "censored_observations": censored_count,
        "usable_observations": len(usable),
        "warning": warning,
    }


def recommend_reorder(
    observations: Iterable[Mapping[str, object]],
    lead_time_days: int = 7,
    on_hand: int = 0,
    case_pack: int = 1,
) -> dict:
    if lead_time_days < 1:
        raise ValueError("Lead time must be at least 1 day")
    if on_hand < 0:
        raise ValueError("On-hand stock cannot be negative")
    if case_pack < 1:
        raise ValueError("Case pack must be at least 1")

    estimate = estimate_demand(observations)
    if estimate["status"] != "estimated":
        return {
            **estimate,
            "lead_time_days": lead_time_days,
            "on_hand": on_hand,
            "case_pack": case_pack,
            "suggested_units": None,
        }

    daily_mean = float(estimate["estimated_daily_demand"])
    period_mean = daily_mean * lead_time_days
    target = poisson_quantile(period_mean, 0.90)
    raw_units = max(0, target - on_hand)
    suggested = math.ceil(raw_units / case_pack) * case_pack

    return {
        **estimate,
        "lead_time_days": lead_time_days,
        "on_hand": on_hand,
        "case_pack": case_pack,
        "suggested_units": suggested,
    }
