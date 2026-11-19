from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional


def poisson_quantile(mean: float, probability: float) -> int:
    if mean <= 0:
        return 0
    cumulative = math.exp(-mean)
    term = cumulative
    value = 0
    while cumulative < probability:
        value += 1
        term *= mean / value
        cumulative += term
        if value > 10000:
            break
    return value


def estimate_demand(observations: Iterable[Mapping[str, object]]) -> dict:
    rows = list(observations)
    if not rows:
        raise ValueError("At least one observation is required")

    usable = []
    censored_count = 0
    for row in rows:
        sales = row.get("sales", row.get("sold", row.get("demand", None)))
        available = row.get("available_stock", row.get("available", None))
        try:
            sales_value = float(sales)
        except (TypeError, ValueError):
            raise ValueError("Sales must be a non-negative number")
        if sales_value < 0 or not math.isfinite(sales_value):
            raise ValueError("Sales must be a non-negative number")

        if available is None:
            continue
        try:
            available_value = float(available)
        except (TypeError, ValueError):
            raise ValueError("Available stock must be a non-negative number or null")
        if available_value < 0 or not math.isfinite(available_value):
            raise ValueError("Available stock must be a non-negative number or null")
        if sales_value > available_value:
            raise ValueError("Recorded sales cannot exceed available stock")

        usable.append((sales_value, available_value))
        if sales_value == available_value:
            censored_count += 1

    if not usable:
        return {
            "estimated_daily_demand": 0.0,
            "p10": 0,
            "p50": 0,
            "p90": 0,
            "censored_observations": 0,
            "usable_observations": 0,
            "warning": "Unknown or missing stock availability: no known observations were supplied.",
        }

    max_sale = max(sales for sales, _ in usable)
    max_lambda = max(10.0, max_sale * 4.0)

    def log_likelihood(mean: float) -> float:
        total = 0.0
        for sales, available in usable:
            if sales < available:
                total += sales * math.log(mean) - mean - sum(math.log(n) for n in range(1, int(sales) + 1))
            elif available == 0:
                continue
            else:
                # P(D >= available) for a Poisson count model.
                cumulative = 0.0
                term = math.exp(-mean)
                cumulative = term
                for k in range(1, int(available)):
                    term *= mean / k
                    cumulative += term
                total += math.log(max(1.0 - cumulative, 1e-12))
        return total

    best_mean = 0.01
    best_log_likelihood = log_likelihood(best_mean)
    step = 0.01
    for index in range(1, int(max_lambda / step) + 1):
        candidate = index * step
        candidate_likelihood = log_likelihood(candidate)
        if candidate_likelihood > best_log_likelihood:
            best_mean = candidate
            best_log_likelihood = candidate_likelihood

    censor_share = censored_count / len(usable)
    warning = None
    if len(usable) < 7:
        warning = "Sparse history: estimate has high uncertainty."
    elif censor_share >= 0.5:
        warning = "Many observations are stockout-censored; verify stock availability."

    return {
        "estimated_daily_demand": round(best_mean, 2),
        "p10": poisson_quantile(best_mean, 0.10),
        "p50": poisson_quantile(best_mean, 0.50),
        "p90": poisson_quantile(best_mean, 0.90),
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
