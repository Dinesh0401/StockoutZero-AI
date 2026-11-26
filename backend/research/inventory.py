"""Sequential inventory-cost simulation for baseline decisions."""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from app.baselines import CaseBaseline, ForecastCase


@dataclass(frozen=True)
class InventoryCosts:
    """Unit costs used by the sequential inventory simulation."""

    holding: float = 0.10
    stockout: float = 2.00
    purchase: float = 0.50


@dataclass(frozen=True)
class InventorySimulation:
    """Aggregated service and cost outcomes."""

    total_cost: float
    stockout_units: int
    fulfilled_units: int
    fill_rate: float
    orders: int
    purchased_units: int = 0


@dataclass(frozen=True)
class RollingPolicyOutcome:
    """One trial-level rolling inventory result and forecast status."""

    trial: int
    simulation: InventorySimulation
    forecasts: tuple[float, ...]
    abstentions: int
    fallback_usage: int


@dataclass(frozen=True)
class RollingPolicyComparison:
    """Paired rolling outcomes for one baseline."""

    name: str
    outcomes: tuple[RollingPolicyOutcome, ...]


@dataclass(frozen=True)
class PairedPolicyMetric:
    """Paired cost or service differences against a reference policy."""

    reference: str
    baseline: str
    mean_difference: float
    paired_trials: int
    bootstrap_interval_90: tuple[float, float] | None


def simulate_inventory(
    demand: Sequence[int],
    reorder_points: Sequence[int],
    costs: InventoryCosts | None = None,
    initial_stock: int = 0,
    lead_time_days: int = 0,
    case_pack: int = 1,
) -> InventorySimulation:
    """Simulate daily replenishment decisions with outstanding purchase orders."""
    costs = costs or InventoryCosts()
    if len(demand) != len(reorder_points):
        raise ValueError("Demand and reorder points must have equal lengths")
    if initial_stock < 0:
        raise ValueError("Initial stock cannot be negative")
    if lead_time_days < 0:
        raise ValueError("Lead time cannot be negative")
    if case_pack < 1:
        raise ValueError("Case pack must be at least 1")

    stock = initial_stock
    arrivals: dict[int, int] = {}
    total_cost = 0.0
    stockout_units = 0
    fulfilled_units = 0
    orders = 0
    purchased_units = 0
    for day, (actual, reorder_point) in enumerate(
        zip(demand, reorder_points, strict=True)
    ):
        if actual < 0 or reorder_point < 0:
            raise ValueError("Demand and reorder points cannot be negative")
        stock += arrivals.pop(day, 0)
        inventory_position = stock + sum(
            quantity for arrival_day, quantity in arrivals.items() if arrival_day > day
        )
        if inventory_position < reorder_point:
            shortage = reorder_point - inventory_position
            ordered = ceil(shortage / case_pack) * case_pack
            purchased_units += ordered
            if lead_time_days == 0:
                stock += ordered
            else:
                arrivals.setdefault(day + lead_time_days, 0)
                arrivals[day + lead_time_days] += ordered
            total_cost += ordered * costs.purchase
            orders += int(ordered > 0)
        fulfilled = min(stock, actual)
        lost = actual - fulfilled
        stock -= fulfilled
        fulfilled_units += fulfilled
        stockout_units += lost
        total_cost += stock * costs.holding + lost * costs.stockout

    total_demand = fulfilled_units + stockout_units
    return InventorySimulation(
        total_cost=total_cost,
        stockout_units=stockout_units,
        fulfilled_units=fulfilled_units,
        fill_rate=fulfilled_units / total_demand if total_demand else 1.0,
        orders=orders,
        purchased_units=purchased_units,
    )


def simulate_forecast_policy(
    demand: Sequence[int],
    daily_forecast: Sequence[float],
    lead_time_days: int,
    safety_days: float = 0.0,
    costs: InventoryCosts | None = None,
    initial_stock: int = 0,
    case_pack: int = 1,
) -> InventorySimulation:
    """Convert rolling forecasts into lead-time reorder points and simulate them."""
    costs = costs or InventoryCosts()
    if len(demand) != len(daily_forecast):
        raise ValueError("Demand and forecasts must have equal lengths")
    if lead_time_days < 0 or safety_days < 0:
        raise ValueError("Lead time and safety days cannot be negative")
    if case_pack < 1:
        raise ValueError("Case pack must be at least 1")
    reorder_points = [
        (
            ceil(max(0.0, forecast) * (lead_time_days + safety_days) / case_pack)
            * case_pack
        )
        for forecast in daily_forecast
    ]
    return simulate_inventory(
        demand,
        reorder_points,
        costs=costs,
        initial_stock=initial_stock,
        lead_time_days=lead_time_days,
        case_pack=case_pack,
    )


def compare_baseline_inventory_policies(
    cases: Sequence[ForecastCase],
    baseline_names: Sequence[str],
    baseline_suite: dict[str, CaseBaseline],
    lead_time_days: int = 7,
    costs: InventoryCosts | None = None,
    case_pack: int = 1,
) -> dict[str, InventorySimulation]:
    """Generate each baseline's forecast and simulate its replenishment outcome."""
    costs = costs or InventoryCosts()
    warnings.warn(
        "compare_baseline_inventory_policies is deprecated; use "
        "compare_rolling_baseline_policies for paired trial-level results.",
        DeprecationWarning,
        stacklevel=2,
    )
    comparisons: dict[str, InventorySimulation] = {}
    for name in baseline_names:
        baseline = baseline_suite[name]
        simulations: list[InventorySimulation] = []
        for case in cases:
            estimate = baseline(case)
            if estimate.status != "estimated" or estimate.estimate is None:
                continue
            demand = list(case.future_demand)
            daily_forecast = [estimate.estimate] * len(demand)
            simulations.append(
                simulate_forecast_policy(
                    demand,
                    daily_forecast,
                    lead_time_days,
                    costs=costs,
                    initial_stock=0,
                    case_pack=case_pack,
                )
            )
        if simulations:
            comparisons[name] = InventorySimulation(
                total_cost=sum(value.total_cost for value in simulations)
                / len(simulations),
                stockout_units=sum(value.stockout_units for value in simulations),
                fulfilled_units=sum(value.fulfilled_units for value in simulations),
                fill_rate=sum(value.fill_rate for value in simulations)
                / len(simulations),
                orders=sum(value.orders for value in simulations),
            )
    return comparisons


def compare_rolling_baseline_policies(
    cases: Sequence[ForecastCase],
    baseline_names: Sequence[str],
    baseline_suite: dict[str, CaseBaseline],
    lead_time_days: int = 7,
    case_pack: int = 1,
    costs: InventoryCosts | None = None,
    abstention_fallback: str = "zero",
) -> dict[str, RollingPolicyComparison]:
    """Run every model on every trial with rolling, decision-time forecasts."""
    costs = costs or InventoryCosts()
    if lead_time_days < 0:
        raise ValueError("Lead time cannot be negative")
    if case_pack < 1:
        raise ValueError("Case pack must be at least 1")
    if abstention_fallback not in {"zero", "conservative"}:
        raise ValueError("Unsupported abstention fallback")

    comparisons: dict[str, RollingPolicyComparison] = {}
    for name in baseline_names:
        baseline = baseline_suite[name]
        trial_outcomes: list[RollingPolicyOutcome] = []
        for trial, case in enumerate(cases):
            sales_history = list(case.sales)
            stock_history = list(case.observations)
            stock = 0
            arrivals: dict[int, int] = {}
            forecasts: list[float] = []
            abstentions = 0
            fallback_usage = 0
            total_cost = 0.0
            stockout_units = 0
            fulfilled_units = 0
            orders = 0
            purchased_units = 0

            for day, latent_demand in enumerate(case.future_demand):
                stock += arrivals.pop(day, 0)
                rolling_case = ForecastCase(
                    sales=tuple(sales_history),
                    observations=tuple(stock_history),
                )
                estimate = baseline(rolling_case)
                if estimate.status == "estimated" and estimate.estimate is not None:
                    forecast = max(0.0, estimate.estimate)
                else:
                    abstentions += 1
                    fallback_usage += 1
                    if abstention_fallback == "conservative":
                        known_sales = [value for value in sales_history if value > 0]
                        forecast = (
                            sum(known_sales) / len(known_sales) if known_sales else 1.0
                        )
                    else:
                        forecast = 0.0

                forecasts.append(forecast)
                inventory_position = stock + sum(
                    quantity
                    for arrival_day, quantity in arrivals.items()
                    if arrival_day > day
                )
                target_inventory = (
                    ceil(forecast * lead_time_days / case_pack) * case_pack
                )
                stock_gap = max(target_inventory - inventory_position, 0)
                raw_order = ceil(stock_gap / case_pack) * case_pack
                if raw_order:
                    orders += 1
                    purchased_units += raw_order
                    total_cost += raw_order * costs.purchase
                    if lead_time_days == 0:
                        stock += raw_order
                    else:
                        arrivals[day + lead_time_days] = (
                            arrivals.get(day + lead_time_days, 0) + raw_order
                        )

                fulfilled = min(stock, latent_demand)
                lost = latent_demand - fulfilled
                stock -= fulfilled
                fulfilled_units += fulfilled
                stockout_units += lost
                total_cost += stock * costs.holding + lost * costs.stockout

                available_before_demand = stock + fulfilled
                sales_history.append(fulfilled)
                stock_history.append(
                    {
                        "sales": fulfilled,
                        "available_stock": available_before_demand,
                    }
                )

            total_demand = fulfilled_units + stockout_units
            simulation = InventorySimulation(
                total_cost=total_cost,
                stockout_units=stockout_units,
                fulfilled_units=fulfilled_units,
                fill_rate=(fulfilled_units / total_demand if total_demand else 1.0),
                orders=orders,
                purchased_units=purchased_units,
            )
            trial_outcomes.append(
                RollingPolicyOutcome(
                    trial=trial,
                    simulation=simulation,
                    forecasts=tuple(forecasts),
                    abstentions=abstentions,
                    fallback_usage=fallback_usage,
                )
            )
        comparisons[name] = RollingPolicyComparison(
            name=name,
            outcomes=tuple(trial_outcomes),
        )
    return comparisons


def paired_policy_cost_differences(
    comparisons: dict[str, RollingPolicyComparison],
    reference: str,
) -> list[PairedPolicyMetric]:
    """Report baseline-minus-reference paired total-cost differences."""
    if reference not in comparisons:
        raise ValueError("Reference policy was not evaluated")
    reference_outcomes = {
        outcome.trial: outcome for outcome in comparisons[reference].outcomes
    }
    metrics: list[PairedPolicyMetric] = []
    for name, comparison in comparisons.items():
        if name == reference:
            continue
        differences = [
            outcome.simulation.total_cost
            - reference_outcomes[outcome.trial].simulation.total_cost
            for outcome in comparison.outcomes
            if outcome.trial in reference_outcomes
        ]
        differences.sort()
        bootstrap_interval = None
        if differences:
            resampled_means: list[float] = []
            rng = random.Random(42)
            for _ in range(1000):
                sample = [rng.choice(differences) for _ in differences]
                resampled_means.append(sum(sample) / len(sample))
            resampled_means.sort()
            bootstrap_interval = (
                resampled_means[50],
                resampled_means[949],
            )
        metrics.append(
            PairedPolicyMetric(
                reference=reference,
                baseline=name,
                mean_difference=(
                    sum(differences) / len(differences) if differences else 0.0
                ),
                paired_trials=len(differences),
                bootstrap_interval_90=bootstrap_interval,
            )
        )
    return metrics
