"""Cost-aware, one-period stock-audit value-of-information analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from scipy.stats import nbinom

from research.inventory import InventoryCosts
from research.uncertain_inventory import InventoryDayEstimate


@dataclass(frozen=True)
class AuditRecommendation:
    """Value-of-information result for a proposed physical stock audit."""

    status: str
    expected_benefit: float | None
    audit_cost: float
    net_expected_benefit: float | None
    expected_loss_without_audit: float | None
    expected_loss_with_audit: float | None
    stockout_probability: float | None
    demand_uncertainty: float | None
    inventory_uncertainty: float | None
    optimal_order_without_audit: int | None
    expected_order_after_audit: float | None
    order_by_audited_stock: tuple[int, ...] | None
    warning: str | None = None


def _expected_unit_loss(
    target_stock: int,
    demand_mean: float,
    dispersion: float,
    costs: InventoryCosts,
) -> float:
    """Expected holding and lost-sales cost at a fixed available-stock target."""
    probability = dispersion / (dispersion + demand_mean)
    if target_stock == 0:
        expected_holding = 0.0
        expected_stockout = demand_mean
    else:
        expected_holding = target_stock * float(
            nbinom.cdf(target_stock - 1, dispersion, probability)
        ) - demand_mean * float(
            nbinom.cdf(target_stock - 2, dispersion + 1, probability)
        )
        expected_stockout = demand_mean * float(
            nbinom.sf(target_stock - 1, dispersion + 1, probability)
        ) - target_stock * float(nbinom.sf(target_stock, dispersion, probability))
    return costs.holding * max(expected_holding, 0.0) + costs.stockout * max(
        expected_stockout, 0.0
    )


def _action_losses(
    inventory: int,
    demand_mean: float,
    dispersion: float,
    costs: InventoryCosts,
    max_order_units: int,
) -> tuple[float, ...]:
    return tuple(
        order * costs.purchase
        + _expected_unit_loss(inventory + order, demand_mean, dispersion, costs)
        for order in range(max_order_units + 1)
    )


def recommend_stock_audit(
    inventory: InventoryDayEstimate | None,
    demand_mean: float,
    demand_dispersion: float,
    audit_cost: float,
    costs: InventoryCosts | None = None,
    max_order_units: int = 100,
) -> AuditRecommendation:
    """Recommend an audit when expected decision improvement exceeds its cost.

    The count is assumed to be exact and obtained before an order decision.
    The decision horizon is one demand period with no lead-time constraint.
    After observing the count, the policy can choose any whole-unit order up to
    ``max_order_units``. Costs are expected purchase, holding, and lost-sales
    costs under a fixed NB2 demand distribution.
    """
    costs = costs or InventoryCosts()
    if not isfinite(demand_mean) or demand_mean <= 0:
        raise ValueError("demand_mean must be finite and positive")
    if not isfinite(demand_dispersion) or demand_dispersion <= 0:
        raise ValueError("demand_dispersion must be finite and positive")
    if not isfinite(audit_cost) or audit_cost < 0:
        raise ValueError("audit_cost must be finite and non-negative")
    if (
        isinstance(max_order_units, bool)
        or not isinstance(max_order_units, int)
        or max_order_units < 0
    ):
        raise ValueError("max_order_units must be a non-negative whole number")
    for name, value in (
        ("holding", costs.holding),
        ("stockout", costs.stockout),
        ("purchase", costs.purchase),
    ):
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} cost must be finite and non-negative")

    if inventory is None:
        return AuditRecommendation(
            status="insufficient_information",
            expected_benefit=None,
            audit_cost=audit_cost,
            net_expected_benefit=None,
            expected_loss_without_audit=None,
            expected_loss_with_audit=None,
            stockout_probability=None,
            demand_uncertainty=None,
            inventory_uncertainty=None,
            optimal_order_without_audit=None,
            expected_order_after_audit=None,
            order_by_audited_stock=None,
            warning="An estimated inventory distribution is required.",
        )

    distribution = inventory.available_stock_distribution
    if (
        not distribution
        or any(not isfinite(value) or value < 0 for value in distribution)
        or abs(sum(distribution) - 1.0) > 1e-8
    ):
        return AuditRecommendation(
            status="insufficient_information",
            expected_benefit=None,
            audit_cost=audit_cost,
            net_expected_benefit=None,
            expected_loss_without_audit=None,
            expected_loss_with_audit=None,
            stockout_probability=None,
            demand_uncertainty=None,
            inventory_uncertainty=None,
            optimal_order_without_audit=None,
            expected_order_after_audit=None,
            order_by_audited_stock=None,
            warning="The inventory-state distribution is invalid or unavailable.",
        )

    inventory_mean = sum(stock * weight for stock, weight in enumerate(distribution))
    inventory_variance = sum(
        (stock - inventory_mean) ** 2 * weight
        for stock, weight in enumerate(distribution)
    )
    demand_variance = demand_mean + demand_mean**2 / demand_dispersion
    demand_probability = demand_dispersion / (demand_dispersion + demand_mean)
    stockout_probability = sum(
        weight * float(nbinom.sf(stock, demand_dispersion, demand_probability))
        for stock, weight in enumerate(distribution)
    )

    losses_by_stock = [
        _action_losses(
            stock,
            demand_mean,
            demand_dispersion,
            costs,
            max_order_units,
        )
        for stock in range(len(distribution))
    ]
    loss_without_audit_by_order = [
        sum(
            weight * losses[order]
            for weight, losses in zip(distribution, losses_by_stock, strict=True)
        )
        for order in range(max_order_units + 1)
    ]
    optimal_order_without_audit = min(
        range(max_order_units + 1),
        key=loss_without_audit_by_order.__getitem__,
    )
    expected_loss_without_audit = loss_without_audit_by_order[
        optimal_order_without_audit
    ]

    order_by_audited_stock: list[int] = []
    expected_loss_with_audit = 0.0
    expected_order_after_audit = 0.0
    for stock, weight in enumerate(distribution):
        losses = losses_by_stock[stock]
        optimal_order = min(range(max_order_units + 1), key=losses.__getitem__)
        order_by_audited_stock.append(optimal_order)
        expected_order_after_audit += weight * optimal_order
        expected_loss_with_audit += weight * losses[optimal_order]

    expected_benefit = expected_loss_without_audit - expected_loss_with_audit
    if expected_benefit < -1e-9:
        raise ArithmeticError("Audit information increased the optimized expected loss")
    expected_benefit = max(expected_benefit, 0.0)
    net_benefit = expected_benefit - audit_cost
    return AuditRecommendation(
        status="recommend" if expected_benefit > audit_cost else "defer",
        expected_benefit=expected_benefit,
        audit_cost=audit_cost,
        net_expected_benefit=net_benefit,
        expected_loss_without_audit=expected_loss_without_audit,
        expected_loss_with_audit=expected_loss_with_audit,
        stockout_probability=stockout_probability,
        demand_uncertainty=sqrt(demand_variance),
        inventory_uncertainty=sqrt(max(inventory_variance, 0.0)),
        optimal_order_without_audit=optimal_order_without_audit,
        expected_order_after_audit=expected_order_after_audit,
        order_by_audited_stock=tuple(order_by_audited_stock),
    )
