"""Sequential inventory-cost simulation for baseline decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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


def simulate_inventory(
    demand: Sequence[int],
    reorder_points: Sequence[int],
    costs: InventoryCosts = InventoryCosts(),
    initial_stock: int = 0,
    lead_time_days: int = 0,
) -> InventorySimulation:
    """Simulate daily replenishment decisions with outstanding purchase orders."""
    if len(demand) != len(reorder_points):
        raise ValueError("Demand and reorder points must have equal lengths")
    if initial_stock < 0:
        raise ValueError("Initial stock cannot be negative")
    if lead_time_days < 0:
        raise ValueError("Lead time cannot be negative")

    stock = initial_stock
    arrivals: dict[int, int] = {}
    total_cost = 0.0
    stockout_units = 0
    fulfilled_units = 0
    orders = 0
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
            ordered = reorder_point - inventory_position
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
    )
