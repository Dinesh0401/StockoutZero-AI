import pytest
from research.audit_policy import recommend_stock_audit
from research.inventory import InventoryCosts
from research.uncertain_inventory import InventoryDayEstimate


def _inventory_estimate(
    distribution: tuple[float, ...],
) -> InventoryDayEstimate:
    expected = sum(
        stock * probability for stock, probability in enumerate(distribution)
    )
    return InventoryDayEstimate(
        expected_available_stock=expected,
        available_stock_distribution=distribution,
        expected_ending_stock=expected,
        ending_stock_distribution=distribution,
        stockout_probability=0.0,
        audit_used=False,
    )


def test_exact_stock_information_has_no_value_for_audit() -> None:
    result = recommend_stock_audit(
        _inventory_estimate((0.0, 0.0, 1.0, 0.0)),
        demand_mean=2.0,
        demand_dispersion=4.0,
        audit_cost=0.1,
        max_order_units=8,
    )

    assert result.status == "defer"
    assert result.expected_benefit == pytest.approx(0.0)
    assert result.inventory_uncertainty == pytest.approx(0.0)
    assert result.order_by_audited_stock is not None
    assert result.order_by_audited_stock[2] == result.optimal_order_without_audit


def test_audit_is_recommended_when_information_value_exceeds_cost() -> None:
    result = recommend_stock_audit(
        _inventory_estimate((0.5, 0.0, 0.0, 0.0, 0.0, 0.5)),
        demand_mean=4.0,
        demand_dispersion=4.0,
        audit_cost=0.01,
        costs=InventoryCosts(holding=0.1, stockout=4.0, purchase=0.2),
        max_order_units=12,
    )

    assert result.status == "recommend"
    assert result.expected_benefit > result.audit_cost
    assert result.net_expected_benefit == pytest.approx(
        result.expected_benefit - result.audit_cost
    )
    assert result.expected_loss_with_audit <= result.expected_loss_without_audit
    assert result.inventory_uncertainty > 0
    assert result.stockout_probability is not None
    assert 0 <= result.stockout_probability <= 1
    assert result.demand_uncertainty is not None
    assert result.expected_order_after_audit is not None


def test_audit_is_deferred_when_its_cost_exceeds_expected_benefit() -> None:
    result = recommend_stock_audit(
        _inventory_estimate((0.5, 0.0, 0.0, 0.0, 0.0, 0.5)),
        demand_mean=4.0,
        demand_dispersion=4.0,
        audit_cost=1_000.0,
        max_order_units=12,
    )

    assert result.status == "defer"
    assert result.net_expected_benefit < 0


def test_missing_or_invalid_inventory_distribution_abstains() -> None:
    missing = recommend_stock_audit(
        None,
        demand_mean=2.0,
        demand_dispersion=4.0,
        audit_cost=1.0,
    )
    invalid = recommend_stock_audit(
        _inventory_estimate((0.5, 0.5)),
        demand_mean=2.0,
        demand_dispersion=4.0,
        audit_cost=1.0,
    )

    assert missing.status == "insufficient_information"
    assert invalid.status == "insufficient_information"
    assert missing.expected_benefit is None


@pytest.mark.parametrize(
    ("demand_mean", "dispersion", "audit_cost", "max_order"),
    (
        (0.0, 4.0, 1.0, 10),
        (2.0, 0.0, 1.0, 10),
        (2.0, 4.0, -1.0, 10),
        (2.0, 4.0, 1.0, -1),
    ),
)
def test_invalid_audit_inputs_are_rejected(
    demand_mean: float,
    dispersion: float,
    audit_cost: float,
    max_order: int,
) -> None:
    with pytest.raises(ValueError):
        recommend_stock_audit(
            _inventory_estimate((1.0,)),
            demand_mean=demand_mean,
            demand_dispersion=dispersion,
            audit_cost=audit_cost,
            max_order_units=max_order,
        )
