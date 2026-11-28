import math

import pytest
from research.uncertain_inventory import (
    InventoryDayObservation,
    InventoryStateConfig,
    infer_inventory_state,
)
from scipy.stats import nbinom


def test_sales_update_stock_distribution_using_censored_demand_likelihood() -> None:
    config = InventoryStateConfig(max_stock=5, demand_mean=3.0)
    result = infer_inventory_state(
        (InventoryDayObservation(sales=2),),
        (0.0, 0.0, 0.5, 0.5, 0.0, 0.0),
        config,
    )

    estimate = result.days[0]
    dispersion = config.demand_dispersion
    probability = dispersion / (dispersion + config.demand_mean)
    likelihood_at_two = nbinom.sf(1, dispersion, probability)
    likelihood_at_three = nbinom.pmf(2, dispersion, probability)
    expected_probability_at_two = likelihood_at_two / (
        likelihood_at_two + likelihood_at_three
    )
    assert estimate.available_stock_distribution[2] == pytest.approx(
        expected_probability_at_two
    )
    assert estimate.available_stock_distribution[3] == pytest.approx(
        1 - expected_probability_at_two
    )
    assert estimate.expected_ending_stock == pytest.approx(
        estimate.expected_available_stock - 2
    )


def test_receipts_removals_sales_and_exact_audit_reconcile_sequentially() -> None:
    result = infer_inventory_state(
        (
            InventoryDayObservation(
                sales=1,
                receipts=2,
                known_removals=1,
                audited_stock=5,
            ),
            InventoryDayObservation(sales=2, audited_stock=3),
        ),
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        InventoryStateConfig(max_stock=8, demand_mean=3.0),
    )

    first, second = result.days
    assert first.expected_available_stock == pytest.approx(6.0)
    assert first.expected_ending_stock == pytest.approx(5.0)
    assert first.audit_used
    assert second.expected_available_stock == pytest.approx(5.0)
    assert second.expected_ending_stock == pytest.approx(3.0)
    assert result.final_stock_distribution[3] == pytest.approx(1.0)


def test_stockout_probability_is_positive_for_zero_available_stock() -> None:
    result = infer_inventory_state(
        (InventoryDayObservation(sales=0),),
        (1.0, 0.0, 0.0),
        InventoryStateConfig(max_stock=2, demand_mean=3.0),
    )

    expected = 1 - nbinom.pmf(
        0,
        4.0,
        4.0 / 7.0,
    )
    assert result.days[0].stockout_probability == pytest.approx(expected)
    assert result.days[0].available_stock_distribution[0] == pytest.approx(1.0)
    assert "no physical audits" in result.warning


def test_poisson_shrinkage_and_closing_audit_update_inventory_state() -> None:
    result = infer_inventory_state(
        (
            InventoryDayObservation(
                sales=0,
                audited_stock=2,
            ),
        ),
        (0.0, 0.0, 0.0, 0.0, 1.0),
        InventoryStateConfig(
            max_stock=4,
            demand_mean=2.0,
            daily_shrinkage_mean=1.0,
        ),
    )

    assert result.days[0].expected_ending_stock == pytest.approx(2.0)
    assert result.days[0].expected_available_stock == pytest.approx(2.0)
    assert result.days[0].ending_stock_distribution[2] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "observation",
    (
        InventoryDayObservation(sales=-1),
        InventoryDayObservation(sales=0, receipts=-1),
        InventoryDayObservation(sales=0, audited_stock=4),
    ),
)
def test_invalid_day_counts_are_rejected(observation) -> None:
    with pytest.raises(ValueError):
        infer_inventory_state(
            (observation,),
            (1.0, 0.0, 0.0),
            InventoryStateConfig(max_stock=2, demand_mean=2.0),
        )


def test_incompatible_sales_and_initial_stock_prior_are_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible with the stock prior"):
        infer_inventory_state(
            (InventoryDayObservation(sales=2),),
            (1.0, 0.0, 0.0),
            InventoryStateConfig(max_stock=2, demand_mean=2.0),
        )


def test_ledger_measurement_model_keeps_a_normalized_posterior() -> None:
    result = infer_inventory_state(
        (InventoryDayObservation(sales=1, recorded_stock=2),),
        (0.0, 0.0, 0.5, 0.5, 0.0),
        InventoryStateConfig(max_stock=4, demand_mean=2.0),
    )

    for distribution in (
        result.days[0].available_stock_distribution,
        result.days[0].ending_stock_distribution,
        result.final_stock_distribution,
    ):
        assert all(math.isfinite(value) and value >= 0 for value in distribution)
        assert sum(distribution) == pytest.approx(1.0)


def test_input_prior_must_be_a_probability_distribution() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        infer_inventory_state(
            (InventoryDayObservation(sales=0),),
            (0.5, 0.5, 0.5),
            InventoryStateConfig(max_stock=2, demand_mean=2.0),
        )


def test_max_stock_must_be_a_whole_number() -> None:
    with pytest.raises(ValueError, match="max_stock"):
        infer_inventory_state(
            (InventoryDayObservation(sales=0),),
            (1.0, 0.0, 0.0),
            InventoryStateConfig(max_stock=2.5, demand_mean=2.0),
        )
