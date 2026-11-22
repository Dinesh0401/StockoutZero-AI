import math

import pytest
from research.hierarchical_demand import (
    CensoredDemandObservation,
    HierarchicalDemandConfig,
    censored_negative_binomial_log_likelihood,
    fit_hierarchical_censored_demand,
    make_synthetic_hierarchical_dataset,
)


def test_synthetic_fit_is_reproducible_and_predicts_positive_demand():
    dataset = make_synthetic_hierarchical_dataset(days=12, seed=7)
    first = fit_hierarchical_censored_demand(dataset.observations)
    second = fit_hierarchical_censored_demand(dataset.observations)

    assert first.status == "estimated"
    assert first == second
    prediction = first.predict_mean("store_a", "product_a")
    assert prediction is not None and prediction > 0


def test_partial_pooling_shrinks_effects_toward_zero():
    rows = tuple(
        CensoredDemandObservation(
            "store_a" if index % 2 == 0 else "store_b",
            "product_a",
            sales,
            20,
        )
        for index, sales in enumerate((1, 2, 1, 2, 12, 14, 13, 15))
    )
    result = fit_hierarchical_censored_demand(
        rows,
        HierarchicalDemandConfig(store_prior_scale=0.25, product_prior_scale=0.25),
    )
    assert result.status == "estimated"
    assert max(abs(value) for value in result.store_effects.values()) < 1.0


def test_all_censored_history_abstains():
    rows = tuple(CensoredDemandObservation("store", "product", 5, 5) for _ in range(8))
    result = fit_hierarchical_censored_demand(rows)
    assert result.status == "insufficient_information"
    assert result.predict_mean("store", "product") is None


def test_zero_and_unknown_availability_are_excluded_not_zero_demand():
    rows = (
        CensoredDemandObservation("store", "product", 0, 0),
        CensoredDemandObservation("store", "product", 0, None),
        CensoredDemandObservation("store", "product", 2, 10),
        CensoredDemandObservation("store", "product", 1, 10),
    )
    result = fit_hierarchical_censored_demand(rows)
    assert result.status == "estimated"
    assert result.observations_used == 2
    assert result.warning is not None


def test_nb_likelihood_rejects_invalid_availability_and_has_finite_value():
    row = CensoredDemandObservation("store", "product", 2, 4)
    value = censored_negative_binomial_log_likelihood((row,), (3.0,), 4.0)
    assert math.isfinite(value)
    with pytest.raises(ValueError, match="positive availability"):
        censored_negative_binomial_log_likelihood(
            (CensoredDemandObservation("store", "product", 0, 0),),
            (3.0,),
            4.0,
        )
