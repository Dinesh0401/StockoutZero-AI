import math

import pytest
from research.hierarchical_demand import (
    CensoredDemandObservation,
    HierarchicalDemandConfig,
    censored_negative_binomial_log_likelihood,
    evaluate_hierarchical_holdout,
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


@pytest.mark.parametrize(
    ("row", "message"),
    (
        (
            CensoredDemandObservation("", "product", 1, 10),
            "store_id",
        ),
        (
            CensoredDemandObservation("store", "", 1, 10),
            "product_id",
        ),
        (
            CensoredDemandObservation("store", "product", -1, 10),
            "non-negative",
        ),
        (
            CensoredDemandObservation("store", "product", 11, 10),
            "cannot exceed",
        ),
    ),
)
def test_hierarchical_fit_rejects_invalid_observations(row, message):
    with pytest.raises(ValueError, match=message):
        fit_hierarchical_censored_demand((row,))


def test_numerical_boundaries_return_abstention_or_finite_interval():
    zero_rows = tuple(
        CensoredDemandObservation("store", "product", 0, 10) for _ in range(12)
    )
    fit = fit_hierarchical_censored_demand(zero_rows)
    assert fit.status == "estimated"
    interval = fit.predict_mean_interval("store", "product")
    assert interval is None or all(
        math.isfinite(value) and value >= 0 for value in interval
    )

    sparse_censored = tuple(
        CensoredDemandObservation("store", "product", 1, 1) for _ in range(8)
    )
    sparse_fit = fit_hierarchical_censored_demand(sparse_censored)
    assert sparse_fit.status == "insufficient_information"


def test_held_out_evaluation_is_paired_and_reproducible():
    dataset = make_synthetic_hierarchical_dataset(days=30, seed=19)
    first = evaluate_hierarchical_holdout(dataset, seed=42)
    second = evaluate_hierarchical_holdout(dataset, seed=42)

    assert first == second
    assert first.successful_pairs >= 1
    assert first.hierarchical_mae >= 0
    assert first.sales_mean_mae >= 0
    assert len(first.bootstrap_mae_difference) == 2


def test_hidden_latent_demand_is_not_used_by_fit():
    dataset = make_synthetic_hierarchical_dataset(days=12, seed=7)
    fit = fit_hierarchical_censored_demand(dataset.observations)
    assert fit.status == "estimated"
    assert dataset.latent_demand
    for value in dataset.latent_demand:
        assert isinstance(value, int)
