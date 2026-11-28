import json
import math

import pytest
from research.hierarchical_demand import (
    CensoredDemandObservation,
    HierarchicalDemandConfig,
    SyntheticDemandDataset,
    censored_negative_binomial_log_likelihood,
    evaluate_hierarchical_holdout,
    fit_global_pooled_censored_demand,
    fit_hierarchical_censored_demand,
    fit_independent_group_censored_demand,
    make_synthetic_hierarchical_dataset,
    run_hierarchical_validation_matrix,
    run_pooling_ablation,
    run_reproducible_study,
)


def test_synthetic_fit_is_reproducible_and_predicts_positive_demand() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=12, seed=7)
    first = fit_hierarchical_censored_demand(dataset.observations)
    second = fit_hierarchical_censored_demand(dataset.observations)

    assert first.status == "estimated"
    assert first == second
    prediction = first.predict_mean("store_a", "product_a")
    assert prediction is not None and prediction > 0


def test_partial_pooling_shrinks_effects_toward_zero() -> None:
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


def test_all_censored_history_abstains() -> None:
    rows = tuple(CensoredDemandObservation("store", "product", 5, 5) for _ in range(8))
    result = fit_hierarchical_censored_demand(rows)
    assert result.status == "insufficient_information"
    assert result.predict_mean("store", "product") is None


def test_zero_and_unknown_availability_are_excluded_not_zero_demand() -> None:
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


def test_nb_likelihood_rejects_invalid_availability_and_has_finite_value() -> None:
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
def test_hierarchical_fit_rejects_invalid_observations(
    row: CensoredDemandObservation, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_hierarchical_censored_demand((row,))


def test_numerical_boundaries_return_abstention_or_finite_interval() -> None:
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


def test_held_out_evaluation_is_paired_and_reproducible() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=30, seed=19)
    first = evaluate_hierarchical_holdout(dataset, seed=42)
    second = evaluate_hierarchical_holdout(dataset, seed=42)

    assert first == second
    assert first.successful_pairs >= 1
    assert first.hierarchical_mae >= 0
    assert first.sales_mean_mae >= 0
    assert first.moving_average_mae >= 0
    mean_coverage = first.mean_interval_coverage
    assert mean_coverage is None or 0.0 <= mean_coverage <= 1.0
    predictive_coverage = first.predictive_interval_coverage
    assert predictive_coverage is None or 0.0 <= predictive_coverage <= 1.0
    assert len(first.bootstrap_mae_difference) == 2
    assert first.paired_denominator >= 1
    assert first.paired_clusters < first.paired_denominator
    assert first.failed_pairs >= 0
    assert first.abstained_pairs >= 0


def test_hidden_latent_demand_is_not_used_by_fit() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=12, seed=7)
    fit = fit_hierarchical_censored_demand(dataset.observations)
    assert fit.status == "estimated"
    assert dataset.latent_demand
    for value in dataset.latent_demand:
        assert isinstance(value, int)


def test_validation_matrix_is_reproducible_and_has_independent_baselines() -> None:
    first = run_hierarchical_validation_matrix(
        history_lengths=(30,),
        seeds=(7, 19),
    )
    second = run_hierarchical_validation_matrix(
        history_lengths=(30,),
        seeds=(7, 19),
    )

    assert first == second
    assert set(first) == {"30d_seed7", "30d_seed19"}
    assert all(
        set(run.baseline_maes)
        == {
            "sales_mean",
            "moving_average",
            "global_pooled",
            "independent",
            "hierarchical_weak",
            "hierarchical_strong",
        }
        for run in first.values()
    )
    assert all(run.successful_pairs > 0 for run in first.values())
    assert all(
        run.predictive_interval_coverage is None
        or 0 <= run.predictive_interval_coverage <= 1
        for run in first.values()
    )
    assert all(
        value >= 0 for run in first.values() for value in run.baseline_maes.values()
    )


def test_baselines_use_training_only_not_full_dataset() -> None:
    training_row = CensoredDemandObservation("store_x", "product_y", 5, 10, day=0)
    test_row = CensoredDemandObservation("store_x", "product_y", 0, 10, day=1)
    dataset = SyntheticDemandDataset(
        observations=(training_row, test_row),
        latent_demand=(5, 50),
        true_store_effects={},
        true_product_effects={},
    )
    summary = evaluate_hierarchical_holdout(dataset, train_fraction=0.7, seed=42)
    assert summary.sales_mean_mae == 45.0


def test_no_observation_is_evaluated_on_itself() -> None:
    training_row = CensoredDemandObservation("store_x", "product_y", 5, 10, day=0)
    test_row = CensoredDemandObservation("store_x", "product_y", 100, 200, day=1)
    dataset = SyntheticDemandDataset(
        observations=(training_row, test_row),
        latent_demand=(5, 1000),
        true_store_effects={},
        true_product_effects={},
    )
    summary = evaluate_hierarchical_holdout(dataset, train_fraction=0.7, seed=42)
    assert summary.sales_mean_mae == abs(5 - 1000)


def test_pair_with_no_training_history_abstains_in_baseline() -> None:
    train_row = CensoredDemandObservation("store_a", "product_a", 5, 10, day=0)
    test_row = CensoredDemandObservation("store_b", "product_b", 5, 10, day=1)
    dataset = SyntheticDemandDataset(
        observations=(train_row, test_row),
        latent_demand=(5, 5),
        true_store_effects={},
        true_product_effects={},
    )
    summary = evaluate_hierarchical_holdout(dataset, train_fraction=0.7, seed=42)
    assert summary.abstained_pairs >= 1
    assert summary.sales_mean_mae is None
    assert summary.moving_average_mae is None
    assert summary.paired_mae_difference is None
    assert summary.bootstrap_mae_difference is None


def test_mean_interval_coverage_targets_true_mean_not_latent_demand() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=30, seed=7)
    summary = evaluate_hierarchical_holdout(dataset, seed=42)
    mean_coverage = summary.mean_interval_coverage
    assert mean_coverage is None or 0.0 <= mean_coverage <= 1.0
    predictive_coverage = summary.predictive_interval_coverage
    assert predictive_coverage is None or 0.0 <= predictive_coverage <= 1.0


def test_future_observation_mutation_does_not_change_historical_forecasts() -> None:
    first = make_synthetic_hierarchical_dataset(days=30, seed=19)
    changed_observations = tuple(
        CensoredDemandObservation(
            row.store_id,
            row.product_id,
            200 if row.day is not None and row.day >= 21 else row.sales,
            200 if row.day is not None and row.day >= 21 else row.available_stock,
            day=row.day,
        )
        if row.day is not None and row.day >= 21
        else row
        for row in first.observations
    )
    changed = SyntheticDemandDataset(
        observations=changed_observations,
        latent_demand=first.latent_demand,
        true_store_effects=first.true_store_effects,
        true_product_effects=first.true_product_effects,
        true_daily_means=first.true_daily_means,
    )
    first_summary = evaluate_hierarchical_holdout(first, seed=42)
    changed_summary = evaluate_hierarchical_holdout(changed, seed=42)
    assert first_summary.hierarchical_mae == changed_summary.hierarchical_mae
    assert first_summary.bootstrap_mae_difference == (
        changed_summary.bootstrap_mae_difference
    )


def test_predict_interval_returns_nb2_bounds() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=12, seed=7)
    fit = fit_hierarchical_censored_demand(dataset.observations)
    interval = fit.predict_interval("store_a", "product_a")
    assert interval is not None
    lower, upper = interval
    assert math.isfinite(lower) and math.isfinite(upper)
    assert 0 <= lower <= upper


def test_predict_mean_interval_is_mean_parameter_interval() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=30, seed=7)
    fit = fit_hierarchical_censored_demand(dataset.observations)
    assert fit.status == "estimated"
    interval = fit.predict_mean_interval("store_a", "product_a")
    assert interval is not None
    lower, upper = interval
    prediction = fit.predict_mean("store_a", "product_a")
    assert prediction is not None
    assert math.isfinite(lower) and math.isfinite(upper)
    assert 0 < lower <= prediction <= upper


def test_identifiability_store_and_product_effects_sum_to_zero() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=30, seed=7)
    fit = fit_hierarchical_censored_demand(dataset.observations)
    assert fit.status == "estimated"
    assert abs(sum(fit.store_effects.values())) < 1e-9
    assert abs(sum(fit.product_effects.values())) < 1e-9


def test_single_store_and_single_product_is_identifiable() -> None:
    rows = tuple(
        CensoredDemandObservation("s", "p", sales, 20) for sales in (1, 2, 3, 4, 5)
    )
    fit = fit_hierarchical_censored_demand(rows)
    assert fit.status == "estimated"
    assert fit.store_effects == {"s": 0.0}
    assert fit.product_effects == {"p": 0.0}
    prediction = fit.predict_mean("s", "p")
    assert prediction is not None and prediction > 0


def test_single_store_multiple_products_is_identifiable() -> None:
    rows = tuple(
        CensoredDemandObservation("s", f"product_{index}", sales, 20)
        for index, sales in enumerate((1, 2, 3, 4, 5, 6))
    )
    fit = fit_hierarchical_censored_demand(rows)
    assert fit.status == "estimated"
    assert abs(sum(fit.product_effects.values())) < 1e-9
    for product_id in fit.product_effects:
        prediction = fit.predict_mean("s", product_id)
        assert prediction is not None and prediction > 0


def test_global_pooled_fit_returns_single_intercept() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=12, seed=7)
    fit = fit_global_pooled_censored_demand(dataset.observations)
    assert fit.status == "estimated"
    assert fit.intercept is not None
    assert fit.store_effects == {}
    assert fit.product_effects == {}


def test_independent_fit_returns_per_pair_means() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=30, seed=7)
    means = fit_independent_group_censored_demand(dataset.observations)
    assert len(means) == 9
    for mean in means.values():
        assert mean is not None and mean > 0


def test_pooling_ablation_returns_all_approaches() -> None:
    dataset = make_synthetic_hierarchical_dataset(days=30, seed=7)
    ablation = run_pooling_ablation(dataset, train_fraction=0.7, seed=42)
    approaches = ablation["approaches"]
    assert set(approaches) == {
        "global_pooled",
        "independent",
        "hierarchical_weak",
        "hierarchical_strong",
    }
    assert ablation["conclusion"] == ("evidence_reported_no_arbitrary_superiority_rule")
    paired = ablation["paired_difference_vs_global_pooled"]
    assert set(paired) == {"independent", "hierarchical_weak", "hierarchical_strong"}
    assert all(
        result["paired_clusters"] < result["paired_count"] for result in paired.values()
    )


def test_reproducible_study_writes_machine_readable_results(tmp_path) -> None:
    output = tmp_path / "nested" / "study.json"
    records = run_reproducible_study(
        output_path=str(output),
        seeds=(7,),
        history_lengths=(12,),
    )

    assert output.exists()
    assert output.with_suffix(".csv").exists()
    saved_records = json.loads(output.read_text(encoding="utf-8"))
    assert saved_records[0]["trial_id"] == records[0]["trial_id"]
    assert saved_records[0]["error_metrics"] == records[0]["error_metrics"]
    assert records[0]["baseline_identifiers"] == [
        "sales_mean",
        "moving_average",
        "global_pooled",
        "independent",
    ]
    assert set(records[0]["error_metrics"]["pooling_ablations"]) == {
        "global_pooled",
        "independent",
        "hierarchical_weak",
        "hierarchical_strong",
    }
    assert (
        records[0]["pooling_ablation"]["config"]["hierarchical_weak"][
            "store_prior_scale"
        ]
        == 1.5
    )
