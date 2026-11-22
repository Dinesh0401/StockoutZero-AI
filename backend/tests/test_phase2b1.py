from app.baselines import BaselineEstimate, ForecastCase, baseline_suite
from research.benchmark import (
    BenchmarkConfig,
    make_cases,
    make_parameter_means,
    run_research_matrix,
)
from research.evaluation import evaluate_suite, paired_mae_difference
from research.inventory import (
    compare_baseline_inventory_policies,
    compare_rolling_baseline_policies,
    paired_policy_cost_differences,
    simulate_forecast_policy,
    simulate_inventory,
)


def test_all_six_baselines_share_one_interface():
    case = ForecastCase(
        sales=(1, 0, 3, 0),
        observations=(
            {"sales": 1, "available_stock": 4},
            {"sales": 0, "available_stock": 0},
            {"sales": 3, "available_stock": 5},
            {"sales": 0, "available_stock": 4},
        ),
    )
    results = {name: baseline(case) for name, baseline in baseline_suite().items()}
    assert set(results) == {
        "sales_mean",
        "moving_average",
        "croston",
        "tsb",
        "poisson_uncensored",
        "poisson_censored",
    }


def test_benchmark_supports_30_and_90_day_histories():
    for days in (30, 90):
        cases, means = make_cases(BenchmarkConfig(history_days=days, trials=3))
        assert len(cases) == len(means) == 3
        assert len(cases[0].sales) == days


def test_suite_reports_every_baseline():
    cases, means = make_cases(BenchmarkConfig(history_days=30, trials=4))
    result = evaluate_suite(baseline_suite(), cases, means)
    assert len(result.summaries) == 6
    assert all(summary.trials == 4 for summary in result.summaries.values())


def test_paired_comparison_returns_difference_interval():
    cases, means = make_cases(BenchmarkConfig(history_days=30, trials=4))
    difference, interval = paired_mae_difference(
        baseline_suite()["sales_mean"],
        baseline_suite()["poisson_censored"],
        cases,
        means,
    )
    assert difference is not None
    assert interval is not None
    assert interval[0] <= interval[1]


def test_inventory_simulation_reports_cost_and_service():
    result = simulate_inventory([5, 0, 10], [5, 5, 5], initial_stock=0)
    assert result.orders == 2
    assert result.stockout_units == 5
    assert 0 < result.fill_rate < 1
    assert result.total_cost > 0


def test_inventory_simulation_tracks_lead_time_purchase_orders():
    result = simulate_inventory([5, 0, 0], [5, 5, 5], lead_time_days=2)
    assert result.orders == 1
    assert result.fulfilled_units == 0
    assert result.stockout_units == 5


def test_forecast_policy_drives_reorder_points():
    result = simulate_forecast_policy(
        demand=[4, 4, 0],
        daily_forecast=[4, 4, 4],
        lead_time_days=1,
        initial_stock=0,
    )
    assert result.orders > 0
    assert result.fulfilled_units > 0


def test_forecast_policy_rounds_order_quantity_to_case_pack():
    result = simulate_forecast_policy(
        demand=[5, 0],
        daily_forecast=[12, 12],
        lead_time_days=1,
        initial_stock=5,
        case_pack=6,
    )
    assert result.purchased_units >= 6
    assert result.purchased_units % 6 == 0


def test_exact_case_pack_order_scenario():
    result = simulate_forecast_policy(
        demand=[0],
        daily_forecast=[12],
        lead_time_days=1,
        initial_stock=5,
        case_pack=6,
    )
    assert result.purchased_units == 12


def test_inventory_orders_only_whole_case_packs():
    result = simulate_inventory(
        demand=[0],
        reorder_points=[12],
        initial_stock=5,
        lead_time_days=1,
        case_pack=6,
    )
    assert result.purchased_units == 12


def test_zero_shortage_places_no_order():
    result = simulate_inventory(
        demand=[0],
        reorder_points=[5],
        initial_stock=5,
        case_pack=6,
    )
    assert result.purchased_units == 0
    assert result.orders == 0


def test_inventory_position_above_target_places_no_order():
    result = simulate_inventory(
        demand=[0],
        reorder_points=[4],
        initial_stock=11,
        lead_time_days=2,
        case_pack=6,
    )
    assert result.purchased_units == 0
    assert result.orders == 0


def test_invalid_case_pack_is_rejected():
    try:
        simulate_inventory([0], [12], case_pack=0)
    except ValueError as exc:
        assert "case pack" in str(exc).lower()
    else:
        raise AssertionError("Invalid case pack must raise ValueError")


def test_initial_inventory_can_be_not_divisible_by_case_pack():
    result = simulate_inventory(
        demand=[0],
        reorder_points=[12],
        initial_stock=5,
        lead_time_days=0,
        case_pack=6,
    )
    assert result.purchased_units == 12
    assert result.purchased_units % 6 == 0


def test_outstanding_order_is_included_in_inventory_position():
    result = simulate_inventory(
        demand=[0, 0],
        reorder_points=[12, 12],
        initial_stock=0,
        lead_time_days=2,
        case_pack=6,
    )
    assert result.purchased_units == 12


def test_history_only_rolling_object_is_used():
    cases, _ = make_cases(BenchmarkConfig(history_days=30, trials=2))

    def history_only(case):
        assert case.future_demand == ()
        assert case.future_availability == ()
        return BaselineEstimate("history_only", 8.0)

    comparisons = compare_rolling_baseline_policies(
        cases,
        ["history_only"],
        {"history_only": history_only},
        lead_time_days=2,
        case_pack=6,
    )
    assert comparisons["history_only"].outcomes


def test_rolling_abstention_reports_explicit_fallback_usage():
    cases, _ = make_cases(BenchmarkConfig(history_days=30, trials=2))

    def abstaining(case):
        return BaselineEstimate("abstaining", None, "insufficient_information")

    comparisons = compare_rolling_baseline_policies(
        cases,
        ["abstaining"],
        {"abstaining": abstaining},
        abstention_fallback="conservative",
    )
    outcome = comparisons["abstaining"].outcomes[0]
    assert outcome.abstentions == len(cases[0].future_demand)
    assert outcome.fallback_usage == outcome.abstentions


def test_inventory_policy_comparison_is_model_driven():
    cases, _ = make_cases(BenchmarkConfig(history_days=30, trials=3))
    comparisons = compare_baseline_inventory_policies(
        cases,
        [
            "sales_mean",
            "moving_average",
            "croston",
            "tsb",
            "poisson_uncensored",
            "poisson_censored",
        ],
        baseline_suite(),
    )
    assert set(comparisons).issubset(baseline_suite())


def test_benchmark_supports_demand_and_availability_regimes():
    cases, means = make_cases(
        BenchmarkConfig(
            history_days=30,
            trials=3,
            demand_mean=8,
            intermittent_probability=0.5,
            availability_profile="severe",
        )
    )
    assert len(cases) == len(means) == 3
    assert any(value == 0 for value in cases[0].sales)


def test_research_matrix_has_default_combinations():
    matrix = run_research_matrix(trials=2)
    assert len(matrix) == 72
    assert all(
        summary.trials == 2
        for result in matrix.values()
        for summary in result.summaries.values()
    )


def test_rolling_comparison_preserves_every_trial():
    cases, _ = make_cases(BenchmarkConfig(history_days=30, trials=3))
    comparisons = compare_rolling_baseline_policies(
        cases,
        ["sales_mean", "croston", "poisson_censored"],
        baseline_suite(),
        lead_time_days=2,
        case_pack=6,
    )
    assert set(comparisons) == {"sales_mean", "croston", "poisson_censored"}
    assert all(len(comparison.outcomes) == 3 for comparison in comparisons.values())


def test_parameter_coverage_does_not_fallback_to_future_means():
    cases, future_means = make_cases(BenchmarkConfig(history_days=30, trials=3))
    result = evaluate_suite(
        {"poisson_censored": baseline_suite()["poisson_censored"]},
        cases,
        future_means,
        parameter_means=None,
    )
    coverage = result.summaries["poisson_censored"].interval_coverage
    assert coverage is None


def test_paired_policy_costs_report_uncertainty():
    cases, _ = make_cases(BenchmarkConfig(history_days=30, trials=3))
    comparisons = compare_rolling_baseline_policies(
        cases,
        ["sales_mean", "croston"],
        baseline_suite(),
        lead_time_days=2,
    )
    metrics = paired_policy_cost_differences(comparisons, "sales_mean")
    croston = next(metric for metric in metrics if metric.baseline == "croston")
    assert croston.paired_trials == 3
    assert croston.bootstrap_interval_90 is not None


def test_intermittent_parameter_mean_is_unconditional():
    config = BenchmarkConfig(
        history_days=30,
        trials=3,
        demand_mean=8,
        intermittent_probability=0.5,
    )
    assert make_parameter_means(config) is None
