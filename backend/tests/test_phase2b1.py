from app.baselines import ForecastCase, baseline_suite
from research.benchmark import BenchmarkConfig, make_cases
from research.evaluation import evaluate_suite, paired_mae_difference
from research.inventory import simulate_forecast_policy, simulate_inventory


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
