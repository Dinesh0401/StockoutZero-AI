from app.baselines import ForecastCase, baseline_suite
from research.benchmark import BenchmarkConfig, make_cases
from research.evaluation import evaluate_suite
from research.inventory import simulate_inventory


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


def test_inventory_simulation_reports_cost_and_service():
    result = simulate_inventory([5, 0, 10], [5, 5, 5], initial_stock=0)
    assert result.orders == 2
    assert result.stockout_units == 5
    assert 0 < result.fill_rate < 1
    assert result.total_cost > 0
