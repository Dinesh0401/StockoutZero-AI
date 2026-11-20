from app.baselines import (
    BaselineEstimate,
    croston,
    moving_average,
    poisson_censored,
    sales_mean,
    tsb,
)
from research.evaluation import evaluate_baseline


def test_sales_only_baselines_return_estimates():
    history = [0, 4, 0, 0, 6, 0, 3]
    assert sales_mean(history).estimate == 13 / 7
    assert moving_average(history, window=3).estimate == 3
    assert croston(history).estimate is not None
    assert tsb(history).estimate is not None


def test_censored_baseline_reports_non_identifiability():
    result = poisson_censored([{"sales": 10, "available_stock": 10}] * 5)
    assert result.status == "insufficient_information"
    assert result.estimate is None


def test_evaluation_reports_failures_separately():
    histories = [[0, 0], [2, 3], [4, 5]]
    def requires_positive_history(history):
        if not any(history):
            return BaselineEstimate("test_baseline", None, "insufficient_information")
        return sales_mean(history)

    summary = evaluate_baseline(requires_positive_history, histories, [0, 2.5, 4.5])
    assert summary.trials == 3
    assert summary.successful_trials == 2
    assert summary.failure_rate == 1 / 3
    assert summary.mae == 0
    assert summary.bootstrap_mae_interval_90 is not None
