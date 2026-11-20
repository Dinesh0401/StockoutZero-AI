"""Controlled 30-day and 90-day baseline experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from importlib import import_module

from app.baselines import ForecastCase, baseline_suite

from research.evaluation import SuiteEvaluation, evaluate_suite

np = import_module("numpy")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a reproducible latent-demand benchmark."""

    history_days: int
    trials: int = 100
    seed: int = 42
    demand_process: str = "poisson"


def make_cases(config: BenchmarkConfig) -> tuple[list[ForecastCase], list[float]]:
    """Generate latent demand, censored sales, and known availability."""
    if config.history_days not in (30, 90):
        raise ValueError("history_days must be 30 or 90")
    if config.trials < 1:
        raise ValueError("trials must be positive")

    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)
    cases: list[ForecastCase] = []
    true_means: list[float] = []
    for _ in range(config.trials):
        if config.demand_process == "poisson":
            latent = np_rng.poisson(16.5, config.history_days).tolist()
        elif config.demand_process == "negative_binomial":
            dispersion = 4.0
            probability = dispersion / (dispersion + 16.5)
            latent = np_rng.negative_binomial(
                dispersion, probability, config.history_days
            ).tolist()
        else:
            raise ValueError("Unsupported demand process")

        observations: list[dict[str, int | None]] = []
        sales_history: list[int] = []
        for demand in latent:
            available = rng.choice([0, 8, 12, 18, 24, 30])
            sales = min(int(demand), available)
            sales_history.append(sales)
            observations.append(
                {"sales": sales, "available_stock": available}
            )
        cases.append(
            ForecastCase(
                sales=tuple(sales_history),
                observations=tuple(observations),
            )
        )
        true_means.append(float(np.mean(latent)))
    return cases, true_means


def run_benchmark(config: BenchmarkConfig) -> SuiteEvaluation:
    """Run the six-baseline benchmark and return paired summaries."""
    cases, true_means = make_cases(config)
    return evaluate_suite(baseline_suite(), cases, true_means, config.seed)


if __name__ == "__main__":
    for days in (30, 90):
        result = run_benchmark(BenchmarkConfig(history_days=days))
        print(f"{days}-day benchmark")
        for name, summary in result.summaries.items():
            print(
                f"{name}: success={summary.successful_trials}/{summary.trials}, "
                f"failure_rate={summary.failure_rate:.3f}, mae={summary.mae}"
            )
