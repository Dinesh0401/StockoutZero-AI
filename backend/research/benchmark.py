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
    forecast_days: int = 14
    trials: int = 100
    seed: int = 42
    demand_process: str = "poisson"
    demand_mean: float = 16.5
    intermittent_probability: float = 0.0
    availability_profile: str = "mixed"


def make_cases(config: BenchmarkConfig) -> tuple[list[ForecastCase], list[float]]:
    """Generate latent demand, censored sales, and known availability."""
    if config.history_days not in (30, 90):
        raise ValueError("history_days must be 30 or 90")
    if config.trials < 1:
        raise ValueError("trials must be positive")
    if config.demand_mean <= 0:
        raise ValueError("demand_mean must be positive")
    if not 0 <= config.intermittent_probability < 1:
        raise ValueError("intermittent_probability must be in [0, 1)")
    availability = {
        "ample": [18, 24, 30, 36],
        "mixed": [0, 8, 12, 18, 24, 30],
        "severe": [0, 0, 4, 8, 12],
    }
    if config.availability_profile not in availability:
        raise ValueError("Unsupported availability profile")

    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)
    cases: list[ForecastCase] = []
    future_means: list[float] = []
    parameter_means: list[float] = []
    for _ in range(config.trials):
        total_days = config.history_days + config.forecast_days
        if config.demand_process == "poisson":
            latent = np_rng.poisson(config.demand_mean, total_days).tolist()
        elif config.demand_process == "negative_binomial":
            dispersion = 4.0
            probability = dispersion / (dispersion + config.demand_mean)
            latent = np_rng.negative_binomial(
                dispersion, probability, total_days
            ).tolist()
        else:
            raise ValueError("Unsupported demand process")

        if config.intermittent_probability:
            active = np_rng.random(total_days) >= config.intermittent_probability
            latent = [
                value if active[index] else 0 for index, value in enumerate(latent)
            ]
        observations: list[dict[str, int | None]] = []
        sales_history: list[int] = []
        for demand in latent[: config.history_days]:
            available = rng.choice(availability[config.availability_profile])
            sales = min(int(demand), available)
            sales_history.append(sales)
            observations.append({"sales": sales, "available_stock": available})
        cases.append(
            ForecastCase(
                sales=tuple(sales_history),
                observations=tuple(observations),
                future_demand=tuple(
                    int(value) for value in latent[config.history_days :]
                ),
            )
        )
        future_means.append(float(np.mean(latent[config.history_days:])))
        parameter_means.append(float(config.demand_mean))
    return cases, future_means


def make_parameter_means(config: BenchmarkConfig) -> list[float]:
    """Return the known latent parameter mean for interval-coverage tests."""
    if config.trials < 1:
        raise ValueError("trials must be positive")
    return [float(config.demand_mean)] * config.trials


def run_benchmark(config: BenchmarkConfig) -> SuiteEvaluation:
    """Run the six-baseline benchmark and return paired summaries."""
    cases, true_means = make_cases(config)
    return evaluate_suite(
        baseline_suite(),
        cases,
        true_means,
        config.seed,
        parameter_means=make_parameter_means(config),
    )


def run_research_matrix(trials: int = 20) -> dict[str, SuiteEvaluation]:
    """Run the default reproducible demand and availability matrix."""
    results: dict[str, SuiteEvaluation] = {}
    for history_days in (30, 90):
        for demand_mean in (4.0, 16.5, 40.0):
            for availability in ("ample", "mixed", "severe"):
                name = f"{history_days}d_mean{demand_mean:g}_{availability}"
                results[name] = run_benchmark(
                    BenchmarkConfig(
                        history_days=history_days,
                        trials=trials,
                        demand_mean=demand_mean,
                        availability_profile=availability,
                    )
                )
    return results


if __name__ == "__main__":
    for days in (30, 90):
        result = run_benchmark(BenchmarkConfig(history_days=days))
        print(f"{days}-day benchmark")
        for name, summary in result.summaries.items():
            print(
                f"{name}: success={summary.successful_trials}/{summary.trials}, "
                f"failure_rate={summary.failure_rate:.3f}, mae={summary.mae}"
            )
