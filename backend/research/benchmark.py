"""Controlled 30-day and 90-day baseline experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

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
        future_availability: list[int] = []
        for _ in latent[config.history_days :]:
            future_availability.append(
                rng.choice(availability[config.availability_profile])
            )
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
                future_availability=tuple(future_availability),
            )
        )
        future_means.append(float(np.mean(latent[config.history_days :])))
    return cases, future_means


def make_parameter_means(config: BenchmarkConfig) -> list[float] | None:
    """Return the known latent parameter mean for interval-coverage tests."""
    if config.trials < 1:
        raise ValueError("trials must be positive")
    if config.demand_process != "poisson":
        return None
    return [
        float(config.demand_mean * (1 - config.intermittent_probability))
    ] * config.trials


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


def run_research_matrix(
    trials: int = 20,
    seeds: tuple[int, ...] = (42,),
) -> dict[str, SuiteEvaluation]:
    """Run the default reproducible demand and availability matrix."""
    results: dict[str, SuiteEvaluation] = {}
    for seed in seeds:
        for history_days in (30, 90):
            for demand_process in ("poisson", "negative_binomial"):
                for demand_mean in (4.0, 16.5, 40.0):
                    for intermittent_probability in (0.0, 0.5):
                        for availability in ("ample", "mixed", "severe"):
                            name = (
                                f"seed{seed}_{history_days}d_{demand_process}"
                                f"_mean{demand_mean:g}"
                                f"_int{intermittent_probability:g}_{availability}"
                            )
                            results[name] = run_benchmark(
                                BenchmarkConfig(
                                    history_days=history_days,
                                    forecast_days=14,
                                    trials=trials,
                                    seed=seed,
                                    demand_process=demand_process,
                                    demand_mean=demand_mean,
                                    intermittent_probability=intermittent_probability,
                                    availability_profile=availability,
                                )
                            )
    return results


def save_matrix_results(
    results: dict[str, SuiteEvaluation],
    path: str | Path,
) -> None:
    """Save benchmark summaries in reproducible machine-readable JSON."""
    serializable = {
        name: {
            summary_name: {
                "name": summary.name,
                "trials": summary.trials,
                "successful_trials": summary.successful_trials,
                "failure_rate": summary.failure_rate,
                "mae": summary.mae,
                "bootstrap_mae_interval_90": summary.bootstrap_mae_interval_90,
                "interval_coverage": summary.interval_coverage,
            }
            for summary_name, summary in result.summaries.items()
        }
        for name, result in results.items()
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    for days in (30, 90):
        result = run_benchmark(BenchmarkConfig(history_days=days))
        print(f"{days}-day benchmark")
        for name, summary in result.summaries.items():
            print(
                f"{name}: success={summary.successful_trials}/{summary.trials}, "
                f"failure_rate={summary.failure_rate:.3f}, mae={summary.mae}"
            )
