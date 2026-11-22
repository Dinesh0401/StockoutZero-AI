import random
from importlib import import_module

from app.forecast import estimate_demand

np = import_module("numpy")


def run(
    trials: int = 100,
    seed: int = 42,
    demand_process: str = "poisson",
) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    censored_mae = []
    naive_mae = []
    successful = 0

    for _ in range(trials):
        if demand_process == "poisson":
            latent_values = np_rng.poisson(lam=16.5, size=14).tolist()
        elif demand_process == "negative_binomial":
            mean = 16.5
            dispersion = 4.0
            probability = dispersion / (dispersion + mean)
            latent_values = np_rng.negative_binomial(
                dispersion, probability, size=14
            ).tolist()
        else:
            raise ValueError("demand_process must be 'poisson' or 'negative_binomial'")

        observations = []
        for latent in latent_values:
            available = rng.choice([0, 8, 12, 18, 24, 30])
            sales = min(latent, available)
            observations.append({"sales": sales, "available_stock": available})

        result = estimate_demand(observations)
        estimate = result["estimated_daily_demand"]
        true_mean = sum(latent_values) / len(latent_values)
        naive_mean = sum(row["sales"] for row in observations) / len(observations)

        if result["status"] == "estimated" and estimate is not None:
            successful += 1
            censored_mae.append(abs(estimate - true_mean))
            naive_mae.append(abs(naive_mean - true_mean))

    print(f"Synthetic stationary-{demand_process.replace('_', ' ').title()} simulation")
    print(f"Trials: {trials}")
    print(f"Successful estimates: {successful}")
    naive_error = sum(naive_mae) / len(naive_mae) if naive_mae else 0
    censored_error = sum(censored_mae) / len(censored_mae) if censored_mae else 0
    print(f"Naive sales mean MAE: {naive_error:.3f}")
    print(f"Censored model MAE: {censored_error:.3f}")


if __name__ == "__main__":
    run()
