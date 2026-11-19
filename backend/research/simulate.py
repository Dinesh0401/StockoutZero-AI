import random

from app.forecast import estimate_demand


def run(trials: int = 100, seed: int = 42):
    rng = random.Random(seed)
    censored_mae = []
    naive_mae = []
    successful = 0

    for _ in range(trials):
        latent_values = [rng.randint(8, 25) for _ in range(14)]
        observations = []
        for latent in latent_values:
            available = rng.choice([0, 8, 12, 18, 24, 30])
            sales = min(latent, available)
            observations.append({"sales": sales, "available_stock": available})

        result = estimate_demand(observations)
        estimate = result["estimated_daily_demand"]
        true_mean = sum(latent_values) / len(latent_values)
        naive_mean = sum(row["sales"] for row in observations) / len(observations)

        if estimate > 0:
            successful += 1
            censored_mae.append(abs(estimate - true_mean))
            naive_mae.append(abs(naive_mean - true_mean))

    print("Synthetic stationary-Poisson simulation")
    print(f"Trials: {trials}")
    print(f"Successful estimates: {successful}")
    print(f"Naive sales mean MAE: {sum(naive_mae) / len(naive_mae) if naive_mae else 0:.3f}")
    print(f"Censored model MAE: {sum(censored_mae) / len(censored_mae) if censored_mae else 0:.3f}")


if __name__ == "__main__":
    run()
