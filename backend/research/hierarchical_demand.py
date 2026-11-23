"""Phase 3A hierarchical censored-demand research baseline.

This module deliberately accepts only historical observations.  Future demand,
post-sale inventory, and audit results are not part of the fitting contract.
The model is a transparent MAP baseline, not a production forecaster.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from math import exp, isfinite, log, log1p, sqrt
from typing import Any

from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom, norm

np = import_module("numpy")


@dataclass(frozen=True)
class CensoredDemandObservation:
    """One historical store-product observation at the demand timestamp."""

    store_id: str
    product_id: str
    sales: int
    available_stock: int | None


@dataclass(frozen=True)
class HierarchicalDemandConfig:
    """Configuration for the reproducible MAP baseline."""

    dispersion: float = 4.0
    store_prior_scale: float = 0.75
    product_prior_scale: float = 0.75
    max_iterations: int = 500
    seed: int = 42


@dataclass(frozen=True)
class HierarchicalDemandFit:
    """Fitted effects and explicit estimation status."""

    status: str
    intercept: float | None
    store_effects: dict[str, float]
    product_effects: dict[str, float]
    objective: float | None
    observations_used: int
    censored_observations: int
    warning: str | None = None
    covariance: tuple[tuple[float, ...], ...] | None = None

    def predict_mean(self, store_id: str, product_id: str) -> float | None:
        """Predict latent daily demand for a known or new store-product pair."""
        if self.status != "estimated" or self.intercept is None:
            return None
        log_mean = (
            self.intercept
            + self.store_effects.get(store_id, 0.0)
            + self.product_effects.get(product_id, 0.0)
        )
        return exp(log_mean)

    def predict_mean_interval(
        self,
        store_id: str,
        product_id: str,
        coverage: float = 0.90,
    ) -> tuple[float, float] | None:
        """Return a Laplace log-scale interval for a known or new hierarchy pair."""
        if coverage <= 0 or coverage >= 1:
            raise ValueError("coverage must be between zero and one")
        if self.status != "estimated" or self.intercept is None:
            return None
        if self.covariance is None:
            return None
        stores = tuple(sorted(self.store_effects))
        products = tuple(sorted(self.product_effects))
        if store_id not in stores or product_id not in products:
            return None
        size = 1 + len(stores) + len(products)
        if len(self.covariance) != size:
            return None

        gradient = [1.0] + [0.0] * (len(stores) + len(products))
        if store_id in stores:
            index = stores.index(store_id)
            for position in range(len(stores)):
                gradient[1 + position] = (
                    1.0 if position == index else 0.0
                ) - 1.0 / len(stores)
        if product_id in products:
            index = products.index(product_id)
            offset = 1 + len(stores)
            for position in range(len(products)):
                gradient[offset + position] = (
                    1.0 if position == index else 0.0
                ) - 1.0 / len(products)

        variance = sum(
            gradient[row] * self.covariance[row][column] * gradient[column]
            for row in range(size)
            for column in range(size)
        )
        if variance < 0 or not isfinite(variance) or variance > 100.0:
            return None
        z_value = float(norm.ppf((1.0 + coverage) / 2.0))
        log_mean = (
            self.intercept
            + self.store_effects.get(store_id, 0.0)
            + self.product_effects.get(product_id, 0.0)
        )
        try:
            lower = exp(log_mean - z_value * sqrt(variance))
            upper = exp(log_mean + z_value * sqrt(variance))
        except OverflowError:
            return None
        if not isfinite(lower) or not isfinite(upper):
            return None
        return (lower, upper)


@dataclass(frozen=True)
class SyntheticDemandDataset:
    """Synthetic observations with hidden truth retained for evaluation only."""

    observations: tuple[CensoredDemandObservation, ...]
    latent_demand: tuple[int, ...]
    true_store_effects: dict[str, float]
    true_product_effects: dict[str, float]


@dataclass(frozen=True)
class HierarchicalHoldoutSummary:
    """Paired held-out accuracy and interval coverage for Phase 3A."""

    hierarchical_mae: float
    sales_mean_mae: float
    paired_mae_difference: float
    bootstrap_mae_difference: tuple[float, float]
    interval_coverage: float | None
    successful_pairs: int


def _negative_binomial_logpmf(demand: int, mean: float, dispersion: float) -> float:
    """Return NB2 log-PMF using mean and size (dispersion) parameters."""
    probability = dispersion / (dispersion + mean)
    return float(
        gammaln(demand + dispersion)
        - gammaln(dispersion)
        - gammaln(demand + 1)
        + dispersion * log(probability)
        + demand * log1p(-probability)
    )


def _negative_binomial_logsf(lower_bound: int, mean: float, dispersion: float) -> float:
    """Return log P(D >= lower_bound) for a censored NB2 observation."""
    if lower_bound <= 0:
        return 0.0
    probability = dispersion / (dispersion + mean)
    return float(nbinom.logsf(lower_bound - 1, dispersion, probability))


def censored_negative_binomial_log_likelihood(
    observations: Sequence[CensoredDemandObservation],
    means: Sequence[float],
    dispersion: float,
) -> float:
    """Evaluate the known-availability censored NB2 log likelihood.

    Positive availability is required.  A zero or unknown availability record
    is not evidence about demand and is rejected rather than treated as zero.
    """
    if len(observations) != len(means):
        raise ValueError("Observations and means must have equal lengths")
    if dispersion <= 0:
        raise ValueError("Dispersion must be positive")
    total = 0.0
    for observation, mean in zip(observations, means, strict=True):
        if mean <= 0 or not isfinite(mean):
            raise ValueError("Means must be finite and positive")
        available = observation.available_stock
        if available is None or available <= 0:
            raise ValueError("Likelihood requires known positive availability")
        if observation.sales < 0 or observation.sales > available:
            raise ValueError("Sales must be between zero and available stock")
        if observation.sales < available:
            total += _negative_binomial_logpmf(observation.sales, mean, dispersion)
        else:
            total += _negative_binomial_logsf(available, mean, dispersion)
    return total


def fit_hierarchical_censored_demand(
    observations: Sequence[CensoredDemandObservation],
    config: HierarchicalDemandConfig = HierarchicalDemandConfig(),
) -> HierarchicalDemandFit:
    """Fit pooled store and product effects with a censored NB2 MAP objective.

    Records with unknown or zero availability are excluded and counted in the
    warning.  A history containing no uncensored positive-stock observation is
    explicitly non-identifiable and returns an abstention result.
    """
    if config.dispersion <= 0:
        raise ValueError("Dispersion must be positive")
    if config.store_prior_scale <= 0 or config.product_prior_scale <= 0:
        raise ValueError("Prior scales must be positive")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    for row in observations:
        if not isinstance(row.store_id, str) or not row.store_id.strip():
            raise ValueError("store_id must be a non-empty string")
        if not isinstance(row.product_id, str) or not row.product_id.strip():
            raise ValueError("product_id must be a non-empty string")
        if (
            isinstance(row.sales, bool)
            or not isinstance(row.sales, int)
            or row.sales < 0
        ):
            raise ValueError("Sales must be a non-negative whole number")
        if row.available_stock is not None and (
            isinstance(row.available_stock, bool)
            or not isinstance(row.available_stock, int)
            or row.available_stock < 0
        ):
            raise ValueError(
                "Available stock must be a non-negative whole number or None"
            )
        if row.available_stock is not None and row.sales > row.available_stock:
            raise ValueError("Sales cannot exceed available stock")
    usable = [
        row
        for row in observations
        if row.available_stock is not None and row.available_stock > 0
    ]
    if not usable:
        return HierarchicalDemandFit(
            "insufficient_information",
            None,
            {},
            {},
            None,
            0,
            0,
            "No positive-stock observations were supplied.",
        )
    uncensored = []
    for row in usable:
        available = row.available_stock
        if available is not None and row.sales < available:
            uncensored.append(row)
    if not uncensored:
        return HierarchicalDemandFit(
            "insufficient_information",
            None,
            {},
            {},
            None,
            len(usable),
            len(usable),
            "All positive-stock observations are censored; demand is not identifiable.",
        )

    stores = tuple(sorted({row.store_id for row in usable}))
    products = tuple(sorted({row.product_id for row in usable}))
    store_index = {value: index for index, value in enumerate(stores)}
    product_index = {value: index for index, value in enumerate(products)}
    sales_mean = max(
        0.1,
        sum(row.sales for row in uncensored) / len(uncensored),
    )
    initial = np.zeros(1 + len(stores) + len(products), dtype=float)
    initial[0] = log(sales_mean)

    def unpack(parameters: Any) -> tuple[float, Any, Any]:
        store_effects = parameters[1 : 1 + len(stores)]
        product_effects = parameters[1 + len(stores) :]
        return (
            float(parameters[0]),
            store_effects - np.mean(store_effects),
            product_effects - np.mean(product_effects),
        )

    def objective(parameters: Any) -> float:
        intercept, store_effects, product_effects = unpack(parameters)
        means = [
            exp(
                intercept
                + store_effects[store_index[row.store_id]]
                + product_effects[product_index[row.product_id]]
            )
            for row in usable
        ]
        likelihood = censored_negative_binomial_log_likelihood(
            usable, means, config.dispersion
        )
        penalty = sum(value * value for value in store_effects) / (
            2 * config.store_prior_scale**2
        ) + sum(value * value for value in product_effects) / (
            2 * config.product_prior_scale**2
        )
        return -likelihood + penalty

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        options={"maxiter": config.max_iterations},
    )
    if not result.success or not isfinite(float(result.fun)):
        return HierarchicalDemandFit(
            "fit_failed",
            None,
            {},
            {},
            None,
            len(usable),
            len(usable) - len(uncensored),
            "MAP optimization did not converge to a finite solution.",
        )
    intercept, store_effects, product_effects = unpack(result.x)
    try:
        inverse = result.hess_inv.todense()
        inverse_array = np.asarray(inverse, dtype=float)
        if not np.all(np.isfinite(inverse_array)):
            covariance = None
        else:
            covariance = tuple(
                tuple(float(value) for value in row) for row in inverse_array
            )
    except (AttributeError, ValueError, np.linalg.LinAlgError):
        try:
            inverse_array = np.asarray(result.hess_inv, dtype=float)
            covariance = (
                tuple(tuple(float(value) for value in row) for row in inverse_array)
                if np.all(np.isfinite(inverse_array))
                else None
            )
        except (AttributeError, ValueError):
            covariance = None
    if covariance is not None and len(covariance) != 1 + len(stores) + len(products):
        covariance = None
    return HierarchicalDemandFit(
        "estimated",
        intercept,
        {store: float(store_effects[index]) for index, store in enumerate(stores)},
        {
            product: float(product_effects[index])
            for index, product in enumerate(products)
        },
        float(result.fun),
        len(usable),
        len(usable) - len(uncensored),
        "Some observations were excluded because availability was unknown or zero."
        if len(usable) != len(observations)
        else None,
        covariance,
    )


def make_synthetic_hierarchical_dataset(
    stores: Sequence[str] = ("store_a", "store_b", "store_c"),
    products: Sequence[str] = ("product_a", "product_b", "product_c"),
    days: int = 30,
    seed: int = 42,
    dispersion: float = 4.0,
) -> SyntheticDemandDataset:
    """Generate reproducible multi-store, multi-product censored histories."""
    if not stores or not products or days < 1:
        raise ValueError("Stores, products, and days must be non-empty")
    if dispersion <= 0:
        raise ValueError("Dispersion must be positive")
    rng = np.random.default_rng(seed)
    store_effects = {
        store: (index - (len(stores) - 1) / 2) * 0.18
        for index, store in enumerate(stores)
    }
    product_effects = {
        product: (index - (len(products) - 1) / 2) * 0.22
        for index, product in enumerate(products)
    }
    observations: list[CensoredDemandObservation] = []
    latent: list[int] = []
    for day in range(days):
        for store in stores:
            for product in products:
                mean = exp(1.7 + store_effects[store] + product_effects[product])
                probability = dispersion / (dispersion + mean)
                demand = int(rng.negative_binomial(dispersion, probability))
                available = 4 + (day + len(store) + len(product)) % 10
                observations.append(
                    CensoredDemandObservation(
                        store, product, min(demand, available), available
                    )
                )
                latent.append(demand)
    return SyntheticDemandDataset(
        tuple(observations), tuple(latent), store_effects, product_effects
    )


def evaluate_hierarchical_holdout(
    dataset: SyntheticDemandDataset,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> HierarchicalHoldoutSummary:
    """Compare hierarchical MAP predictions with sales mean on a deterministic split."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if len(dataset.observations) != len(dataset.latent_demand):
        raise ValueError("Observations and latent demand must have equal lengths")
    if not dataset.observations:
        raise ValueError("At least one observation is required")

    rng = np.random.default_rng(seed)
    pair_count = len(dataset.observations)
    indices = np.arange(pair_count)
    rng.shuffle(indices)
    train_size = max(1, int(pair_count * train_fraction))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    training = tuple(dataset.observations[index] for index in train_indices)
    fit = fit_hierarchical_censored_demand(training)
    hierarchical_errors: list[float] = []
    sales_errors: list[float] = []
    differences: list[float] = []
    interval_hits: list[bool] = []

    for index in test_indices:
        row = dataset.observations[int(index)]
        hidden = float(dataset.latent_demand[int(index)])
        hierarchical = fit.predict_mean(row.store_id, row.product_id)
        sales_history = [
            training_row.sales
            for training_row in training
            if training_row.store_id == row.store_id
            and training_row.product_id == row.product_id
        ]
        sales_mean = sum(sales_history) / len(sales_history) if sales_history else None
        if hierarchical is not None:
            hierarchical_error = abs(hierarchical - hidden)
            hierarchical_errors.append(hierarchical_error)
        else:
            hierarchical_error = None
        if sales_mean is not None:
            sales_errors.append(abs(sales_mean - hidden))
        if hierarchical_error is not None and sales_mean is not None:
            differences.append(hierarchical_error - abs(sales_mean - hidden))
            interval = fit.predict_mean_interval(row.store_id, row.product_id)
            if interval is not None:
                interval_hits.append(interval[0] <= hidden <= interval[1])

    bootstrap_values: list[float] = []
    if differences:
        bootstrap_rng = np.random.default_rng(seed)
        for _ in range(1000):
            sample = bootstrap_rng.choice(
                np.asarray(differences), size=len(differences), replace=True
            )
            bootstrap_values.append(float(np.mean(sample)))
        bootstrap_values.sort()
        bootstrap_interval = (
            bootstrap_values[50],
            bootstrap_values[949],
        )
    else:
        bootstrap_interval = (0.0, 0.0)

    return HierarchicalHoldoutSummary(
        hierarchical_mae=(
            sum(hierarchical_errors) / len(hierarchical_errors)
            if hierarchical_errors
            else 0.0
        ),
        sales_mean_mae=(sum(sales_errors) / len(sales_errors) if sales_errors else 0.0),
        paired_mae_difference=(
            sum(differences) / len(differences) if differences else 0.0
        ),
        bootstrap_mae_difference=bootstrap_interval,
        interval_coverage=(
            sum(interval_hits) / len(interval_hits) if interval_hits else None
        ),
        successful_pairs=len(differences),
    )
