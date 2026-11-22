"""Phase 3A hierarchical censored-demand research baseline.

This module deliberately accepts only historical observations.  Future demand,
post-sale inventory, and audit results are not part of the fitting contract.
The model is a transparent MAP baseline, not a production forecaster.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom


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


@dataclass(frozen=True)
class SyntheticDemandDataset:
    """Synthetic observations with hidden truth retained for evaluation only."""

    observations: tuple[CensoredDemandObservation, ...]
    latent_demand: tuple[int, ...]
    true_store_effects: dict[str, float]
    true_product_effects: dict[str, float]


def _negative_binomial_logpmf(
    demand: int, mean: float, dispersion: float
) -> float:
    """Return NB2 log-PMF using mean and size (dispersion) parameters."""
    probability = dispersion / (dispersion + mean)
    return float(
        gammaln(demand + dispersion)
        - gammaln(dispersion)
        - gammaln(demand + 1)
        + dispersion * log(probability)
        + demand * log1p(-probability)
    )


def _negative_binomial_logsf(
    lower_bound: int, mean: float, dispersion: float
) -> float:
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
            total += _negative_binomial_logpmf(
                observation.sales, mean, dispersion
            )
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
    uncensored = [row for row in usable if row.sales < row.available_stock]
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

    def unpack(parameters: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        store_effects = parameters[1 : 1 + len(stores)]
        product_effects = parameters[1 + len(stores) :]
        return (
            float(parameters[0]),
            store_effects - np.mean(store_effects),
            product_effects - np.mean(product_effects),
        )

    def objective(parameters: np.ndarray) -> float:
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
        penalty = (
            sum(value * value for value in store_effects)
            / (2 * config.store_prior_scale**2)
            + sum(value * value for value in product_effects)
            / (2 * config.product_prior_scale**2)
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
    return HierarchicalDemandFit(
        "estimated",
        intercept,
        {
            store: float(store_effects[index]) for index, store in enumerate(stores)
        },
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
