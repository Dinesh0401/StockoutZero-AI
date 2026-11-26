"""Phase 3A hierarchical censored-demand research baseline.

This module deliberately accepts only historical observations.  Future demand,
post-sale inventory, and audit results are not part of the fitting contract.
The model is a transparent MAP baseline, not a production forecaster.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from importlib import import_module
from math import exp, isfinite, log, log1p, sqrt
from pathlib import Path
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
    day: int | None = None


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
    """Fitted effects and explicit estimation status.

    ``dispersion`` is the NB2 size parameter used to build predictive
    intervals for realized demand.
    """

    status: str
    intercept: float | None
    store_effects: dict[str, float]
    product_effects: dict[str, float]
    objective: float | None
    observations_used: int
    censored_observations: int
    warning: str | None = None
    covariance: tuple[tuple[float, ...], ...] | None = None
    dispersion: float = 0.0

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
        """Return a Laplace log-scale interval for the ESTIMATED MEAN.

        This is a parameter-uncertainty interval for the fitted conditional
        mean, not a predictive interval for realized future demand.  Realized
        demand includes observation-level noise and will typically be wider.
        Use :meth:`predict_interval` for predictive coverage of realized demand.
        """
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
        n_store_free = max(len(stores) - 1, 0)
        n_product_free = max(len(products) - 1, 0)
        size = 1 + n_store_free + n_product_free
        if len(self.covariance) != size:
            return None

        gradient = [0.0] * size
        gradient[0] = 1.0  # d/d(intercept)
        store_index = {value: index for index, value in enumerate(stores)}
        product_index = {value: index for index, value in enumerate(products)}
        si = store_index[store_id]
        if len(stores) == 1:
            pass  # single store effect is fixed at 0, no free direction
        elif si == len(stores) - 1:
            for k in range(n_store_free):
                gradient[1 + k] = -1.0  # last = -sum(free)
        else:
            gradient[1 + si] = 1.0
        pi = product_index[product_id]
        offset = 1 + n_store_free
        if len(products) == 1:
            pass
        elif pi == len(products) - 1:
            for j in range(n_product_free):
                gradient[offset + j] = -1.0
        else:
            gradient[offset + pi] = 1.0

        variance = 0.0
        for row in range(size):
            for column in range(size):
                variance += (
                    gradient[row] * self.covariance[row][column] * gradient[column]
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

    def predict_interval(
        self,
        store_id: str,
        product_id: str,
        coverage: float = 0.90,
    ) -> tuple[float, float] | None:
        """Return an NB2 predictive interval for REALIZED demand.

        Uses the fitted conditional mean and dispersion to quantile the
        negative-binomial distribution of the realized count.  This is the
        appropriate quantity for coverage checks against held-out latent
        demand, unlike :meth:`predict_mean_interval`.
        """
        if coverage <= 0 or coverage >= 1:
            raise ValueError("coverage must be between zero and one")
        mu = self.predict_mean(store_id, product_id)
        if mu is None or not isfinite(mu) or mu <= 0:
            return None
        if (
            self.dispersion is None
            or not isfinite(self.dispersion)
            or self.dispersion <= 0
        ):
            return None
        p = self.dispersion / (self.dispersion + mu)
        try:
            lower = float(nbinom.ppf((1.0 - coverage) / 2.0, self.dispersion, p))
            upper = float(nbinom.ppf((1.0 + coverage) / 2.0, self.dispersion, p))
        except (OverflowError, ValueError):
            return None
        if not isfinite(lower) or not isfinite(upper) or lower < 0:
            return None
        return (lower, upper)


@dataclass(frozen=True)
class SyntheticDemandDataset:
    """Synthetic observations with hidden truth retained for evaluation only."""

    observations: tuple[CensoredDemandObservation, ...]
    latent_demand: tuple[int, ...]
    true_store_effects: dict[str, float]
    true_product_effects: dict[str, float]
    true_daily_means: dict[tuple[str, str], float] = field(default_factory=dict)


@dataclass(frozen=True)
class HierarchicalHoldoutSummary:
    """Paired held-out accuracy and interval coverage for Phase 3A."""

    hierarchical_mae: float | None
    sales_mean_mae: float | None
    moving_average_mae: float | None
    paired_mae_difference: float | None
    bootstrap_mae_difference: tuple[float, float] | None
    interval_coverage: float | None
    successful_pairs: int
    failed_pairs: int = 0
    abstained_pairs: int = 0
    paired_denominator: int = 0
    paired_clusters: int = 0
    mean_interval_coverage: float | None = None
    predictive_interval_coverage: float | None = None


@dataclass(frozen=True)
class HierarchicalValidationRun:
    """Phase 3A evidence for one history length and seed."""

    name: str
    hierarchical_mae: float | None
    baseline_maes: dict[str, float | None]
    paired_difference: float | None
    paired_bootstrap_interval: tuple[float, float] | None
    interval_coverage: float | None
    successful_pairs: int
    predictive_interval_coverage: float | None = None


@dataclass(frozen=True)
class HierarchicalValidationMatrix:
    """Aggregate evidence across seeds and history lengths."""

    runs: dict[str, HierarchicalValidationRun]


def _validate_observations(observations: Sequence[CensoredDemandObservation]) -> None:
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


def _fit_censored_nb2_single_mean(
    observations: Sequence[CensoredDemandObservation],
    dispersion: float,
) -> float | None:
    """MLE mean for a single intercept-only censored NB2 over usable rows."""
    usable = [
        row
        for row in observations
        if row.available_stock is not None and row.available_stock > 0
    ]
    if not usable:
        return None
    uncensored = [row for row in usable if row.sales < (row.available_stock or 0)]
    if not uncensored:
        return None
    sales_mean = max(
        0.1,
        sum(row.sales for row in uncensored) / len(uncensored),
    )

    def neg_loglik(x: Any) -> float:
        intercept = float(x[0])
        mean = exp(intercept)
        if not isfinite(mean) or mean <= 0:
            return float("inf")
        try:
            return -censored_negative_binomial_log_likelihood(
                usable, [mean] * len(usable), dispersion
            )
        except ValueError:
            return float("inf")

    result = minimize(
        neg_loglik,
        np.asarray([log(sales_mean)], dtype=float),
        method="L-BFGS-B",
    )
    if not result.success or not isfinite(float(result.fun)):
        return None
    mean = exp(float(result.x[0]))
    if not isfinite(mean) or mean <= 0:
        return None
    return mean


def fit_hierarchical_censored_demand(
    observations: Sequence[CensoredDemandObservation],
    config: HierarchicalDemandConfig | None = None,
) -> HierarchicalDemandFit:
    """Fit pooled store and product effects with a censored NB2 MAP objective.

    Uses an identifiable sum-to-zero (contrast) parameterization: the last
    store and product effect are defined as the negative sum of the free
    effects, so no redundant direction remains and the Hessian is non-singular
    given enough data.

    Records with unknown or zero availability are excluded and counted in the
    warning.  A history containing no uncensored positive-stock observation is
    explicitly non-identifiable and returns an abstention result.
    """
    config = config or HierarchicalDemandConfig()
    if config.dispersion <= 0:
        raise ValueError("Dispersion must be positive")
    if config.store_prior_scale <= 0 or config.product_prior_scale <= 0:
        raise ValueError("Prior scales must be positive")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    _validate_observations(observations)
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
            None,
            config.dispersion,
        )
    uncensored = [row for row in usable if row.sales < (row.available_stock or 0)]
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
            None,
            config.dispersion,
        )

    stores = tuple(sorted({row.store_id for row in usable}))
    products = tuple(sorted({row.product_id for row in usable}))
    store_index = {value: index for index, value in enumerate(stores)}
    product_index = {value: index for index, value in enumerate(products)}
    n_stores = len(stores)
    n_products = len(products)
    n_store_free = max(n_stores - 1, 0)
    n_product_free = max(n_products - 1, 0)
    sales_mean = max(
        0.1,
        sum(row.sales for row in uncensored) / len(uncensored),
    )
    n_free = 1 + n_store_free + n_product_free
    initial = np.zeros(n_free, dtype=float)
    initial[0] = log(sales_mean)

    def effects_from_parameters(
        parameters: Any,
    ) -> tuple[float, list[float], list[float]]:
        intercept = float(parameters[0])
        store_free = parameters[1 : 1 + n_store_free]
        product_free = parameters[1 + n_store_free :]
        if n_stores == 1:
            store_effects = [0.0]
        else:
            store_effects = [float(value) for value in store_free] + [
                -float(np.sum(store_free))
            ]
        if n_products == 1:
            product_effects = [0.0]
        else:
            product_effects = [float(value) for value in product_free] + [
                -float(np.sum(product_free))
            ]
        return intercept, store_effects, product_effects

    def objective(parameters: Any) -> float:
        intercept, store_effects, product_effects = effects_from_parameters(parameters)
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
            None,
            config.dispersion,
        )
    intercept, store_effects, product_effects = effects_from_parameters(result.x)
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
    if covariance is not None and len(covariance) != n_free:
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
        config.dispersion,
    )


def fit_global_pooled_censored_demand(
    observations: Sequence[CensoredDemandObservation],
    config: HierarchicalDemandConfig | None = None,
) -> HierarchicalDemandFit:
    """Intercept-only NB2 censored fit (global pooling, no store/product effects)."""
    config = config or HierarchicalDemandConfig()
    if config.dispersion <= 0:
        raise ValueError("Dispersion must be positive")
    _validate_observations(observations)
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
            None,
            config.dispersion,
        )
    uncensored = [row for row in usable if row.sales < (row.available_stock or 0)]
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
            None,
            config.dispersion,
        )
    mean = _fit_censored_nb2_single_mean(usable, config.dispersion)
    if mean is None:
        return HierarchicalDemandFit(
            "fit_failed",
            None,
            {},
            {},
            None,
            len(usable),
            len(usable) - len(uncensored),
            "Global pooled optimization did not converge to a finite solution.",
            None,
            config.dispersion,
        )
    return HierarchicalDemandFit(
        "estimated",
        log(mean),
        {},
        {},
        None,
        len(usable),
        len(usable) - len(uncensored),
        "Some observations were excluded because availability was unknown or zero."
        if len(usable) != len(observations)
        else None,
        None,
        config.dispersion,
    )


def fit_independent_group_censored_demand(
    observations: Sequence[CensoredDemandObservation],
    config: HierarchicalDemandConfig | None = None,
) -> dict[tuple[str, str], float | None]:
    """Per store-product censored NB2 mean using only that pair's usable rows."""
    config = config or HierarchicalDemandConfig()
    if config.dispersion <= 0:
        raise ValueError("Dispersion must be positive")
    _validate_observations(observations)
    usable = [
        row
        for row in observations
        if row.available_stock is not None and row.available_stock > 0
    ]
    by_pair: dict[tuple[str, str], list[CensoredDemandObservation]] = {}
    for row in usable:
        by_pair.setdefault((row.store_id, row.product_id), []).append(row)
    means: dict[tuple[str, str], float | None] = {}
    for pair, pair_observations in by_pair.items():
        means[pair] = _fit_censored_nb2_single_mean(
            pair_observations, config.dispersion
        )
    return means


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
    true_daily_means: dict[tuple[str, str], float] = {}
    for store in stores:
        for product in products:
            true_daily_means[(store, product)] = exp(
                1.7 + store_effects[store] + product_effects[product]
            )
    for day in range(days):
        for store in stores:
            for product in products:
                mean = exp(1.7 + store_effects[store] + product_effects[product])
                probability = dispersion / (dispersion + mean)
                demand = int(rng.negative_binomial(dispersion, probability))
                available = 4 + (day + len(store) + len(product)) % 10
                observations.append(
                    CensoredDemandObservation(
                        store, product, min(demand, available), available, day=day
                    )
                )
                latent.append(demand)
    return SyntheticDemandDataset(
        tuple(observations),
        tuple(latent),
        store_effects,
        product_effects,
        true_daily_means,
    )


def _chronological_split(
    dataset: SyntheticDemandDataset, train_fraction: float
) -> tuple[list[int], list[int]]:
    """Split observations by day; fall back to index order when no days."""
    if not dataset.observations:
        return [], []
    days_present = [row.day for row in dataset.observations if row.day is not None]
    if not days_present:
        indices = list(range(len(dataset.observations)))
        train_size = max(1, int(len(indices) * train_fraction))
        return indices[:train_size], indices[train_size:]
    unique_days = sorted(set(days_present))
    train_day_count = max(1, int(len(unique_days) * train_fraction))
    train_day_count = min(train_day_count, len(unique_days) - 1)
    if train_day_count < 1:
        train_day_count = 1
    train_days = set(unique_days[:train_day_count])
    train_indices = [
        index for index, row in enumerate(dataset.observations) if row.day in train_days
    ]
    test_indices = [
        index
        for index, row in enumerate(dataset.observations)
        if row.day not in train_days
    ]
    return train_indices, test_indices


def evaluate_hierarchical_holdout(
    dataset: SyntheticDemandDataset,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> HierarchicalHoldoutSummary:
    """Compare hierarchical MAP predictions with sales mean on a chronological split.

    Baselines are fit ONLY on training observations and scored ONLY on test
    latent demand.  No test-period data enters any prediction; no latent
    demand enters fitting.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if len(dataset.observations) != len(dataset.latent_demand):
        raise ValueError("Observations and latent demand must have equal lengths")
    if not dataset.observations:
        raise ValueError("At least one observation is required")

    train_indices, test_indices = _chronological_split(dataset, train_fraction)
    training = tuple(dataset.observations[index] for index in train_indices)
    fit = fit_hierarchical_censored_demand(training)

    training_sales: dict[tuple[str, str], list[int]] = {}
    for index in train_indices:
        row = dataset.observations[index]
        training_sales.setdefault((row.store_id, row.product_id), []).append(row.sales)

    hierarchical_errors: list[float] = []
    sales_errors: list[float] = []
    moving_average_errors: list[float] = []
    paired_differences_by_group: dict[tuple[str, str], list[float]] = {}
    mean_interval_hits: list[bool] = []
    mean_interval_total = 0
    predictive_interval_hits: list[bool] = []
    predictive_interval_total = 0
    failed_pairs = 0
    abstained_pairs = 0

    for index in test_indices:
        row = dataset.observations[index]
        hidden = float(dataset.latent_demand[index])
        hierarchical = fit.predict_mean(row.store_id, row.product_id)
        history = training_sales.get((row.store_id, row.product_id), [])
        sales_mean = sum(history) / len(history) if history else None
        moving_average = sum(history[-7:]) / len(history[-7:]) if history else None

        hierarchical_error: float | None = None
        if hierarchical is not None:
            hierarchical_error = abs(hierarchical - hidden)
            hierarchical_errors.append(hierarchical_error)
        else:
            failed_pairs += 1

        if sales_mean is None:
            abstained_pairs += 1
        else:
            sales_errors.append(abs(sales_mean - hidden))
        if moving_average is not None:
            moving_average_errors.append(abs(moving_average - hidden))

        if hierarchical_error is not None and sales_mean is not None:
            paired_differences_by_group.setdefault(
                (row.store_id, row.product_id), []
            ).append(hierarchical_error - abs(sales_mean - hidden))

        true_mean = dataset.true_daily_means.get((row.store_id, row.product_id))
        mean_interval = fit.predict_mean_interval(row.store_id, row.product_id)
        if mean_interval is not None:
            mean_interval_total += 1
            if (
                true_mean is not None
                and mean_interval[0] <= true_mean <= mean_interval[1]
            ):
                mean_interval_hits.append(True)
            else:
                mean_interval_hits.append(False)
        predictive_interval = fit.predict_interval(row.store_id, row.product_id)
        if predictive_interval is not None:
            predictive_interval_total += 1
            if predictive_interval[0] <= hidden <= predictive_interval[1]:
                predictive_interval_hits.append(True)
            else:
                predictive_interval_hits.append(False)

    cluster_differences = [
        sum(values) / len(values)
        for values in paired_differences_by_group.values()
        if values
    ]
    bootstrap_values: list[float] = []
    if cluster_differences:
        bootstrap_rng = np.random.default_rng(seed)
        for _ in range(1000):
            sample = bootstrap_rng.choice(
                np.asarray(cluster_differences),
                size=len(cluster_differences),
                replace=True,
            )
            bootstrap_values.append(float(np.mean(sample)))
        bootstrap_values.sort()
        bootstrap_interval = (bootstrap_values[50], bootstrap_values[949])
    else:
        bootstrap_interval = None

    paired_denominator = sum(
        len(values) for values in paired_differences_by_group.values()
    )
    paired_difference = (
        sum(cluster_differences) / len(cluster_differences)
        if cluster_differences
        else None
    )
    mean_interval_coverage: float | None = (
        sum(mean_interval_hits) / len(mean_interval_hits)
        if mean_interval_hits
        else None
    )
    predictive_interval_coverage: float | None = (
        sum(predictive_interval_hits) / len(predictive_interval_hits)
        if predictive_interval_hits
        else None
    )

    return HierarchicalHoldoutSummary(
        hierarchical_mae=(
            sum(hierarchical_errors) / len(hierarchical_errors)
            if hierarchical_errors
            else None
        ),
        sales_mean_mae=(
            sum(sales_errors) / len(sales_errors) if sales_errors else None
        ),
        moving_average_mae=(
            sum(moving_average_errors) / len(moving_average_errors)
            if moving_average_errors
            else None
        ),
        paired_mae_difference=(paired_difference),
        bootstrap_mae_difference=bootstrap_interval,
        interval_coverage=mean_interval_coverage,
        successful_pairs=paired_denominator,
        failed_pairs=failed_pairs,
        abstained_pairs=abstained_pairs,
        paired_denominator=paired_denominator,
        paired_clusters=len(cluster_differences),
        mean_interval_coverage=mean_interval_coverage,
        predictive_interval_coverage=predictive_interval_coverage,
    )


def run_hierarchical_validation_matrix(
    history_lengths: Sequence[int] = (30, 90),
    seeds: Sequence[int] = (7, 19, 42),
) -> dict[str, HierarchicalValidationRun]:
    """Run reproducible held-out validation scenarios with training-only baselines."""
    runs: dict[str, HierarchicalValidationRun] = {}
    for history_days in history_lengths:
        if history_days < 3:
            raise ValueError("history_lengths must contain at least three days")
        for seed in seeds:
            dataset = make_synthetic_hierarchical_dataset(
                days=history_days,
                seed=seed,
            )
            summary = evaluate_hierarchical_holdout(dataset, seed=100 + seed)
            ablation = run_pooling_ablation(dataset, seed=100 + seed)
            baseline_maes: dict[str, float | None] = {
                "sales_mean": summary.sales_mean_mae,
                "moving_average": summary.moving_average_mae,
            }
            baseline_maes.update(
                {
                    name: metrics["mae"]
                    for name, metrics in ablation["approaches"].items()
                }
            )

            name = f"{history_days}d_seed{seed}"
            runs[name] = HierarchicalValidationRun(
                name=name,
                hierarchical_mae=summary.hierarchical_mae,
                baseline_maes=baseline_maes,
                paired_difference=summary.paired_mae_difference,
                paired_bootstrap_interval=summary.bootstrap_mae_difference,
                interval_coverage=summary.interval_coverage,
                successful_pairs=summary.successful_pairs,
                predictive_interval_coverage=summary.predictive_interval_coverage,
            )
    return runs


def _evaluate_predictor(
    predict: Any,
    dataset: SyntheticDemandDataset,
    test_indices: Sequence[int],
) -> tuple[float | None, int, int]:
    """Evaluate a predict(store, product) -> float | None callable on test rows.

    Returns (mae, failures, count) where failures is the number of None
    predictions and count is the number of non-None predictions.
    """
    errors: list[float] = []
    failures = 0
    count = 0
    for index in test_indices:
        row = dataset.observations[index]
        hidden = float(dataset.latent_demand[index])
        forecast = predict(row.store_id, row.product_id)
        if forecast is None or not isfinite(forecast):
            failures += 1
            continue
        errors.append(abs(forecast - hidden))
        count += 1
    mae = sum(errors) / len(errors) if errors else None
    return mae, failures, count


def run_pooling_ablation(
    dataset: SyntheticDemandDataset,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> dict:
    """Compare pooling strategies on the same chronological train/test split.

    Returns MAE per approach, failure/abstention counts, paired differences
    versus global pooled, and a short conclusion.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if len(dataset.observations) != len(dataset.latent_demand):
        raise ValueError("Observations and latent demand must have equal lengths")
    if not dataset.observations:
        raise ValueError("At least one observation is required")
    train_indices, test_indices = _chronological_split(dataset, train_fraction)
    training = tuple(dataset.observations[index] for index in train_indices)

    global_fit = fit_global_pooled_censored_demand(training)
    global_predict = (
        global_fit.predict_mean
        if global_fit.status == "estimated"
        else lambda store_id, product_id: None
    )
    global_mae, global_failures, global_count = _evaluate_predictor(
        global_predict, dataset, test_indices
    )

    independent_means = fit_independent_group_censored_demand(training)

    def independent_predict(store_id: str, product_id: str) -> float | None:
        return independent_means.get((store_id, product_id))

    independent_mae, independent_failures, independent_count = _evaluate_predictor(
        independent_predict, dataset, test_indices
    )

    weak_config = HierarchicalDemandConfig(
        store_prior_scale=1.5,
        product_prior_scale=1.5,
    )
    weak_fit = fit_hierarchical_censored_demand(training, weak_config)
    weak_mae, weak_failures, weak_count = _evaluate_predictor(
        weak_fit.predict_mean, dataset, test_indices
    )

    strong_config = HierarchicalDemandConfig(
        store_prior_scale=0.25,
        product_prior_scale=0.25,
    )
    strong_fit = fit_hierarchical_censored_demand(training, strong_config)
    strong_mae, strong_failures, strong_count = _evaluate_predictor(
        strong_fit.predict_mean, dataset, test_indices
    )

    grouped_differences: dict[str, dict[tuple[str, str], list[float]]] = {
        "independent": {},
        "hierarchical_weak": {},
        "hierarchical_strong": {},
    }
    for index in test_indices:
        row = dataset.observations[index]
        hidden = float(dataset.latent_demand[index])
        global_forecast = global_predict(row.store_id, row.product_id)
        if global_forecast is None:
            continue
        global_error = abs(global_forecast - hidden)
        independent_forecast = independent_predict(row.store_id, row.product_id)
        if independent_forecast is not None:
            grouped_differences["independent"].setdefault(
                (row.store_id, row.product_id), []
            ).append(abs(independent_forecast - hidden) - global_error)
        weak_forecast = weak_fit.predict_mean(row.store_id, row.product_id)
        if weak_forecast is not None:
            grouped_differences["hierarchical_weak"].setdefault(
                (row.store_id, row.product_id), []
            ).append(abs(weak_forecast - hidden) - global_error)
        strong_forecast = strong_fit.predict_mean(row.store_id, row.product_id)
        if strong_forecast is not None:
            grouped_differences["hierarchical_strong"].setdefault(
                (row.store_id, row.product_id), []
            ).append(abs(strong_forecast - hidden) - global_error)

    cluster_differences = {
        name: [sum(values) / len(values) for values in groups.values() if values]
        for name, groups in grouped_differences.items()
    }

    def mean_value(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def bootstrap_interval(
        values: Sequence[float], seed_value: int
    ) -> tuple[float, float] | None:
        if not values:
            return None
        rng = np.random.default_rng(seed_value)
        samples = [
            float(
                np.mean(
                    rng.choice(
                        np.asarray(values),
                        size=len(values),
                        replace=True,
                    )
                )
            )
            for _ in range(1000)
        ]
        samples.sort()
        return (samples[50], samples[949])

    approaches = {
        "global_pooled": {
            "mae": global_mae,
            "failures": global_failures,
            "estimates": global_count,
        },
        "independent": {
            "mae": independent_mae,
            "failures": independent_failures,
            "estimates": independent_count,
        },
        "hierarchical_weak": {
            "mae": weak_mae,
            "failures": weak_failures,
            "estimates": weak_count,
        },
        "hierarchical_strong": {
            "mae": strong_mae,
            "failures": strong_failures,
            "estimates": strong_count,
        },
    }
    paired_difference_vs_global = {}
    for offset, name in enumerate(
        ("independent", "hierarchical_weak", "hierarchical_strong")
    ):
        paired_difference_vs_global[name] = {
            "mean_difference": mean_value(cluster_differences[name]),
            "bootstrap_interval": bootstrap_interval(
                cluster_differences[name], seed + offset
            ),
            "paired_count": sum(
                len(values) for values in grouped_differences[name].values()
            ),
            "paired_clusters": len(cluster_differences[name]),
        }

    return {
        "approaches": approaches,
        "paired_difference_vs_global_pooled": paired_difference_vs_global,
        "conclusion": "evidence_reported_no_arbitrary_superiority_rule",
        "train_fraction": train_fraction,
        "seed": seed,
        "config": {
            "global_pooled": asdict(HierarchicalDemandConfig()),
            "independent": asdict(HierarchicalDemandConfig()),
            "hierarchical_weak": asdict(weak_config),
            "hierarchical_strong": asdict(strong_config),
        },
        "test_pairs": len(test_indices),
    }


def run_reproducible_study(
    output_path: str = "phase3a_study_results.json",
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    history_lengths: Sequence[int] = (30, 90),
    train_fraction: float = 0.7,
) -> list[dict[str, Any]]:
    """Run multiple seeds/history lengths and write a machine-readable study.

    Records are written to JSON (and a flattened CSV sibling) but the function
    is not invoked by the test suite.
    """
    if not seeds:
        raise ValueError("At least one seed is required")
    if not history_lengths or any(days < 3 for days in history_lengths):
        raise ValueError("history_lengths must contain at least three days")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    trial_id = 0
    for history_days in history_lengths:
        for seed in seeds:
            start = time.perf_counter()
            dataset = make_synthetic_hierarchical_dataset(days=history_days, seed=seed)
            train_indices, test_indices = _chronological_split(dataset, train_fraction)
            training = tuple(dataset.observations[index] for index in train_indices)
            fit_config = HierarchicalDemandConfig()
            fit = fit_hierarchical_censored_demand(training, fit_config)
            summary = evaluate_hierarchical_holdout(
                dataset, train_fraction=train_fraction, seed=100 + seed
            )
            ablation = run_pooling_ablation(
                dataset, train_fraction=train_fraction, seed=seed
            )
            runtime = time.perf_counter() - start

            forecasts: dict[str, float | None] = {}
            targets: dict[str, float] = {}
            for index in test_indices:
                row = dataset.observations[index]
                key = f"{row.store_id}|{row.product_id}|day{row.day}"
                forecasts[key] = fit.predict_mean(row.store_id, row.product_id)
                targets[key] = float(dataset.latent_demand[index])

            record: dict[str, Any] = {
                "trial_id": trial_id,
                "seed": seed,
                "history_days": history_days,
                "config": {
                    "train_fraction": train_fraction,
                    "dispersion": fit.dispersion,
                    "store_prior_scale": fit_config.store_prior_scale,
                    "product_prior_scale": fit_config.product_prior_scale,
                    "max_iterations": fit_config.max_iterations,
                    "bootstrap_seed": 100 + seed,
                },
                "train_test_boundaries": {
                    "train_days": len(train_indices),
                    "test_days": len(test_indices),
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                },
                "model_identifiers": ["hierarchical_map"],
                "baseline_identifiers": [
                    "sales_mean",
                    "moving_average",
                    "global_pooled",
                    "independent",
                ],
                "model_status": fit.status,
                "forecast": forecasts,
                "ground_truth_target": targets,
                "error_metrics": {
                    "hierarchical_mae": summary.hierarchical_mae,
                    "sales_mean_mae": summary.sales_mean_mae,
                    "moving_average_mae": summary.moving_average_mae,
                    "paired_mae_difference": summary.paired_mae_difference,
                    "pooling_ablations": {
                        name: metrics["mae"]
                        for name, metrics in ablation["approaches"].items()
                    },
                },
                "coverage_metrics": {
                    "interval_coverage": summary.interval_coverage,
                    "mean_interval_coverage": summary.mean_interval_coverage,
                    "predictive_interval_coverage": (
                        summary.predictive_interval_coverage
                    ),
                },
                "paired_denominator": summary.paired_denominator,
                "paired_clusters": summary.paired_clusters,
                "bootstrap_interval": (
                    list(summary.bootstrap_mae_difference)
                    if summary.bootstrap_mae_difference is not None
                    else None
                ),
                "runtime_seconds": runtime,
                "numerical_failures": {
                    "failed_pairs": summary.failed_pairs,
                    "abstained_pairs": summary.abstained_pairs,
                    "fit_status": fit.status,
                },
                "pooling_ablation": ablation,
            }
            records.append(record)
            trial_id += 1

    with destination.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, default=str)

    csv_path = destination.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trial_id",
                "seed",
                "history_days",
                "model_status",
                "hierarchical_mae",
                "sales_mean_mae",
                "moving_average_mae",
                "global_pooled_mae",
                "independent_mae",
                "hierarchical_weak_mae",
                "hierarchical_strong_mae",
                "paired_mae_difference",
                "mean_interval_coverage",
                "predictive_interval_coverage",
                "paired_denominator",
                "paired_clusters",
                "failed_pairs",
                "abstained_pairs",
                "runtime_seconds",
                "conclusion",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["trial_id"],
                    record["seed"],
                    record["history_days"],
                    record["model_status"],
                    record["error_metrics"]["hierarchical_mae"],
                    record["error_metrics"]["sales_mean_mae"],
                    record["error_metrics"]["moving_average_mae"],
                    record["error_metrics"]["pooling_ablations"]["global_pooled"],
                    record["error_metrics"]["pooling_ablations"]["independent"],
                    record["error_metrics"]["pooling_ablations"]["hierarchical_weak"],
                    record["error_metrics"]["pooling_ablations"]["hierarchical_strong"],
                    record["error_metrics"]["paired_mae_difference"],
                    record["coverage_metrics"]["mean_interval_coverage"],
                    record["coverage_metrics"]["predictive_interval_coverage"],
                    record["paired_denominator"],
                    record["paired_clusters"],
                    record["numerical_failures"]["failed_pairs"],
                    record["numerical_failures"]["abstained_pairs"],
                    record["runtime_seconds"],
                    record["pooling_ablation"]["conclusion"],
                ]
            )
    return records
