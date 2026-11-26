"""Research-only finite-state Bayesian filter for uncertain sellable inventory."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isclose, isfinite, log

from scipy.stats import nbinom, poisson


@dataclass(frozen=True)
class InventoryDayObservation:
    """Events and closing observations for one complete inventory day.

    Receipts and known removals occur before demand. Recorded and audited stock
    are end-of-day counts; an audit is treated as exact when its configured
    error scale is zero.
    """

    sales: int
    receipts: int = 0
    known_removals: int = 0
    recorded_stock: int | None = None
    audited_stock: int | None = None


@dataclass(frozen=True)
class InventoryStateConfig:
    """Fixed assumptions for finite-state inventory reconciliation."""

    max_stock: int
    demand_mean: float
    demand_dispersion: float = 4.0
    daily_shrinkage_mean: float = 0.0
    ledger_error_scale: float = 2.0
    audit_error_scale: float = 0.0


@dataclass(frozen=True)
class InventoryDayEstimate:
    """Posterior sellable-stock estimates for one demand day."""

    expected_available_stock: float
    available_stock_distribution: tuple[float, ...]
    expected_ending_stock: float
    ending_stock_distribution: tuple[float, ...]
    stockout_probability: float
    audit_used: bool


@dataclass(frozen=True)
class InventoryStateEstimate:
    """Sequential posterior estimates; distributions are indexed by unit count."""

    status: str
    days: tuple[InventoryDayEstimate, ...]
    final_stock_distribution: tuple[float, ...]
    warning: str | None


def _validate_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative whole number")


def _normalize_log_weights(log_weights: list[float]) -> list[float] | None:
    finite_weights = [value for value in log_weights if isfinite(value)]
    if not finite_weights:
        return None
    maximum = max(finite_weights)
    weights = [
        exp(value - maximum) if isfinite(value) else 0.0 for value in log_weights
    ]
    total = sum(weights)
    if not isfinite(total) or total <= 0:
        return None
    return [value / total for value in weights]


def _observation_log_likelihood(
    sales: int,
    available: int,
    demand_mean: float,
    dispersion: float,
) -> float:
    probability = dispersion / (dispersion + demand_mean)
    if sales > available:
        return float("-inf")
    if sales < available:
        return float(nbinom.logpmf(sales, dispersion, probability))
    return float(nbinom.logsf(available - 1, dispersion, probability))


def _measurement_log_likelihood(
    observed: int,
    actual: int,
    error_scale: float,
    max_stock: int,
) -> float:
    if error_scale == 0:
        return 0.0 if observed == actual else float("-inf")
    log_likelihoods = [
        -abs(value - actual) / error_scale for value in range(max_stock + 1)
    ]
    maximum = max(log_likelihoods)
    normalizer = maximum + log(sum(exp(value - maximum) for value in log_likelihoods))
    return -abs(observed - actual) / error_scale - normalizer


def _end_stock_distribution(
    availability_distribution: list[float],
    sales: int,
    max_stock: int,
    recorded_stock: int | None,
    audited_stock: int | None,
    config: InventoryStateConfig,
) -> list[float] | None:
    log_weights = [float("-inf")] * (max_stock + 1)
    for available, weight in enumerate(availability_distribution):
        ending = available - sales
        if weight <= 0 or ending < 0:
            continue
        log_weight = log(weight)
        if recorded_stock is not None:
            log_weight += _measurement_log_likelihood(
                recorded_stock,
                ending,
                config.ledger_error_scale,
                max_stock,
            )
        if audited_stock is not None:
            log_weight += _measurement_log_likelihood(
                audited_stock,
                ending,
                config.audit_error_scale,
                max_stock,
            )
        log_weights[ending] = log_weight
    return _normalize_log_weights(log_weights)


def infer_inventory_state(
    observations: tuple[InventoryDayObservation, ...],
    initial_stock_prior: tuple[float, ...],
    config: InventoryStateConfig,
) -> InventoryStateEstimate:
    """Filter latent stock counts using sales, events, ledger counts, and audits.

    The state is sellable units at the start of each day. Known receipts and
    removals occur before demand; unrecorded shrinkage is Poisson distributed.
    Sales are linked to available stock through a fixed NB2 demand model. A
    closing ledger count and audit update the post-sales state before it becomes
    the next day's prior. No future-day observations are used.
    """
    if (
        isinstance(config.max_stock, bool)
        or not isinstance(config.max_stock, int)
        or config.max_stock < 1
    ):
        raise ValueError("max_stock must be at least one")
    if not isfinite(config.demand_mean) or config.demand_mean <= 0:
        raise ValueError("demand_mean must be finite and positive")
    if not isfinite(config.demand_dispersion) or config.demand_dispersion <= 0:
        raise ValueError("demand_dispersion must be finite and positive")
    if not isfinite(config.daily_shrinkage_mean) or config.daily_shrinkage_mean < 0:
        raise ValueError("daily_shrinkage_mean must be finite and non-negative")
    if not isfinite(config.ledger_error_scale) or config.ledger_error_scale < 0:
        raise ValueError("ledger_error_scale must be finite and non-negative")
    if not isfinite(config.audit_error_scale) or config.audit_error_scale < 0:
        raise ValueError("audit_error_scale must be finite and non-negative")
    if len(initial_stock_prior) != config.max_stock + 1:
        raise ValueError("Initial stock prior must cover counts from zero to max_stock")
    if any(
        isinstance(value, bool) or not isfinite(value) or value < 0
        for value in initial_stock_prior
    ):
        raise ValueError("Initial stock probabilities must be finite and non-negative")
    prior_total = sum(initial_stock_prior)
    if prior_total <= 0 or not isclose(prior_total, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("Initial stock probabilities must sum to one")
    if not observations:
        raise ValueError("At least one inventory day is required")

    prior = [value / prior_total for value in initial_stock_prior]
    estimates: list[InventoryDayEstimate] = []
    probability = config.demand_dispersion / (
        config.demand_dispersion + config.demand_mean
    )

    for day_index, observation in enumerate(observations):
        _validate_count(observation.sales, "sales")
        _validate_count(observation.receipts, "receipts")
        _validate_count(observation.known_removals, "known_removals")
        for name, count in (
            ("recorded_stock", observation.recorded_stock),
            ("audited_stock", observation.audited_stock),
        ):
            if count is not None:
                _validate_count(count, name)
                if count > config.max_stock:
                    raise ValueError(f"{name} cannot exceed max_stock")

        log_availability = [float("-inf")] * (config.max_stock + 1)
        for starting_stock, state_probability in enumerate(prior):
            if state_probability == 0:
                continue
            available_before_shrinkage = max(
                starting_stock + observation.receipts - observation.known_removals,
                0,
            )
            if available_before_shrinkage > config.max_stock:
                raise ValueError(
                    "Receipts exceed max_stock; increase the configured state range"
                )
            shrinkage_probabilities = [
                float(poisson.pmf(loss, config.daily_shrinkage_mean))
                for loss in range(available_before_shrinkage + 1)
            ]
            shrinkage_probabilities[-1] = float(
                poisson.sf(
                    available_before_shrinkage - 1,
                    config.daily_shrinkage_mean,
                )
            )
            for loss, loss_probability in enumerate(shrinkage_probabilities):
                available = available_before_shrinkage - loss
                if loss_probability <= 0:
                    continue
                sales_log_likelihood = _observation_log_likelihood(
                    observation.sales,
                    available,
                    config.demand_mean,
                    config.demand_dispersion,
                )
                if not isfinite(sales_log_likelihood):
                    continue
                log_weight = (
                    log(state_probability)
                    + log(loss_probability)
                    + sales_log_likelihood
                )
                previous = log_availability[available]
                if isfinite(previous):
                    larger = max(previous, log_weight)
                    log_availability[available] = larger + log(
                        exp(previous - larger) + exp(log_weight - larger)
                    )
                else:
                    log_availability[available] = log_weight

        availability = _normalize_log_weights(log_availability)
        if availability is None:
            raise ValueError(
                f"Sales on day {day_index} are incompatible with the stock prior"
            )

        ending = _end_stock_distribution(
            availability,
            observation.sales,
            config.max_stock,
            observation.recorded_stock,
            observation.audited_stock,
            config,
        )
        if ending is None:
            raise ValueError(
                f"Closing stock observations on day {day_index} conflict with "
                "the sales and stock model"
            )

        # A closing count and audit can refine the inferred availability as well.
        refined_availability = [0.0] * (config.max_stock + 1)
        for end_stock, end_probability in enumerate(ending):
            available = end_stock + observation.sales
            if available <= config.max_stock:
                refined_availability[available] += end_probability

        expected_available = sum(
            stock * weight for stock, weight in enumerate(refined_availability)
        )
        expected_ending = sum(stock * weight for stock, weight in enumerate(ending))
        stockout_probability = sum(
            weight * float(nbinom.sf(stock, config.demand_dispersion, probability))
            for stock, weight in enumerate(refined_availability)
        )
        estimates.append(
            InventoryDayEstimate(
                expected_available_stock=expected_available,
                available_stock_distribution=tuple(refined_availability),
                expected_ending_stock=expected_ending,
                ending_stock_distribution=tuple(ending),
                stockout_probability=stockout_probability,
                audit_used=observation.audited_stock is not None,
            )
        )
        prior = ending

    audit_count = sum(day.audited_stock is not None for day in observations)
    warning = (
        "Inventory states are conditional on fixed demand and shrinkage assumptions; "
        "no physical audits were supplied."
        if audit_count == 0
        else (
            "Inventory states are conditional on fixed demand and shrinkage "
            "assumptions."
        )
    )
    return InventoryStateEstimate(
        status="estimated",
        days=tuple(estimates),
        final_stock_distribution=tuple(prior),
        warning=warning,
    )
