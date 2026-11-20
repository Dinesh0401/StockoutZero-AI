# Clean Code and Documentation Standards

This document specifies conventions for the **StockoutZero AI** research starter and its future Python API and React/TypeScript frontend. The current runtime is Python + FastAPI + plain browser JavaScript.

## 1. Architecture: separate responsibilities

Current layout:

```text
backend/
  app/
    main.py                  # FastAPI routes and request schemas (existing)
    forecast.py              # Baseline demand estimation and reorder logic (existing)
    static/index.html        # Small plain-JavaScript dashboard (existing)
  research/simulate.py       # Synthetic benchmark (existing)
  tests/
docs/
```

Do **not** rearrange files only to look enterprise-grade. As the code actually grows, migrate in small tested steps toward:

```text
backend/
  app/
    api/                     # HTTP routes, validation, response schemas
    domain/                  # Sales/stock concepts, explicit forecast states
    services/                # Use cases and coordination
    forecasting/             # Pure algorithms + probability utilities
    inventory/               # Human-approved reorder decision logic
    infrastructure/          # Databases and external service adapters
  research/
    baselines/               # Comparable reference estimators
    datasets/                # Loaders and data contracts
    experiments/             # Reproducible configs and evaluation
  tests/
    unit/
    integration/
    research/
frontend/                    # Add only after choosing React + TypeScript
docs/
```

Dependency direction: API and infrastructure depend on domain interfaces; the domain should not import FastAPI, SQLAlchemy, or UI code. Forecasting code should accept typed data and return typed results without network access or file writes.

## 2. Python style: typing, pydoc and maintainability

Use Python 3.11+ syntax when the runtime supports it; PEP 8 naming and PEP 257 docstring conventions; **Google-style docstring sections**. `pydoc` is the Python documentation renderer, not a special docstring format.

```python
from dataclasses import dataclass
from enum import Enum


class EstimateStatus(str, Enum):
    """Explain whether evidence supports an inventory estimate."""

    READY = "ready"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class DemandEstimate:
    """Represent a point estimate without implying calibrated uncertainty.

    Attributes:
        status: Whether the data support reporting a numeric estimate.
        mean_units_per_day: Expected daily demand, if identifiable.
        warning: Human-readable caveat for a store operator.
    """

    status: EstimateStatus
    mean_units_per_day: float | None
    warning: str | None = None


def calculate_reorder_units(
    target_units: int,
    on_hand_units: int,
    case_pack_units: int,
) -> int:
    """Round a nonnegative reorder requirement to whole case packs.

    Args:
        target_units: Desired inventory position, measured in whole units.
        on_hand_units: Current available inventory in whole units.
        case_pack_units: Number of units in one supplier case pack.

    Returns:
        Number of units to order, as a case-pack multiple.

    Raises:
        ValueError: If any quantity is negative or case pack is below one.

    Notes:
        This pure arithmetic helper does not account for outstanding
        purchase orders, lead-time uncertainty, or holding costs.
    """
    if target_units < 0 or on_hand_units < 0 or case_pack_units < 1:
        raise ValueError("Invalid inventory or case-pack quantity")

    shortage = max(0, target_units - on_hand_units)
    return ((shortage + case_pack_units - 1) // case_pack_units) * case_pack_units
```

This is an **illustrative style example**, not a claim that the current API exports these classes.

Key rules:

- Keep functions cohesive and dependency-injected when external services are needed.
- Represent count data as integers; define units in names, schemas, and documentation.
- Handle unidentifiable estimates explicitly rather than silently interpreting them as zero demand.
- Document statistical assumptions and warnings, especially censoring semantics and temporal granularity.
- Prefer typed result objects over heterogeneous dictionaries for internal services; serialize with Pydantic at the API boundary.
- Write clear error messages and distinguish user input errors from unexpected software defects.
- Prefer NumPy/SciPy stable numerical routines and test edge cases.

Local documentation check from `backend/` (adjust the module as appropriate):

```powershell
python -m pydoc app.forecast
python -m pytest -q
```

## 3. JavaScript JSDoc versus TypeScript TSDoc

The current `index.html` contains regular JavaScript. Use **JSDoc** for nontrivial functions:

```js
/**
 * Get a validated inventory recommendation from the local API.
 * @param {{observations: Array<{sales: number, available_stock: number | null}>, lead_time_days: number, on_hand: number, case_pack: number}} payload
 * @returns {Promise<{suggested_units: number, warning: string | null}>}
 * @throws {Error} When the service rejects the request.
 */
async function fetchRecommendation(payload) {
  const response = await fetch("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error("Recommendation unavailable");
  return response.json();
}
```

When a real TypeScript frontend is added, enable `strict` checking and use TSDoc on exported functions, types, hooks, and services:

```ts
export interface InventoryObservation {
  readonly sales: number;
  readonly availableStock: number | null;
}

/**
 * Returns whether a known daily stock count is compatible with sales.
 *
 * @param observation - A retail sales/availability observation.
 * @returns True if stock is unknown or sales do not exceed available stock.
 * @remarks Unknown availability is not equivalent to a stockout.
 */
export function isValidObservation(
  observation: InventoryObservation,
): boolean {
  return (
    Number.isInteger(observation.sales) &&
    observation.sales >= 0 &&
    (observation.availableStock === null ||
      (Number.isInteger(observation.availableStock) &&
        observation.availableStock >= observation.sales))
  );
}
```

For external API payloads use runtime validation and map snake_case wire fields to camelCase UI fields explicitly; TypeScript types alone do not validate incoming JSON. Avoid unnecessary comments that repeat code.

## 4. Testing and automated checks

Quality gates we should add incrementally (not all are currently installed):

| Stage | Check |
| --- | --- |
| Existing Python tests | `cd backend && python -m pytest -q` |
| Formatting | `ruff format --check .` |
| Linting | `ruff check .` |
| Python types | `mypy` or `pyright` over configured source packages |
| Future TypeScript compile | `tsc --noEmit` with `strict: true` |
| Future frontend lint/format | ESLint and Prettier |
| Research reproducibility | Fixed seeds, temporal splits, recorded configs and outputs |
| PR review | Tests, docstrings, API compatibility and benchmark limitations |

Do not claim checks passed unless actually run. A warning or a rejected dataset is evidence to document, not something to conceal.

## 5. Review checklist

- Does each module have one primary responsibility?
- Are public functions typed and meaningfully documented?
- Are inputs validated without losing the distinction between missing, unknown and zero?
- Are forecast uncertainty and non-identifiability represented honestly?
- Is every changed behavior covered by focused unit and API tests?
- Can an experiment reproduce its inputs, seed, metrics and result?
- Are business decisions explainable and human-approved?
- Is a larger architectural refactor justified by actual code growth?
