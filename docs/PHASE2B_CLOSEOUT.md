# Phase 2B Closeout Review

## Status

Phase 2B is ready for human review and should remain in Draft Pull Request #3.
It must not be merged automatically.

## Completed work

- Added a common case-based interface for sales mean, moving average, Croston,
  TSB, uncensored Poisson, and censored Poisson baselines.
- Added deterministic 30-day and 90-day synthetic benchmark generation.
- Added Poisson and Negative Binomial demand-process options.
- Added intermittent-demand, demand-level, and availability-profile settings.
- Added separate 14-day future-demand evaluation windows.
- Added failure and success coverage reporting.
- Added paired MAE and inventory-cost comparisons.
- Added supplier lead times, outstanding purchase orders, holding, purchase,
  and stockout costs.
- Added whole-case purchase rounding in inventory simulations.
- Added abstention states and explicit fallback reporting.
- Added Ruff format, Ruff lint, mypy, and Pytest GitHub Actions gates.

## Quality verification

The latest reviewed implementation passed 49 Pytest tests. Ruff format,
Ruff lint, mypy, and GitHub Actions also passed on the exact feature-branch
commits.

## Code changes

The final case-pack correction computes:

```text
shortage = target_inventory - inventory_position
order_units = ceil(shortage / case_pack) * case_pack
```

The case-pack parameter is propagated through forecast-policy and
inventory-policy helpers. Rolling forecasts receive only historical observations
and preserve future demand as evaluator-only data.

## Remaining scientific limitations

- Synthetic performance is not real-world evidence.
- Profile-likelihood intervals are parameter uncertainty, not automatically
  calibrated future-demand prediction intervals.
- Intermittent and censored results require more statistically adequate seeds,
  trials, reference implementations, and coverage validation.
- TSB and Croston should be tested against independent reference software.
- Inventory comparisons do not yet update each baseline with a fully realistic
  merchant lead-time and purchasing policy across every experiment.
- The hierarchical demand and uncertain-stock models have not been implemented.

## Merge recommendation

Merge only after explicit human approval of Draft Pull Request #3 and final
review of this closeout. The next implementation milestone is a separate
Phase 3A branch created from updated `main`, not the Phase 2B branch.
