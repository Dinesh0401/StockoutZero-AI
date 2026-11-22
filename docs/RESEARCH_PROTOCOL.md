# StockoutZero AI Research Protocol

## Baseline
The starter uses a stationary Poisson censored-likelihood fit with SciPy's stable probability functions and bounded optimization. It is a demonstration, not a validated flagship model.

The estimator returns `insufficient_information` rather than a forecast when every usable observation is censored or only confirms zero availability. A reorder recommendation is withheld in that state.

Observed sales and stock quantities are integer counts. Fractional quantities are rejected until a unit-of-measure model is introduced.

## Inputs
Each observation must contain recorded sales and may contain sellable stock available. Unknown stock availability must remain null rather than zero.

## Metrics
Evaluate latent-demand MAE, predictive interval coverage, stockout rate, fill rate, excess inventory, inventory cost, and audit value. Every benchmark must report total trials, successful estimates, failure rate, and accuracy calculated only on successful trials. Bootstrap confidence intervals must be paired with the failure rate rather than hiding failed trials.

## Required comparisons
- Sales mean
- Moving average
- Croston / TSB
- Uncensored count model
- Censored likelihood
- Hierarchical censored model
- Uncertain-availability model

## Phase 2B.1 benchmark
`python -m research.benchmark` evaluates six common baselines on identical
30-day and 90-day training histories against a separate 14-day future window.
It reports success coverage, failure rate, MAE among successful estimates, and
a bootstrap 90% interval for MAE. Paired model comparisons use the same cases
and report bootstrap intervals for the MAE difference.
The benchmark includes a configurable matrix across 30/90-day histories,
Poisson/Negative Binomial demand, low/medium/high demand, intermittent/non-
intermittent demand, and ample/mixed/severe availability. The smoke matrix
covers 72 configurations for one seed. Larger reporting runs use more trials
and independent seeds. It uses separate 14-day future demand windows;
it is not evidence that the censored model generalizes to live retail data.

The sequential inventory simulator in `research.inventory` tracks outstanding
purchase orders, supplier lead time, purchase/holding/stockout costs, and fill
rate. The `simulate_forecast_policy` helper converts model forecasts into
reorder targets and `compare_baseline_inventory_policies` compares model-driven
outcomes. Controlled experiments still need calibrated safety stock, supplier
constraints, and real merchant validation.

## Limitations
The Poisson likelihood excludes hierarchical, promotional, temporal, and
uncertain-stock effects. The current profile-likelihood interval is an
approximate parameter interval, not a calibrated future-demand prediction
interval. Synthetic and inventory conclusions remain demonstrations until
independently reviewed and validated on real or reference datasets.
