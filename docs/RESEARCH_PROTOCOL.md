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
- Moving average
- Croston / TSB
- Uncensored count model
- Censored likelihood
- Hierarchical censored model
- Uncertain-availability model

## Limitations
The Poisson likelihood excludes hierarchical, promotional, temporal, and uncertain-stock effects. The simulator currently supports stationary Poisson and Negative Binomial latent demand, but does not yet implement the full 30/90-day benchmark matrix, sequential inventory costs, or calibrated predictive intervals. Synthetic results must report excluded trials and uncertainty.
