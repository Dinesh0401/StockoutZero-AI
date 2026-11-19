# StockoutZero AI Research Protocol

## Baseline
The starter uses a stationary Poisson censored-likelihood fit. It is a demonstration, not a validated flagship model.

## Inputs
Each observation must contain recorded sales and may contain sellable stock available. Unknown stock availability must remain null rather than zero.

## Metrics
Evaluate latent-demand MAE, predictive interval coverage, stockout rate, fill rate, excess inventory, inventory cost, and audit value.

## Required comparisons
- Sales mean
- Moving average
- Croston / TSB
- Uncensored count model
- Censored likelihood
- Hierarchical censored model
- Uncertain-availability model

## Limitations
The Poisson likelihood excludes hierarchical, promotional, temporal, and uncertain-stock effects. Synthetic results must report excluded trials and uncertainty.
