# Phase 3 Technical Design

## Status

Phase 3 is a research milestone. It must not be implemented as a production
forecast until Phase 2 evidence and review are complete. Draft Pull Request #3
is merged, and Phase 3A now contains the first research-only implementation.

## Phase 3A implementation boundary

`backend/research/hierarchical_demand.py` implements the initial NumPy/SciPy
MAP baseline with a Negative Binomial (NB2) mean/dispersion parameterization.
Store and product effects are centered and regularized for partial pooling.
Only historical observations with known positive beginning-of-demand
availability enter the likelihood:

- `S < A`: Negative Binomial probability mass at observed sales.
- `S = A > 0`: Negative Binomial survival probability at the stock boundary.
- `A = 0` or unknown: excluded and reported, never interpreted as zero demand.

The fit returns an explicit `insufficient_information` state for all-censored
histories and `fit_failed` for non-finite or unsuccessful optimization. This
baseline does not infer uncertain stock, perform audits, or expose a production
API. Synthetic data and tests are provided for reproducibility and comparison
work; hidden latent demand remains evaluator-only.

## Research question

Can jointly modeling latent customer demand and unreliable stock availability,
combined with cost-aware inventory audits, improve replenishment decisions for
small retailers operating with sparse histories?

## Component A: Hierarchical censored demand

### Initial model

For store `s`, product `i`, and day `t`, latent demand is modeled as:

```text
D[i,t,s] ~ NegativeBinomial(mu[i,t,s], phi[i,s])
log(mu[i,t,s]) =
    alpha_category
  + u_store
  + v_product
  + beta_weekday
  + beta_promotion
  + beta_seasonality
```

Observed sales follow:

```text
S[i,t,s] = min(D[i,t,s], A_available[i,t,s])
```

`A_available` explicitly means stock that is sellable immediately before
customer demand occurs, not end-of-day inventory.

The first implementation will use the tractable censored-likelihood form:

- `S < A`: observe `P(D = S)`.
- `S = A > 0`: observe `P(D >= A)`.
- `A = 0`: this observation is excluded unless an availability model provides
  an identifiable nonzero-stock probability.
- `A = unknown`: the observation is excluded from demand fitting until
  availability inference is implemented.

The initial hierarchy will pool product intercepts inside categories and store
effects inside organizations. Promotions and seasonality are added only when
their data-quality requirements are met.

### Inference strategy

1. Implement a NumPy/SciPy maximum-a-posteriori baseline.
2. Validate gradients and censored log-probabilities with controlled tests.
3. Add PyMC posterior inference after the MAP implementation is reproducible.
4. Preserve posterior predictive samples instead of replacing uncertainty with
   plug-in quantiles.

### Identifiability

The model is not identifiable from sales and unknown stock alone. The design
requires known availability, censoring markers, audit observations, and
sufficient variation in availability. Failed or weakly identified fits must
produce explicit abstention states.

## Component B: Uncertain stock availability

### State model

Actual sellable inventory will be a latent count. The design separates
inventory states and event timestamps:

```text
B[i,t,s] = beginning-of-day recorded sellable inventory
R[i,t,s] = supplier receipts with arrival times <= demand time
P[i,t,s] = purchases already received but not yet reflected in the ledger
E[i,t,s] = stock expired or otherwise unusable during the day
X[i,t,s] = shrinkage, damage, or unexplained loss
T[i,t,s] = confirmed returns entering sellable stock
U[i,t,s] = audit correction applied to the state

A_before_demand[i,t,s] =
    clamp(B[i,t,s] + R[i,t,s] + P[i,t,s] + T[i,t,s] + U[i,t,s] - E[i,t,s] - X[i,t,s],
          0,
          maximum_case_count)

S[i,t,s] = min(D[i,t,s], A_before_demand[i,t,s])
A_end[i,t,s] = A_before_demand[i,t,s] - S[i,t,s]
```

The forecast decision is made before demand and uses only:

- Beginning-of-day inventory.
- Receipts known before the decision timestamp.
- Outstanding purchase orders.
- Historical sales, stock events, and audits.

Actual demand, same-day post-demand stock, and later audit observations must
not enter the decision-time forecast. Lost demand is not inferred from end-day
sales unless a stockout censoring model is explicit.

The first state representation will use an audited-or-inferred availability
distribution rather than inventing exact stock quantities.

Each observation must carry one of these states:

- `confirmed_available`
- `confirmed_zero`
- `unknown`
- `possibly_out_of_stock`
- `audited`

### Inference inputs

- Stock-event ledger.
- Recorded sales.
- Supplier receipt quantities and timestamps.
- Manual stock-audit measurements.
- Reconciliation adjustments.
- Data-quality and missing-update indicators.

### Tractability

The first version will use a constrained count-state filter or probabilistic
inventory reconciliation. Neural and particle methods are not required until
a transparent probabilistic implementation has established a baseline and
identified its failure modes.

## Component C: Cost-aware stock audits

An audit is recommended only if:

```text
expected_inventory_loss_without_audit
- expected_inventory_loss_with_audit
> audit_cost
```

### Policies

- Never audit.
- Audit every day.
- Random auditing.
- Uncertainty-threshold auditing.
- Cost-aware value-of-information auditing.

### Decision outputs

The audit decision must include:

- Estimated expected benefit.
- Audit cost.
- Probability of stockout.
- Demand uncertainty.
- Inventory-record uncertainty.
- `recommend`, `defer`, or `insufficient_information` status.

The system must remain a human decision aid. It must not automatically count
stock or place orders in the MVP.

## Data contracts

The implementation will define explicit domain types for:

```text
SalesObservation
StockEvent
StockAudit
SupplierReceipt
DemandForecast
InventoryRecommendation
AuditRecommendation
ExperimentTrial
```

No merchant identifier or business record may cross tenant boundaries. APIs
will preserve tenant ID, product ID, time zone, units, and source provenance.

## Baselines and ablations

Establish identical evaluation streams for:

- Sales mean.
- Moving average.
- Croston.
- TSB.
- Uncensored Poisson.
- Censored Poisson.
- Hierarchical censored Poisson.
- Hierarchical censored Negative Binomial.
- Hierarchical model with uncertain stock.
- Audit-aware inventory policy.

Independent ablations remove:

- Hierarchical pooling.
- Availability uncertainty.
- Cost-aware audit selection.
- Audit observations.

## Acceptance criteria

- Identifiable synthetic scenarios report posterior or interval coverage.
- Non-identifiable scenarios return abstentions.
- Hidden latent demand is unavailable to forecasting functions.
- Every competing policy receives the same trial and future demand stream.
- Audit, replenishment, and stockout decisions preserve outstanding orders.
- Paired cost and fill-rate differences include uncertainty intervals.
- Results are saved with configuration, seed, trial, failure, and ablations.
- Tests cover sparse histories, stockouts, missing updates, audit errors,
  censoring boundaries, numerical failures, and non-identifiability.

## Phase 3A validation status

The implementation now includes:

- Strict validation for non-empty store/product identifiers, integer sales,
  non-negative availability, and sales not exceeding availability.
- Finite Laplace-style predictive intervals with explicit return of no interval
  when covariance or overflow information is unreliable.
- Deterministic held-out synthetic comparison against a sales mean baseline.
- Paired MAE differences and bootstrap uncertainty for the synthetic comparison.
- Hidden latent demand retained only in evaluator datasets; fitting receives
  observed sales and known availability only.
- A reproducible validation-matrix API reports held-out performance against
  sales mean and moving average baselines across configured seeds and history
  lengths, with paired bootstrap intervals and interval coverage.

This work does not yet establish statistically demonstrated pooling benefit or
predictive calibration across a full multiseed experiment matrix. Approval also
requires independent baselines, identifiable synthetic coverage studies, and
real-world merchant validation. The held-out evaluator is evidence scaffolding,
not proof of model superiority.

An initial reproducible six-run synthetic matrix produced the following paired
MAE differences (hierarchical minus sales mean) and interval coverage:

| History | Seed | Hierarchical MAE | Sales mean MAE | Paired difference | 90% bootstrap interval | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 days | 7 | 3.6323 | 3.3400 | +0.2923 | [-0.0025, 0.5737] | 85.19% |
| 30 days | 19 | 3.1438 | 3.0089 | +0.1349 | [-0.0240, 0.3103] | 82.72% |
| 30 days | 42 | 2.9094 | 3.0766 | -0.1672 | [-0.3583, 0.0243] | 92.59% |
| 90 days | 7 | 2.7242 | 2.5661 | +0.1581 | [0.0385, 0.2627] | 88.89% |
| 90 days | 19 | 2.8428 | 2.9248 | -0.0819 | [-0.1929, 0.0373] | 89.71% |
| 90 days | 42 | 2.8339 | 3.0222 | -0.1884 | [-0.2974, -0.0825] | 86.83% |

Negative paired differences favor the hierarchical model. The matrix is mixed:
some intervals exclude zero favorably, some exclude zero unfavorably, and others
cross zero. It is therefore scientific evidence for further testing, not a
merger or production-readiness claim.

## Remaining limitations

This design does not claim an original method or superiority. Prior-art review,
licensing, merchant consent, independent reference implementations, and
real-world validation are required before making publication or deployment
claims.
