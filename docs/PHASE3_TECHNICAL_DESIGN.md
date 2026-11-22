# Phase 3 Technical Design

## Status

Phase 3 is a research milestone. It must not be implemented as a production
forecast until Phase 2 evidence and review are complete. The implementation
starts only after Draft Pull Request #3 is merged.

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
S[i,t,s] = min(D[i,t,s], A[i,t,s])
```

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

Actual sellable inventory will be a latent count bounded by ledger, returns,
shrinkage, and supplier arrivals:

```text
A[i,t,s] = clamp(
    ledger_opening
  + receipts
  - sales
  - shrinkage
  - expiry
  + returns,
    0,
    maximum_case_count
)
```

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

## Remaining limitations

This design does not claim an original method or superiority. Prior-art review,
licensing, merchant consent, independent reference implementations, and
real-world validation are required before making publication or deployment
claims.
