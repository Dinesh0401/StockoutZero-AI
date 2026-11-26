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

### Phase 3B initial implementation

`backend/research/uncertain_inventory.py` implements a research-only finite-state
Bayesian filter over sellable unit counts. It starts from an explicit
probability distribution and processes each day in event order: opening stock,
known receipts, known removals, Poisson unrecorded shrinkage, censored sales,
and end-of-day book/audit measurements. Sales likelihoods use an NB2 demand
distribution: uncensored sales use the probability mass, while sales equal to
available stock use the survival probability. Exact audits can collapse the
posterior; ledger counts use a bounded, normalized discrete-Laplace error model.
The filter returns expected beginning availability, a full stock-count
distribution, and model-conditional stockout probability for each day.

This is not a fitted availability model or production forecast. Demand mean,
demand dispersion, shrinkage rate, count range, and initial inventory prior
are fixed inputs; outputs are conditional on those assumptions. Impossible
sales/audit combinations fail explicitly. Merchant validation, learned
shrinkage parameters, event timestamp ingestion, and integration into
replenishment decisions remain outstanding.

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
  sales mean, moving average, global-pooled NB2, per-pair NB2, and hierarchical
  pooling variants across configured seeds and history lengths.
- Paired bootstrap intervals resample store-product clusters rather than
  treating repeated daily observations for the same pair as independent.
- The reproducible study runner writes configuration, forecasts, hidden
  evaluation targets, pooling ablations, fit failures, runtime, and both
  parameter-mean and realized-demand interval coverage to JSON and CSV.

This work does not yet establish statistically demonstrated pooling benefit or
predictive calibration across a full multiseed experiment matrix. These
synthetic runs are evidence scaffolding, not proof of model superiority or
calibration. Approval also requires independent reference implementations,
identifiable synthetic coverage studies, and real-world merchant validation.

The default 10-run synthetic study (`python -m research.phase3a`) used five
independent seeds and 30-/90-day histories. The table reports held-out MAE,
paired cluster-level differences (hierarchical minus sales mean), 90% bootstrap
intervals resampled over the nine store-product clusters in each run, and
realized-demand predictive interval coverage:

| History | Seed | Hierarchical MAE | Sales mean MAE | Paired difference | 90% cluster bootstrap interval | Predictive coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 days | 1 | 2.9933 | 3.1064 | -0.1131 | [-0.3471, 0.1043] | 90.12% |
| 30 days | 2 | 2.6069 | 2.6749 | -0.0680 | [-0.2161, 0.1303] | 96.30% |
| 30 days | 3 | 2.7967 | 2.8407 | -0.0440 | [-0.1106, 0.0234] | 92.59% |
| 30 days | 4 | 2.8333 | 2.9494 | -0.1161 | [-0.3959, 0.1181] | 95.06% |
| 30 days | 5 | 3.0236 | 2.8589 | +0.1647 | [-0.0183, 0.3696] | 96.30% |
| 90 days | 1 | 3.4602 | 3.3868 | +0.0734 | [-0.1003, 0.2900] | 88.89% |
| 90 days | 2 | 3.1029 | 3.1467 | -0.0438 | [-0.1332, 0.0476] | 90.48% |
| 90 days | 3 | 3.1580 | 3.1589 | -0.0009 | [-0.2266, 0.1449] | 91.67% |
| 90 days | 4 | 3.0324 | 3.0541 | -0.0218 | [-0.2254, 0.1736] | 94.44% |
| 90 days | 5 | 2.9715 | 2.9599 | +0.0115 | [-0.0877, 0.1049] | 93.65% |

Mean predictive coverage was 94.07% for 30-day histories and 91.83% for
90-day histories. Mean hierarchical and sales-mean MAE were 2.8507 vs 2.8861
(30 days) and 3.1450 vs 3.1413 (90 days). Every paired interval crosses zero;
the hierarchical model does not demonstrate a reliable accuracy advantage in
this study. Coverage is an empirical synthetic result, not a calibration
guarantee. Nine clusters per run and five data seeds are still too few to
support broad scientific or production claims.

## Remaining limitations

This design does not claim an original method or superiority. Prior-art review,
licensing, merchant consent, independent reference implementations, and
real-world validation are required before making publication or deployment
claims.
