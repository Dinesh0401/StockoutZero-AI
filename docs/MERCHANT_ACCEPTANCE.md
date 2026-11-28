# Merchant Acceptance and Calibration Checklist

## Human acceptance

- Merchant can import sales and known availability from an approved POS export.
- Unknown or missing availability remains unknown.
- Recorded sales never exceed availability.
- An estimate is withheld when the history is censored or non-identifiable.
- Reorders are case-pack aligned and require approval.
- Audit recommendations are informational and require approval.
- Merchant can revoke integration access and delete uploaded tenant data.

## Live pilot

Before a pilot, obtain written consent and a bounded test period. Record:

- Store and POS identifiers without cross-tenant mixing.
- Hourly or daily sales and inventory update timestamps.
- Known stockouts and physical audit observations.
- Pilot demand, fulfilled units, stockouts, holding and purchase costs.
- User actions and overridden recommendations.

A pilot has not passed until the merchant-side team confirms the accepted
rejection, abstention, and reorder workflows. Synthetic interval coverage is
not evidence of live calibration.

## Independent review

Independent reviewers must reproduce the study seeds and reference
implementations, audit source provenance and consent, check tenant isolation,
and approve a documented performance threshold before launch. A model with
mixed paired MAE or intervals crossing zero remains unproven.
