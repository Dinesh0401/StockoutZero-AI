# StockoutZero AI Product Specification

## Scope
The MVP serves independent grocery and household-goods stores. It estimates latent daily demand from sales and known inventory availability, reports uncertainty, and proposes a case-pack-aligned reorder after human approval.

## Core rules
- `sales < available_stock`: sales are treated as observed demand.
- `sales == available_stock > 0`: observation is censored at available stock.
- `available_stock == 0`: the zero observation is not enough to identify demand.
- `available_stock == null`: availability is unknown and the observation is excluded from censored likelihood fitting.
- Recorded sales may never exceed known available stock.

## MVP API
- `GET /` dashboard
- `GET /health` service status
- `POST /estimate` estimate demand
- `POST /recommend` estimate demand and suggest a lead-time replenishment
