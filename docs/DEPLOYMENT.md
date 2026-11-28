# Deployment and Operations

## Configuration

- `STOCKOUTZERO_API_KEY`: Optional outside local development. Set a long random
  secret in any shared or hosted environment.
- `STOCKOUTZERO_REQUIRE_TENANT`: Set to `true` so every request carries
  `X-Tenant-ID`.
- `STOCKOUTZERO_CORS_ORIGINS`: Comma-separated browser origins. Never use
  wildcard CORS for an authenticated deployment.

## Release gates

Every release must pass:

1. Unit and API tests.
2. Ruff formatting and lint.
3. mypy.
4. Container build.
5. `/health` and `/ready` smoke checks.
6. API-key and tenant-isolation checks.
7. Synthetic calibration report review.

## Observability

`/metrics` exposes request count, error count, and uptime. Forward these metrics
and application logs to the hosting provider. Add provider-specific uptime,
latency, and error-rate alerts. Do not log API keys, tenant secrets, or raw CSV
content.

## Backup and retention

The MVP is stateless. A production deployment must define a documented database
schema, encrypted backup schedule, tenant deletion workflow, and CSV retention
policy before merchant data is stored.

## Deployment boundary

This project does not place orders or count stock automatically. A production
deployment must keep reorder and audit decisions under human approval.
