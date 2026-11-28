# StockoutZero AI

A working research starter for estimating latent customer demand when sales are censored by stock availability.

## Run locally

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the dashboard and `http://127.0.0.1:8000/docs` for the API.

Run the synthetic experiment:

```powershell
cd backend
python -m research.simulate
```

Run the Phase 2B.1 six-baseline benchmark for 30-day and 90-day histories:

```powershell
cd backend
python -m research.benchmark
```

Run the reproducible Phase 3A hierarchical-demand study (five seeds, 30- and
90-day histories by default). It writes a detailed JSON result and a CSV
summary; both output paths can be customized:

```powershell
cd backend
python -m research.phase3a --output results\phase3a.json
```

For a quick single-seed run:

```powershell
python -m research.phase3a --seeds 7 --history-lengths 30 --output results\phase3a-smoke.json
```

Phase 3B currently provides a research-only finite-state Bayesian inventory
filter in `research.uncertain_inventory`. It combines censored sales,
receipts/removals, noisy ledger counts, and physical audits. It is not exposed
through the production API and requires calibrated demand and shrinkage inputs.

Phase 3C provides a research-only cost-aware audit recommendation in
`research.audit_policy`. It compares expected inventory losses with and without
an exact audit and returns `recommend`, `defer`, or
`insufficient_information`. It does not place orders or require an audit.

Quality checks:

```powershell
python -m ruff format --check app research tests
python -m ruff check app research tests
python -m mypy app research
```

See [product specification](docs/PRODUCT_SPEC.md) and [research protocol](docs/RESEARCH_PROTOCOL.md).

Phase 3 research boundaries, validation results, and remaining requirements are
documented in [the technical design](docs/PHASE3_TECHNICAL_DESIGN.md). The
hierarchical model remains research-only and is not exposed as a production
forecast.

The [Phase 2B closeout review](docs/PHASE2B_CLOSEOUT.md) lists completed
work, verification, merge conditions, and remaining scientific limitations.
