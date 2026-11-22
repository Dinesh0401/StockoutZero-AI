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

Quality checks:

```powershell
python -m ruff check app research tests
python -m mypy app research
```

See [product specification](docs/PRODUCT_SPEC.md) and [research protocol](docs/RESEARCH_PROTOCOL.md).

Phase 3 implementation is defined in [the technical design](docs/PHASE3_TECHNICAL_DESIGN.md). It remains a review document until Phase 2 is merged.
