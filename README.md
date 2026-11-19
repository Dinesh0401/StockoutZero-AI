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

See [product specification](docs/PRODUCT_SPEC.md) and [research protocol](docs/RESEARCH_PROTOCOL.md).
