import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse, Response

from .forecast import estimate_demand, recommend_reorder
from .pos_import import CSVImportRequest, import_pos_csv

app = FastAPI(title="StockoutZero AI", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("STOCKOUTZERO_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Tenant-ID"],
)

_metrics_lock = Lock()
_request_count = 0
_error_count = 0
_request_started = time.monotonic()


@app.middleware("http")
async def security_and_observability(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    global _request_count, _error_count
    with _metrics_lock:
        _request_count += 1
    started = time.monotonic()
    response = await call_next(request)
    elapsed = time.monotonic() - started
    if response.status_code >= 400:
        with _metrics_lock:
            _error_count += 1
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Server-Timing"] = f"app;dur={elapsed * 1000:.2f}"
    return response


class Observation(BaseModel):
    sales: int = Field(ge=0)
    available_stock: int | None = Field(default=None, ge=0)


class EstimateRequest(BaseModel):
    observations: list[Observation]


class ReorderRequest(EstimateRequest):
    lead_time_days: int = Field(default=7, ge=1)
    on_hand: int = Field(default=0, ge=0)
    case_pack: int = Field(default=1, ge=1)


def _authorize(
    x_api_key: str | None,
    x_tenant_id: str | None,
) -> str:
    required_api_key = os.getenv("STOCKOUTZERO_API_KEY")
    if required_api_key and x_api_key != required_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    require_tenant = os.getenv("STOCKOUTZERO_REQUIRE_TENANT", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    if require_tenant and (not x_tenant_id or not x_tenant_id.strip()):
        raise HTTPException(status_code=400, detail="X-Tenant-ID is required")
    tenant_id = (x_tenant_id or "default").strip() or "default"
    if len(tenant_id) > 128:
        raise HTTPException(status_code=400, detail="X-Tenant-ID is too long")
    return tenant_id


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness() -> dict[str, object]:
    return {
        "status": "ready",
        "uptime_seconds": round(time.monotonic() - _request_started, 3),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    with _metrics_lock:
        request_count = _request_count
        error_count = _error_count
    uptime = time.monotonic() - _request_started
    body = (
        "# HELP stockoutzero_requests_total HTTP requests received.\n"
        "# TYPE stockoutzero_requests_total counter\n"
        f"stockoutzero_requests_total {request_count}\n"
        "# HELP stockoutzero_errors_total HTTP error responses.\n"
        "# TYPE stockoutzero_errors_total counter\n"
        f"stockoutzero_errors_total {error_count}\n"
        "# HELP stockoutzero_uptime_seconds Service uptime.\n"
        "# TYPE stockoutzero_uptime_seconds gauge\n"
        f"stockoutzero_uptime_seconds {uptime:.3f}\n"
    )
    return PlainTextResponse(body, media_type="text/plain")


@app.post("/estimate")
def estimate(
    request: EstimateRequest,
    x_api_key: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
) -> dict[str, object]:
    tenant_id = _authorize(x_api_key, x_tenant_id)
    try:
        result = estimate_demand([row.model_dump() for row in request.observations])
        return {**result, "tenant_id": tenant_id}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/forecast")
def forecast(
    request: EstimateRequest,
    x_api_key: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
) -> dict[str, object]:
    return estimate(request, x_api_key=x_api_key, x_tenant_id=x_tenant_id)


@app.post("/recommend")
def recommend(
    request: ReorderRequest,
    x_api_key: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
) -> dict[str, object]:
    tenant_id = _authorize(x_api_key, x_tenant_id)
    try:
        result = recommend_reorder(
            [row.model_dump() for row in request.observations],
            lead_time_days=request.lead_time_days,
            on_hand=request.on_hand,
            case_pack=request.case_pack,
        )
        return {**result, "tenant_id": tenant_id}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/pos/import")
def pos_import(
    request: CSVImportRequest,
    x_api_key: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
) -> dict[str, object]:
    tenant_id = _authorize(x_api_key, x_tenant_id)
    try:
        imported = import_pos_csv(request.csv_text)
        return {
            "tenant_id": tenant_id,
            "status": "validated",
            "observations": [row.model_dump() for row in imported.observations],
            "row_count": len(imported.observations),
            "rejected_rows": imported.rejected_rows,
            "warning": imported.warning,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    html_path = Path(__file__).with_name("static") / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
