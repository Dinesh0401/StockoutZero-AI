from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .forecast import estimate_demand, recommend_reorder

app = FastAPI(title="StockoutZero AI", version="0.1.0")


class Observation(BaseModel):
    sales: int = Field(ge=0)
    available_stock: int | None = Field(default=None, ge=0)


class EstimateRequest(BaseModel):
    observations: list[Observation]


class ReorderRequest(EstimateRequest):
    lead_time_days: int = Field(default=7, ge=1)
    on_hand: int = Field(default=0, ge=0)
    case_pack: int = Field(default=1, ge=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/estimate")
def estimate(request: EstimateRequest) -> dict[str, object]:
    try:
        return estimate_demand([row.model_dump() for row in request.observations])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/forecast")
def forecast(request: EstimateRequest) -> dict[str, object]:
    return estimate(request)


@app.post("/recommend")
def recommend(request: ReorderRequest) -> dict[str, object]:
    try:
        return recommend_reorder(
            [row.model_dump() for row in request.observations],
            lead_time_days=request.lead_time_days,
            on_hand=request.on_hand,
            case_pack=request.case_pack,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    html_path = Path(__file__).with_name("static") / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
