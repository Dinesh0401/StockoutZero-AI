from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    assert "StockoutZero AI" in response.text


def test_estimate():
    response = client.post("/estimate", json={"observations": [{"sales": 8, "available_stock": 10}] * 7})
    assert response.status_code == 200
    assert "estimated_daily_demand" in response.json()


def test_estimate_alias():
    response = client.post("/forecast", json={"observations": [{"sales": 8, "available_stock": 10}] * 7})
    assert response.status_code == 200


def test_invalid_estimate():
    response = client.post("/estimate", json={"observations": [{"sales": 12, "available_stock": 10}]})
    assert response.status_code == 422


def test_recommend():
    response = client.post("/recommend", json={
        "observations": [{"sales": 8, "available_stock": 10}] * 7,
        "lead_time_days": 7,
        "on_hand": 3,
        "case_pack": 6,
    })
    assert response.status_code == 200
    assert response.json()["suggested_units"] % 6 == 0
