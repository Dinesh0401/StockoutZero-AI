from app.main import app
from fastapi.testclient import TestClient

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
    response = client.post(
        "/estimate", json={"observations": [{"sales": 8, "available_stock": 10}] * 7}
    )
    assert response.status_code == 200
    assert "estimated_daily_demand" in response.json()


def test_estimate_alias():
    response = client.post(
        "/forecast", json={"observations": [{"sales": 8, "available_stock": 10}] * 7}
    )
    assert response.status_code == 200


def test_invalid_estimate():
    response = client.post(
        "/estimate", json={"observations": [{"sales": 12, "available_stock": 10}]}
    )
    assert response.status_code == 422


def test_recommend():
    response = client.post(
        "/recommend",
        json={
            "observations": [{"sales": 8, "available_stock": 10}] * 7,
            "lead_time_days": 7,
            "on_hand": 3,
            "case_pack": 6,
        },
    )
    assert response.status_code == 200
    assert response.json()["suggested_units"] % 6 == 0


def test_ready_and_metrics_endpoints():
    assert client.get("/ready").status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "stockoutzero_requests_total" in metrics.text


def test_tenant_header_is_preserved():
    response = client.post(
        "/estimate",
        headers={"X-Tenant-ID": "merchant-123"},
        json={"observations": [{"sales": 8, "available_stock": 10}] * 7},
    )
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "merchant-123"


def test_api_key_is_required_when_configured(monkeypatch):
    monkeypatch.setenv("STOCKOUTZERO_API_KEY", "test-secret")
    missing = client.post(
        "/estimate",
        json={"observations": [{"sales": 8, "available_stock": 10}] * 7},
    )
    accepted = client.post(
        "/estimate",
        headers={"X-API-Key": "test-secret", "X-Tenant-ID": "merchant-123"},
        json={"observations": [{"sales": 8, "available_stock": 10}] * 7},
    )
    assert missing.status_code == 401
    assert accepted.status_code == 200


def test_tenant_is_required_when_configured(monkeypatch):
    monkeypatch.setenv("STOCKOUTZERO_REQUIRE_TENANT", "true")
    response = client.post(
        "/estimate",
        json={"observations": [{"sales": 8, "available_stock": 10}] * 7},
    )
    assert response.status_code == 400


def test_pos_csv_import():
    response = client.post(
        "/pos/import",
        headers={"X-Tenant-ID": "merchant-123"},
        json={"csv_text": ("day,sales,available_stock\n1,8,10\n2,12,\nbad,3,5\n")},
    )
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "merchant-123"
    assert response.json()["row_count"] == 2
    assert response.json()["rejected_rows"] == [4]
