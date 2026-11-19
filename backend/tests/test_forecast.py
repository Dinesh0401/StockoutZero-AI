import pytest

from app.forecast import estimate_demand, recommend_reorder


def test_empty_observations():
    with pytest.raises(ValueError):
        estimate_demand([])


def test_negative_sales_rejected():
    with pytest.raises(ValueError):
        estimate_demand([{"sales": -1, "available_stock": 10}])


def test_sales_exceed_stock_rejected():
    with pytest.raises(ValueError):
        estimate_demand([{"sales": 12, "available_stock": 10}])


def test_unknown_stock_is_distinct_from_zero():
    result = estimate_demand([{"sales": 5, "available_stock": None}])
    assert result["usable_observations"] == 0
    assert "unknown" in result["warning"].lower()


def test_censored_observation_is_counted():
    result = estimate_demand([{"sales": 10, "available_stock": 10}] * 7)
    assert result["censored_observations"] == 7


def test_reorder_is_case_pack_aligned():
    result = recommend_reorder([{"sales": 5, "available_stock": 10}] * 7, lead_time_days=7, on_hand=0, case_pack=12)
    assert result["suggested_units"] % 12 == 0


def test_reorder_rejects_invalid_pack():
    with pytest.raises(ValueError):
        recommend_reorder([{"sales": 5, "available_stock": 10}], case_pack=0)


def test_forecast_intervals_are_non_negative():
    result = estimate_demand([{"sales": 8, "available_stock": 10}] * 10)
    assert 0 <= result["p10"] <= result["p50"] <= result["p90"]
