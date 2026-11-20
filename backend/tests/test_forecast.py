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
    assert result["status"] == "insufficient_information"


def test_censored_observation_is_counted():
    result = estimate_demand(
        [{"sales": 10, "available_stock": 10}] * 6
        + [{"sales": 8, "available_stock": 12}]
    )
    assert result["censored_observations"] == 6


def test_all_positive_censored_observations_are_not_identifiable():
    result = estimate_demand([{"sales": 10, "available_stock": 10}] * 7)
    assert result["status"] == "insufficient_information"
    assert result["estimated_daily_demand"] is None


def test_all_zero_known_stock_observations_are_not_confident_demand():
    result = estimate_demand([{"sales": 0, "available_stock": 0}] * 7)
    assert result["status"] == "insufficient_information"


def test_zero_sales_with_stock_is_a_boundary_estimate():
    result = estimate_demand([{"sales": 0, "available_stock": 10}] * 7)
    assert result["status"] == "boundary_estimate"
    assert result["estimated_daily_demand"] is not None


def test_fractional_sales_are_rejected():
    with pytest.raises(ValueError):
        estimate_demand([{"sales": 2.5, "available_stock": 10}])


def test_reorder_is_case_pack_aligned():
    result = recommend_reorder([{"sales": 5, "available_stock": 10}] * 7, lead_time_days=7, on_hand=0, case_pack=12)
    assert result["suggested_units"] % 12 == 0


def test_reorder_rejects_invalid_pack():
    with pytest.raises(ValueError):
        recommend_reorder([{"sales": 5, "available_stock": 10}], case_pack=0)


def test_forecast_intervals_are_non_negative():
    result = estimate_demand([{"sales": 8, "available_stock": 10}] * 10)
    assert 0 <= result["p10"] <= result["p50"] <= result["p90"]


def test_reorder_stops_when_demand_is_not_identifiable():
    result = recommend_reorder([{"sales": 10, "available_stock": 10}] * 7)
    assert result["status"] == "insufficient_information"
    assert result["suggested_units"] is None
