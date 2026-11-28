import pytest
from app.pos_import import import_pos_csv


def test_imports_sales_aliases_and_unknown_availability():
    result = import_pos_csv("day,sold,available\n1,8,10\n2,12,\n")
    assert len(result.observations) == 2
    assert result.observations[0].sales == 8
    assert result.observations[0].available_stock == 10
    assert result.observations[1].available_stock is None
    assert result.rejected_rows == ()


def test_rejects_invalid_rows_but_keeps_valid_rows():
    result = import_pos_csv("sales,available_stock\n8,10\nbad,10\n-1,10\n12,10\n")
    assert len(result.observations) == 1
    assert result.rejected_rows == (3, 4, 5)
    assert result.warning is not None


def test_rejects_missing_sales_header():
    with pytest.raises(ValueError, match="sales or sold"):
        import_pos_csv("day,available_stock\n1,10\n")


def test_rejects_csv_with_no_valid_rows():
    with pytest.raises(ValueError, match="no valid observations"):
        import_pos_csv("sales,available_stock\nbad,10\n")
