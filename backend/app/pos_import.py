"""Strict merchant POS CSV import contract."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from pydantic import BaseModel, Field


class CSVObservation(BaseModel):
    sales: int = Field(ge=0)
    available_stock: int | None = Field(default=None, ge=0)
    day: int | None = Field(default=None, ge=0)


class CSVImportRequest(BaseModel):
    csv_text: str = Field(min_length=1, max_length=5_000_000)


@dataclass(frozen=True)
class POSImportResult:
    observations: tuple[CSVObservation, ...]
    rejected_rows: tuple[int, ...]
    warning: str | None


def import_pos_csv(csv_text: str) -> POSImportResult:
    """Parse a POS export with sales and optional availability columns.

    Accepted headers:
    - sales (required), available_stock (optional)
    - sold (alias), available (alias), day (optional)
    Unknown or empty availability is imported as None and never converted to
    an observed zero.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise ValueError("CSV headers are required")
    normalized_headers = {
        header.strip().lower(): header for header in reader.fieldnames
    }
    sales_header = normalized_headers.get("sales") or normalized_headers.get("sold")
    if sales_header is None:
        raise ValueError("CSV must contain a sales or sold column")

    availability_header = normalized_headers.get(
        "available_stock"
    ) or normalized_headers.get("available")
    day_header = normalized_headers.get("day")
    observations: list[CSVObservation] = []
    rejected_rows: list[int] = []

    for row_number, row in enumerate(reader, start=2):
        try:
            sales_text = (row.get(sales_header) or "").strip()
            if not sales_text:
                raise ValueError
            sales = int(sales_text)
            if str(sales) != sales_text and sales_text not in {
                f"+{sales}",
                f"0{sales}",
            }:
                raise ValueError

            available: int | None = None
            if availability_header:
                available_text = (row.get(availability_header) or "").strip()
                if available_text:
                    available = int(available_text)

            day: int | None = None
            if day_header:
                day_text = (row.get(day_header) or "").strip()
                if day_text:
                    day = int(day_text)

            observation = CSVObservation(
                sales=sales,
                available_stock=available,
                day=day,
            )
            if sales > (available if available is not None else sales):
                raise ValueError
            observations.append(observation)
        except (TypeError, ValueError):
            rejected_rows.append(row_number)

    if not observations:
        raise ValueError("CSV contained no valid observations")
    warning = (
        f"{len(rejected_rows)} invalid CSV row(s) were rejected."
        if rejected_rows
        else None
    )
    return POSImportResult(tuple(observations), tuple(rejected_rows), warning)
