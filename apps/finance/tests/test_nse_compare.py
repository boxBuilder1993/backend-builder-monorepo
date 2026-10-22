"""Acceptance: the new parsers rebuild the old poller's Parquet byte for byte from the raw files it
kept. Runs only where that store exists (FINANCE_COMPARE_DIR, default the poller clone)."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from finance.build import nse as N
from finance.files import parquet_rel, raw_rel
from finance.sources import nse as S

STORE = Path(
    os.environ.get("FINANCE_COMPARE_DIR", "~/Workspaces/indian-finance-data/data")
).expanduser()

DAYS = [
    date(2021, 9, 1),   # first backfill day, old cm layout
    date(2022, 8, 8),   # NSE served Excel at the .csv URL
    date(2024, 1, 2),   # first UDiFF day
    date(2024, 7, 5),   # last old-layout day
    date(2024, 7, 8),   # first UDiFF-only day
    date(2026, 10, 21), # a recent day
]  # fmt: skip


def _need(rel: str) -> Path:
    p = STORE / rel
    if not p.exists():
        pytest.skip(f"{rel} not in {STORE}")
    return p


@pytest.mark.parametrize("table", ["equity", "index", "commodity"])
@pytest.mark.parametrize("day", DAYS)
def test_rebuild_matches_old_parquet(table: str, day: date) -> None:
    old = _need(parquet_rel(table, day))
    if table == "equity":
        _need(raw_rel("nse_cm", day, "zip"))
    elif table == "index":
        _need(raw_rel("nse_index", day, "csv.gz"))
    else:
        _need(raw_rel("mcx", day, "json.gz"))
    df = S.build_day(STORE, table, day)
    expected = pl.read_parquet(old)
    assert df.schema == expected.schema
    assert df.equals(expected), f"{table} {day}: frames differ"
    assert N.parquet_bytes(df) == old.read_bytes(), f"{table} {day}: same frame, different bytes"
