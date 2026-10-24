"""Acceptance for the reference, us and kite legs: rebuild from the old poller's raw files and
compare with its Parquet byte for byte. Runs only where that store exists."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from finance.build import kite as BK
from finance.build import reference as R
from finance.build import us as BU
from finance.build.nse import parquet_bytes
from finance.files import parquet_rel, raw_rel, reference_rel

STORE = Path(
    os.environ.get("FINANCE_COMPARE_DIR", "~/Workspaces/indian-finance-data/data")
).expanduser()


def _need(rel: str) -> Path:
    p = STORE / rel
    if not p.exists():
        pytest.skip(f"{rel} not in {STORE}")
    return p


def _same(df: pl.DataFrame, rel: str) -> None:
    old = _need(rel)
    expected = pl.read_parquet(old)
    assert df.schema == expected.schema
    assert df.equals(expected), f"{rel}: frames differ"
    assert parquet_bytes(df) == old.read_bytes(), f"{rel}: same frame, different bytes"


def test_corp_actions_holidays_symbol_changes_equity_master() -> None:
    meta = _need("raw/nse_meta")
    _same(
        R.parse_corp_actions(sorted(meta.glob("corp_actions_*.json.gz"))),
        reference_rel("corp_actions"),
    )
    _same(R.parse_holidays(sorted(meta.glob("holidays_*.json.gz"))), reference_rel("holidays"))
    _same(R.parse_symbol_changes(meta / "symbolchange.csv.gz"), reference_rel("symbol_changes"))
    _same(R.parse_equity_master(meta / "equity_l.csv.gz"), reference_rel("equity_master"))


def test_securities() -> None:
    """The legacy file was built from the equity days of its night; the store may hold more days
    now. Same days: byte for byte. More days: the chains (isin -> sid, first_date) must agree on
    every ISIN the legacy file knew, and every new ISIN must be from a day it had not seen."""
    old_path = _need(reference_rel("securities"))
    sec = R.build_securities(STORE)
    assert sec is not None
    old = pl.read_parquet(old_path)
    if sec["last_date"].max() == old["last_date"].max():
        _same(sec, reference_rel("securities"))
        return
    known = set(old["isin"].to_list())
    common = sec.filter(pl.col("isin").is_in(list(known))).sort(["sid", "first_date", "isin"])
    assert common.select("isin", "sid", "first_date").equals(
        old.sort(["sid", "first_date", "isin"]).select("isin", "sid", "first_date")
    )
    extra = sec.filter(~pl.col("isin").is_in(list(known)))
    assert (extra["first_date"] > old["last_date"].max()).all()


def _days(pattern: str, pos: slice) -> list[date]:
    files = sorted(STORE.glob(pattern))
    if not files:
        pytest.skip(f"no {pattern} in {STORE}")
    picked = [files[0], files[len(files) // 2], files[-1]]
    return [
        date(int(p.name[pos][:4]), int(p.name[pos][4:6]), int(p.name[pos][6:8])) for p in picked
    ]


def test_us_listings_latest_day() -> None:
    """Only the newest snapshot: listing files are built once and never rebuilt, so older ones carry
    the classification rules of the night they were built."""
    day = _days("raw/us_listings/*/us_listings_*_nasdaqlisted.txt.gz", slice(12, 20))[-1]
    _same(BU.parse_listings(STORE, day), BU.us_listings_rel(day))


def test_equity_1m_days() -> None:
    """A day's Parquet is its raw file plus every supplement (instruments backfilled later), deduped."""
    for day in sorted(set(_days("raw/kite_1m/*/kite_1m_????????*.jsonl.gz", slice(8, 16)))):
        files = sorted((STORE / "raw/kite_1m").glob(f"*/kite_1m_{day:%Y%m%d}*.jsonl.gz"))
        frames = [BK.parse(f.read_bytes(), day) for f in files]
        df = pl.concat(frames).unique(subset=["symbol", "ts"], keep="last").sort(["symbol", "ts"])
        _same(df, parquet_rel("equity_1m", day))
