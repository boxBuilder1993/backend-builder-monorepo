"""MCX futures bhavcopy rows (the JSON the site returns) to the commodity table."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# polars expressions are typed loosely; the frames are checked by the tests, not the checker.

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import polars as pl

SCHEMA = {
    "date": pl.String, "symbol": pl.String, "instrument": pl.String, "expiry": pl.String,
    "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
    "prev_close": pl.Float64, "volume": pl.Float64, "value_lakh": pl.Float64,
    "open_interest": pl.Float64, "volume_desc": pl.String,
}  # fmt: skip


def mcx_date(rows: list[dict[str, Any]]) -> date:
    return datetime.strptime(rows[0]["Date"], "%m/%d/%Y").date()


def parse_mcx(blob: bytes) -> pl.DataFrame:
    rows: list[dict[str, Any]] = json.loads(blob)
    recs = [
        {
            "date": r["Date"],
            "symbol": (r.get("Symbol") or "").strip(),
            "instrument": r.get("InstrumentName"),
            "expiry": (r.get("ExpiryDate") or "").strip(),
            "open": r.get("Open"),
            "high": r.get("High"),
            "low": r.get("Low"),
            "close": r.get("Close"),
            "prev_close": r.get("PreviousClose"),
            "volume": r.get("Volume"),
            "value_lakh": r.get("Value"),
            "open_interest": r.get("OpenInterest"),
            "volume_desc": (r.get("VolumeInThousands") or "").strip(),
        }
        for r in rows
    ]
    df = pl.DataFrame(recs, schema=SCHEMA)
    return df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%m/%d/%Y"),
        pl.col("expiry").str.to_titlecase().str.strptime(pl.Date, "%d%b%Y", strict=False),
        pl.col("volume").cast(pl.Int64),
        pl.col("open_interest").cast(pl.Int64),
    ).sort(["symbol", "expiry"])
