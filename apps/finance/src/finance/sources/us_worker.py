"""Child process for one Yahoo batch: symbols and start date in on stdin as JSON, Parquet bytes out on
stdout. Exists so yfinance's leaked open files die with the process."""

# ruff: noqa: E501
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

import io
import json
import logging
import sys
from datetime import date
from typing import Any

import polars as pl

from finance.build.us import RAW_SCHEMA

DOWNLOAD_THREADS = 8


def frame_from_yahoo(df: Any, symbols: list[str]) -> pl.DataFrame:
    """yfinance's wide frame (ticker, field) to long rows. Absent tickers are simply not present."""
    rows: list[pl.DataFrame] = []
    if df is None or df.empty:
        return pl.DataFrame(schema=RAW_SCHEMA)
    present = set(df.columns.get_level_values(0)) if hasattr(df.columns, "levels") else set()
    for sym in symbols:
        if sym not in present:
            continue
        sub = df[sym].dropna(subset=["Close"])
        if sub.empty:
            continue
        rows.append(
            pl.DataFrame(
                {
                    "symbol": [sym] * len(sub),
                    "date": [d.date() for d in sub.index],
                    "open": sub["Open"].astype(float).to_list(),
                    "high": sub["High"].astype(float).to_list(),
                    "low": sub["Low"].astype(float).to_list(),
                    "close": sub["Close"].astype(float).to_list(),
                    "adj_close": sub["Adj Close"].astype(float).to_list(),
                    "volume": sub["Volume"].fillna(0).astype("int64").to_list(),
                    "dividend": sub["Dividends"].fillna(0).astype(float).to_list(),
                    "split": sub["Stock Splits"].fillna(0).astype(float).to_list(),
                },
                schema=RAW_SCHEMA,
            )
        )
    return pl.concat(rows) if rows else pl.DataFrame(schema=RAW_SCHEMA)


def download(symbols: list[str], start: date | None) -> pl.DataFrame:
    if not symbols:
        return pl.DataFrame(schema=RAW_SCHEMA)
    import yfinance as yf

    kw: dict[str, Any] = {"period": "max"} if start is None else {"start": start.isoformat()}
    df = yf.download(
        symbols, auto_adjust=False, actions=True, progress=False,
        threads=DOWNLOAD_THREADS,  # pyright: ignore[reportArgumentType]  (int is accepted)
        group_by="ticker", **kw,
    )  # fmt: skip
    return frame_from_yahoo(df, symbols)


def main() -> int:
    logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    req = json.loads(sys.stdin.read())
    start = date.fromisoformat(req["start"]) if req.get("start") else None
    df = download(req["symbols"], start)
    buf = io.BytesIO()
    df.write_parquet(buf)
    sys.stdout.buffer.write(buf.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
