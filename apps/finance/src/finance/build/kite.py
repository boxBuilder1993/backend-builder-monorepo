"""Raw Kite candle files to one Parquet per trading day: equity_1m. Every candle must carry the
file's date; a candle from another day is a source fault and fails the day."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import gzip
import json
from datetime import date, datetime

import polars as pl

SCHEMA = {
    "ts": pl.Datetime("us"), "symbol": pl.Utf8, "token": pl.Int64, "open": pl.Float64,
    "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
}  # fmt: skip


def parse(raw: bytes, day: date) -> pl.DataFrame:
    rows: list[tuple[object, ...]] = []
    lines = gzip.decompress(raw).splitlines() if raw[:2] == b"\x1f\x8b" else raw.splitlines()
    for line in lines:
        rec = json.loads(line)
        for ts, o, h, lo, c, v in rec["candles"]:
            t = datetime.fromisoformat(ts).replace(tzinfo=None)
            if t.date() != day:
                raise ValueError(f"{rec['symbol']}: candle {t} in file for {day}")
            rows.append((t, rec["symbol"], rec["token"], o, h, lo, c, v))
    return pl.DataFrame(rows, schema=SCHEMA, orient="row").sort(["symbol", "ts"])
