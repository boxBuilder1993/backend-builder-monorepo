"""Raw NSE files to one schema per table. Every parser takes bytes and returns a polars frame.

The frames are sorted on a total order and written with zstd, so a rebuild of the same raw day is
byte-identical to the file the old poller wrote, which is how each source is accepted.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# polars expressions are typed loosely; the frames are checked by the tests, not the checker.

from __future__ import annotations

import io
import zipfile
from datetime import date

import polars as pl

from finance.build.xlsx import to_csv_if_xlsx


def as_float(col: str, alias: str) -> pl.Expr:
    return pl.col(col).str.strip_chars().cast(pl.Float64, strict=False).alias(alias)


def as_int(col: str, alias: str) -> pl.Expr:
    return pl.col(col).str.strip_chars().cast(pl.Float64, strict=False).cast(pl.Int64).alias(alias)


def as_str(col: str, alias: str) -> pl.Expr:
    return pl.col(col).str.strip_chars().alias(alias)


def as_date(col: str, alias: str, fmt: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.strip_chars()
        .str.to_titlecase()
        .str.strptime(pl.Date, fmt, strict=False)
        .alias(alias)
    )


def csv_frame(data: bytes) -> pl.DataFrame:
    """Every column as a string, header names stripped, unnamed trailing columns dropped."""
    df = pl.read_csv(io.BytesIO(data), infer_schema_length=0, truncate_ragged_lines=True)
    df = df.rename({c: c.strip() for c in df.columns})
    return df.drop([c for c in df.columns if c == "" or c.startswith("column_")])


def first_row_value(text: str, column: str) -> str:
    """The first data row's value in a CSV column; what the date checks read."""
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("file has no data rows")
    header = [c.strip() for c in lines[0].split(",")]
    try:
        i = header.index(column)
    except ValueError:
        raise ValueError(f"column {column!r} not in header {header[:6]}") from None
    return lines[1].split(",")[i].strip()


# ---- equities ---------------------------------------------------------------------------------

EQUITY_COLS = [
    "date", "isin", "symbol", "series", "instrument", "open", "high", "low", "close", "last",
    "prev_close", "volume", "turnover", "trades", "avg_price", "deliv_qty", "deliv_pct", "fmt",
]  # fmt: skip


def cm_text(blob: bytes) -> str:
    z = zipfile.ZipFile(io.BytesIO(blob))
    return z.read(z.namelist()[0]).decode("utf-8", "replace")


def cm_date(blob: bytes) -> date:
    """The trading date inside a cm bhavcopy zip, either layout."""
    text = cm_text(blob)
    if "TradDt" in text.splitlines()[0]:
        return date.fromisoformat(first_row_value(text, "TradDt"))
    return _parse_dmy(first_row_value(text, "TIMESTAMP"))


def _parse_dmy(s: str) -> date:
    from datetime import datetime

    return datetime.strptime(s.title(), "%d-%b-%Y").date()


def parse_cm(blob: bytes) -> pl.DataFrame:
    """cm bhavcopy zip, either layout: date isin symbol series instrument ohlc last prev_close
    volume turnover (rupees) trades fmt."""
    df = csv_frame(cm_text(blob).encode())
    if "TradDt" in df.columns:
        return df.select(
            as_date("TradDt", "date", "%Y-%m-%d"),
            as_str("ISIN", "isin"),
            as_str("TckrSymb", "symbol"),
            as_str("SctySrs", "series"),
            as_str("FinInstrmTp", "instrument"),
            as_float("OpnPric", "open"),
            as_float("HghPric", "high"),
            as_float("LwPric", "low"),
            as_float("ClsPric", "close"),
            as_float("LastPric", "last"),
            as_float("PrvsClsgPric", "prev_close"),
            as_int("TtlTradgVol", "volume"),
            as_float("TtlTrfVal", "turnover"),
            as_int("TtlNbOfTxsExctd", "trades"),
            pl.lit("udiff").alias("fmt"),
        )
    return df.select(
        as_date("TIMESTAMP", "date", "%d-%b-%Y"),
        as_str("ISIN", "isin"),
        as_str("SYMBOL", "symbol"),
        as_str("SERIES", "series"),
        pl.lit(None, dtype=pl.String).alias("instrument"),
        as_float("OPEN", "open"),
        as_float("HIGH", "high"),
        as_float("LOW", "low"),
        as_float("CLOSE", "close"),
        as_float("LAST", "last"),
        as_float("PREVCLOSE", "prev_close"),
        as_int("TOTTRDQTY", "volume"),
        as_float("TOTTRDVAL", "turnover"),
        as_int("TOTALTRADES", "trades"),
        pl.lit("old").alias("fmt"),
    )


def full_date(blob: bytes) -> date:
    """The date inside sec_bhavdata_full; NSE serves the previous session's file on non-trading
    days."""
    text = to_csv_if_xlsx(blob).decode("utf-8", "replace")
    return _parse_dmy(first_row_value(text, "DATE1"))


def parse_full(blob: bytes) -> pl.DataFrame:
    """sec_bhavdata_full: date symbol series avg_price deliv_qty deliv_pct ('-' becomes null)."""
    df = csv_frame(to_csv_if_xlsx(blob))
    return df.select(
        as_date("DATE1", "date", "%d-%b-%Y"),
        as_str("SYMBOL", "symbol"),
        as_str("SERIES", "series"),
        as_float("AVG_PRICE", "avg_price"),
        as_int("DELIV_QTY", "deliv_qty"),
        as_float("DELIV_PER", "deliv_pct"),
    )


def equity_day(cm: pl.DataFrame, full: pl.DataFrame | None) -> pl.DataFrame:
    """The equity table for one day: every cm row, delivery fields joined on (symbol, series)."""
    if full is None:
        out = cm.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("avg_price"),
            pl.lit(None, dtype=pl.Int64).alias("deliv_qty"),
            pl.lit(None, dtype=pl.Float64).alias("deliv_pct"),
        )
    else:
        out = cm.join(
            full.drop("date").unique(subset=["symbol", "series"], keep="first"),
            on=["symbol", "series"],
            how="left",
        )
    return out.select(EQUITY_COLS).sort(["symbol", "series"])


# ---- indices ----------------------------------------------------------------------------------


def index_date(blob: bytes) -> date:
    from datetime import datetime

    text = blob.decode("utf-8", "replace")
    return datetime.strptime(first_row_value(text, "Index Date"), "%d-%m-%Y").date()


def parse_index(blob: bytes) -> pl.DataFrame:
    df = csv_frame(blob)
    return df.select(
        as_date("Index Date", "date", "%d-%m-%Y"),
        as_str("Index Name", "index_name"),
        as_float("Open Index Value", "open"),
        as_float("High Index Value", "high"),
        as_float("Low Index Value", "low"),
        as_float("Closing Index Value", "close"),
        as_float("Points Change", "points_change"),
        as_float("Change(%)", "change_pct"),
        as_int("Volume", "volume"),
        as_float("Turnover (Rs. Cr.)", "turnover_cr"),
        as_float("P/E", "pe"),
        as_float("P/B", "pb"),
        as_float("Div Yield", "div_yield"),
    ).sort("index_name")


# ---- writing ----------------------------------------------------------------------------------


def parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd")
    return buf.getvalue()


def check_day(df: pl.DataFrame, day: date, what: str) -> None:
    """Every row of a daily table carries that day; anything else is a parser or source fault."""
    bad = df.filter(pl.col("date") != day).height
    if bad:
        raise ValueError(f"{what} {day}: {bad} rows carry a different date")
