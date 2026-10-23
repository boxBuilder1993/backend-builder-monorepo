"""US tables: the Nasdaq Trader listing snapshots to us_listings (one per day) and us_symbols, the
Yahoo raw batches to us_equity (one Parquet per session). A nightly batch carries a few days per
symbol, so the affected session files are merged (new rows win) and rewritten; a manifest under the
table records which raw batches have been folded in, so re-runs are free."""

# ruff: noqa: E501
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

import polars as pl

from finance.build.nse import parquet_bytes
from finance.files import read_raw, write_if_changed
from finance.settings import LISTING_FILES, US_BENCHMARKS

RAW_SCHEMA = {
    "symbol": pl.String, "date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
    "close": pl.Float64, "adj_close": pl.Float64, "volume": pl.Int64, "dividend": pl.Float64,
    "split": pl.Float64,
}  # fmt: skip
EQ_COLS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividend",
    "split",
]
LISTINGS_SCHEMA = {
    "date": pl.Date, "symbol": pl.String, "yahoo": pl.String, "name": pl.String, "exchange": pl.String,
    "etf": pl.Boolean, "test": pl.Boolean, "kind": pl.String,
}  # fmt: skip
CHUNK = 15


def listing_rel(day: date, f: str) -> str:
    return f"raw/us_listings/{day:%Y}/us_listings_{day:%Y%m%d}_{f.split('.')[0]}.txt.gz"


def listing_days(root: Path) -> list[date]:
    return sorted(
        {
            datetime.strptime(p.name.split("_")[2], "%Y%m%d").date()
            for p in (root / "raw" / "us_listings").glob("*/us_listings_*_nasdaqlisted.txt.gz")
        }
    )


def parse_listings(root: Path, day: date) -> pl.DataFrame:
    """Both files for one day to: symbol (Nasdaq form), yahoo (Yahoo form), name, exchange, etf,
    test, kind."""
    frames: list[pl.DataFrame] = []
    for f in LISTING_FILES:
        p = root / listing_rel(day, f)
        if not p.exists():
            continue
        lines = read_raw(p).decode("utf-8", "replace").splitlines()
        lines = [ln for ln in lines if ln and not ln.startswith("File Creation Time")]
        df = pl.read_csv(
            io.BytesIO("\n".join(lines).encode()), separator="|", infer_schema_length=0
        )
        df = df.rename({c: c.strip() for c in df.columns})
        if "ACT Symbol" in df.columns:  # otherlisted: NYSE, NYSE American, Arca, BATS, IEX
            ex = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "BATS", "V": "IEX"}
            df = df.select(
                pl.col("NASDAQ Symbol").alias("symbol"),
                pl.col("Security Name").alias("name"),
                pl.col("Exchange").replace(ex).alias("exchange"),
                pl.col("ETF").alias("etf"),
                pl.col("Test Issue").alias("test"),
            )
        else:
            df = df.select(
                pl.col("Symbol").alias("symbol"),
                pl.col("Security Name").alias("name"),
                pl.lit("NASDAQ").alias("exchange"),
                pl.col("ETF").alias("etf"),
                pl.col("Test Issue").alias("test"),
            )
        frames.append(df)
    if not frames:
        return pl.DataFrame(schema=LISTINGS_SCHEMA)
    df = (
        pl.concat(frames)
        .with_columns(
            pl.lit(day).alias("date"),
            (pl.col("etf") == "Y").alias("etf"),
            (pl.col("test") == "Y").alias("test"),
            pl.col("symbol")
            .str.replace_all(r"\.", "-")
            .str.replace_all(r"\$", "-P")
            .alias("yahoo"),
            pl.when(
                pl.col("symbol").str.contains(r"\$")
                | pl.col("name").str.contains(r"(?i)preferred|preference")
            )
            .then(pl.lit("preferred"))
            .when(
                pl.col("symbol").str.contains(r"\+") | pl.col("name").str.contains(r"(?i)\bwarrant")
            )
            .then(pl.lit("warrant"))
            .when(
                pl.col("symbol").str.contains("=") | pl.col("name").str.contains(r"(?i)\bunits?\b")
            )
            .then(pl.lit("unit"))
            .when(
                pl.col("symbol").str.contains(r"\^")
                | pl.col("name").str.contains(r"(?i)\brights?\b")
            )
            .then(pl.lit("right"))
            .when(pl.col("name").str.contains(r"(?i)\bnotes?\b|debenture"))
            .then(pl.lit("debt"))
            .when(pl.col("etf") == "Y")
            .then(pl.lit("etf"))
            .otherwise(pl.lit("stock"))
            .alias("kind"),
        )
        .select("date", "symbol", "yahoo", "name", "exchange", "etf", "test", "kind")
    )
    return df.unique(subset=["symbol"], keep="first").sort("symbol")


def price_universe(root: Path, day: date) -> pl.DataFrame:
    """Common stock (ADRs and SPACs included), not test issues, plus the benchmarks."""
    df = parse_listings(root, day)
    keep = df.filter(
        (~pl.col("test")) & ((pl.col("kind") == "stock") | pl.col("symbol").is_in(US_BENCHMARKS))
    )
    return keep.select("symbol", "yahoo", "name", "exchange", "kind")


def us_equity_rel(day: date) -> str:
    return f"parquet/us_equity/year={day:%Y}/us_equity_{day:%Y%m%d}.parquet"


def manifest_path(root: Path) -> Path:
    return root / "parquet" / "us_equity" / "_ingested.txt"


def raw_batches(root: Path) -> list[Path]:
    return sorted((root / "raw" / "yahoo").glob("*/yahoo_*_b*.parquet"))


def build_us_equity(root: Path) -> list[date]:
    """Fold every raw batch not yet in the manifest into the session files. Returns the sessions
    whose file was written."""
    manifest = manifest_path(root)
    done = set(manifest.read_text().split()) if manifest.exists() else set()
    new = [p for p in raw_batches(root) if p.name not in done]
    written: list[date] = []
    for i in range(0, len(new), CHUNK):
        chunk = new[i : i + CHUNK]
        fresh = pl.concat([pl.read_parquet(p, columns=list(RAW_SCHEMA)) for p in chunk]).select(
            EQ_COLS
        )
        fresh = (
            fresh.with_row_index("_i")
            .sort("_i")
            .unique(subset=["symbol", "date"], keep="last")
            .drop("_i")
        )
        for (d,), part in fresh.partition_by("date", as_dict=True).items():
            assert isinstance(d, date)
            p = root / us_equity_rel(d)
            if p.exists():
                part = pl.concat([pl.read_parquet(p), part]).unique(subset=["symbol"], keep="last")
            if write_if_changed(p, parquet_bytes(part.sort("symbol"))):
                written.append(d)
        done |= {p.name for p in chunk}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(sorted(done)) + "\n")
    return sorted(set(written))


def us_listings_rel(day: date) -> str:
    return f"parquet/us_listings/year={day:%Y}/us_listings_{day:%Y%m%d}.parquet"


def build_us_listings(root: Path) -> tuple[list[date], pl.DataFrame | None]:
    """One snapshot per listing day (never rebuilt), and the per-symbol summary us_symbols."""
    days = listing_days(root)
    written: list[date] = []
    for d in days:
        p = root / us_listings_rel(d)
        if not p.exists() and write_if_changed(p, parquet_bytes(parse_listings(root, d))):
            written.append(d)
    files = sorted((root / "parquet" / "us_listings").glob("year=*/*.parquet"))
    if not files:
        return written, None
    allsnap = pl.concat([pl.read_parquet(p) for p in files])
    symbols = (
        allsnap.group_by("symbol")
        .agg(
            pl.col("yahoo").last(),
            pl.col("name").last(),
            pl.col("exchange").last(),
            pl.col("kind").last(),
            pl.col("etf").last(),
            pl.col("test").last(),
            pl.col("date").min().alias("first_seen"),
            pl.col("date").max().alias("last_seen"),
            pl.len().alias("n_snapshots"),
        )
        .sort(["symbol", "first_seen"])
    )
    return written, symbols
