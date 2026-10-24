"""The us leg: the whole listed US market from Yahoo Finance, universe from Nasdaq Trader.

Yahoo serves one range per symbol, not one file per day, so the unit of work is a symbol. A symbol is
a gap when the store has no rows for it or its last session is behind the market's latest. Each night
every gapped symbol is asked for its missing range; a new listing gets full history. Raw answers are
kept per batch per night under raw/yahoo/. Symbols that answer empty several nights running are
dormant until they reappear in the listings. Each batch runs in a short-lived child process because
yfinance leaks open files. The US close is 01:30 IST, so the evening run picks up the prior session.
"""

# ruff: noqa: E501
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

import functools
import gzip
import io
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import requests

from finance.build import us as U
from finance.files import reference_rel, sha256, write_if_changed
from finance.http import HEADERS, Transient, retry
from finance.settings import (
    LISTING_FILES,
    NASDAQ_SCREENER_URL,
    NASDAQ_TRADER,
    US_BATCH,
    US_DORMANT_AFTER,
    US_INDICES,
    US_MIN_HIT_RATE,
    US_PAUSE,
)
from finance.sources import Ledger, LegResult, Written

log = logging.getLogger("finance.us")

DORMANT_REL = "raw/yahoo/dormant.json.gz"
Batch = Callable[[list[str], date | None], pl.DataFrame]


def _written(root: Path, rel: str, kind: str, source: str, day: date | None) -> Written:
    p = root / rel
    return Written(rel, kind, source, day, p.stat().st_size, sha256(p))


def fetch_listings(root: Path, today: date) -> list[Written]:
    """Nasdaq Trader publishes today's list only; saved under today's date, both files."""
    s = requests.Session()
    s.headers.update(HEADERS)
    out: list[Written] = []
    for f in LISTING_FILES:

        def _get(u: str = NASDAQ_TRADER.format(f=f)) -> bytes:
            r = s.get(u, timeout=60)
            if r.status_code != 200 or not r.text.startswith(("Symbol|", "ACT Symbol|")):
                raise Transient(f"HTTP {r.status_code}")
            return r.content

        content = retry(_get, what=f)
        rel = U.listing_rel(today, f)
        write_if_changed(root / rel, gzip.compress(content, mtime=0))
        out.append(_written(root, rel, "raw", "nasdaq", today))
    return out


def fetch_batch(symbols: list[str], start: date | None) -> pl.DataFrame:
    """One Yahoo batch in a fresh child (finance.sources.us_worker)."""
    req = json.dumps({"symbols": symbols, "start": start.isoformat() if start else None}).encode()
    proc = subprocess.run(
        [sys.executable, "-m", "finance.sources.us_worker"],
        input=req,
        capture_output=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise Transient(
            f"batch worker exit {proc.returncode}: {proc.stderr.decode(errors='replace')[-300:]}"
        )
    return pl.read_parquet(io.BytesIO(proc.stdout))


def load_dormant(root: Path) -> dict[str, dict[str, Any]]:
    p = root / DORMANT_REL
    return json.loads(gzip.decompress(p.read_bytes())) if p.exists() else {}


def save_dormant(root: Path, d: dict[str, dict[str, Any]]) -> None:
    write_if_changed(
        root / DORMANT_REL, gzip.compress(json.dumps(d, sort_keys=True, indent=0).encode(), mtime=0)
    )


def last_dates(root: Path) -> dict[str, date]:
    """Last stored session per symbol, from the us_equity Parquet: the only state that matters."""
    files = list((root / "parquet" / "us_equity").glob("year=*/*.parquet"))
    if not files:
        return {}
    glob = (root / "parquet" / "us_equity" / "*" / "*.parquet").as_posix()
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"select symbol, max(date) from read_parquet('{glob}', hive_partitioning=true) group by 1"
        ).fetchall()
    finally:
        con.close()
    return {r[0]: r[1] for r in rows}


def gaps(
    universe: pl.DataFrame, last: dict[str, date], dormant: dict[str, dict[str, Any]], today: date
) -> tuple[list[str], list[str], date | None]:
    """(symbols needing full history, symbols needing a tail, latest session in the store)."""
    latest = max(last.values()) if last else None
    full: list[str] = []
    tail: list[str] = []
    for sym in universe["yahoo"].to_list():
        if sym not in last and dormant.get(sym, {}).get("empty", 0) >= US_DORMANT_AFTER:
            continue
        if sym not in last:
            full.append(sym)
        elif latest is None or last[sym] < latest or (today - last[sym]).days > 3:
            tail.append(sym)
    return full, tail, latest


def fetch_screener(root: Path, today: date) -> list[Written]:
    """Nasdaq's stock screener, reduced to its stable columns and stored as a dated snapshot only
    when those bytes change. Every snapshot held is registered. A screener outage is a note, not the
    leg's failure: the newest snapshot keeps serving."""
    s = requests.Session()
    s.headers.update(HEADERS | {"Accept": "application/json"})

    def _get() -> bytes:
        r = s.get(NASDAQ_SCREENER_URL, timeout=60)
        if r.status_code != 200 or not r.content.startswith(b"{"):
            raise Transient(f"HTTP {r.status_code}")
        return r.content

    content = U.screener_snapshot(retry(_get, what="nasdaq screener"))
    snaps = sorted((root / "raw" / "nasdaq_screener").glob("*/nasdaq_screener_*.csv.gz"))
    latest = gzip.decompress(snaps[-1].read_bytes()) if snaps else None
    if latest != content:
        rel = U.screener_rel(today)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(root / rel, gzip.compress(content, mtime=0))
    out: list[Written] = []
    for q in sorted((root / "raw" / "nasdaq_screener").glob("*/nasdaq_screener_*.csv.gz")):
        day = date(int(q.name[16:20]), int(q.name[20:22]), int(q.name[22:24]))
        out.append(_written(root, q.relative_to(root).as_posix(), "raw", "nasdaq_screener", day))
    return out


def with_indices(universe: pl.DataFrame) -> pl.DataFrame:
    """The index symbols (not in the listings) appended to the price universe."""
    extra = pl.DataFrame(
        {
            "symbol": list(US_INDICES),
            "yahoo": list(US_INDICES),
            "name": list(US_INDICES.values()),
            "exchange": ["INDEX"] * len(US_INDICES),
            "kind": ["index"] * len(US_INDICES),
        }
    )
    return pl.concat([universe, extra.filter(~pl.col("symbol").is_in(universe["symbol"]))])


def fetch_prices(
    root: Path,
    today: date,
    universe: pl.DataFrame,
    batch: Batch = fetch_batch,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[Written], dict[str, Any]]:
    """Every gapped symbol asked for its missing range, in batches; raw Parquet per batch; the dormant
    file updated. Returns the files written and the counts."""
    dormant = load_dormant(root)
    last = last_dates(root)
    full, tail, latest = gaps(universe, last, dormant, today)
    log.info(
        "us: universe %d, %d need full history, %d need a tail (store at %s), %d dormant",
        universe.height, len(full), len(tail), latest, len(dormant),
    )  # fmt: skip
    jobs: list[tuple[list[str], date | None]] = [
        (full[i : i + US_BATCH], None) for i in range(0, len(full), US_BATCH)
    ]
    if tail:
        start = min(last[s] for s in tail) - timedelta(days=5)
        jobs += [(tail[i : i + US_BATCH], start) for i in range(0, len(tail), US_BATCH)]
    written: list[Written] = []
    fetched: set[str] = set()
    n_fail = n_rows = 0
    stamp = time.strftime("%H%M%S")  # a day may see several runs; a batch file is never rewritten
    for n, (syms, start) in enumerate(jobs, 1):
        try:
            df = retry(
                functools.partial(batch, syms, start),
                what=f"yahoo batch {n}",
                attempts=3,
                base_delay=15,
            )
        except Exception as exc:  # the symbols stay gaps for the next run
            log.error("yahoo batch %d (%d symbols): %s", n, len(syms), exc)
            n_fail += 1
            continue
        got = set(df["symbol"].unique().to_list()) if df.height else set()
        if len(got) < US_MIN_HIT_RATE * len(syms):
            log.error(
                "yahoo batch %d: only %d of %d symbols answered; failed batch",
                n,
                len(got),
                len(syms),
            )
            n_fail += 1
            sleep(US_PAUSE * 5)
            continue
        for s in syms:
            if s in got:
                dormant.pop(s, None)
            else:
                rec = dormant.get(s, {"empty": 0, "since": today.isoformat()})
                rec["empty"] += 1
                dormant[s] = rec
        fetched |= got
        if df.height:
            rel = f"raw/yahoo/{today:%Y}/yahoo_{today:%Y%m%d}_{stamp}_b{n:03d}.parquet"
            buf = io.BytesIO()
            df.sort(["symbol", "date"]).write_parquet(buf, compression="zstd")
            write_if_changed(root / rel, buf.getvalue())
            written.append(_written(root, rel, "raw", "yahoo", today))
            n_rows += df.height
        sleep(US_PAUSE)
    dormant = {s: r for s, r in dormant.items() if s not in fetched}
    save_dormant(root, dormant)
    if (root / DORMANT_REL).exists():
        written.append(_written(root, DORMANT_REL, "raw", "yahoo", None))
    counts = {
        "universe": universe.height, "gapped": len(full) + len(tail), "fetched_symbols": len(fetched),
        "rows": n_rows, "batches": len(jobs), "failed_batches": n_fail,
        "dormant": sum(1 for r in dormant.values() if r["empty"] >= US_DORMANT_AFTER),
    }  # fmt: skip
    return written, counts


def register_existing(root: Path) -> list[Written]:
    """Every raw batch and listing file already on disk, so a fresh ledger knows them."""
    out: list[Written] = []
    for p in U.raw_batches(root):
        day = date(int(p.name[6:10]), int(p.name[10:12]), int(p.name[12:14]))
        out.append(_written(root, p.relative_to(root).as_posix(), "raw", "yahoo", day))
    for d in U.listing_days(root):
        for f in LISTING_FILES:
            rel = U.listing_rel(d, f)
            if (root / rel).exists():
                out.append(_written(root, rel, "raw", "nasdaq", d))
    for q in sorted((root / "raw" / "nasdaq_screener").glob("*/nasdaq_screener_*.csv.gz")):
        day = date(int(q.name[16:20]), int(q.name[20:22]), int(q.name[22:24]))
        out.append(_written(root, q.relative_to(root).as_posix(), "raw", "nasdaq_screener", day))
    return out


def build(root: Path) -> list[Written]:
    out: list[Written] = []
    U.build_us_equity(root)
    for p in sorted((root / "parquet" / "us_equity").glob("year=*/*.parquet")):
        day = date(int(p.name[10:14]), int(p.name[14:16]), int(p.name[16:18]))
        out.append(_written(root, p.relative_to(root).as_posix(), "parquet", "us_equity", day))
    _, symbols = U.build_us_listings(root)
    for d in U.listing_days(root):
        rel = U.us_listings_rel(d)
        if (root / rel).exists():
            out.append(_written(root, rel, "parquet", "us_listings", d))
    if symbols is not None:
        rel = reference_rel("us_symbols")
        write_if_changed(
            root / rel,
            __import__("finance.build.nse", fromlist=["parquet_bytes"]).parquet_bytes(symbols),
        )
        out.append(_written(root, rel, "reference", "us_symbols", None))
    snaps = sorted((root / "raw" / "nasdaq_screener").glob("*/nasdaq_screener_*.csv.gz"))
    if snaps:
        rel = reference_rel("us_sectors")
        write_if_changed(
            root / rel,
            __import__("finance.build.nse", fromlist=["parquet_bytes"]).parquet_bytes(
                U.parse_screener(snaps)
            ),
        )
        out.append(_written(root, rel, "reference", "us_sectors", None))
    return out


def leg(root: Path, today: date, ledger: Ledger) -> LegResult:
    result = LegResult()
    try:
        result.written += fetch_listings(root, today)
        listing_day = today
    except Exception as exc:  # a listing hiccup must not stop the price leg
        days = U.listing_days(root)
        if not days:
            raise
        listing_day = days[-1]
        log.warning("us: listings fetch failed (%s); using the %s snapshot", exc, listing_day)
        result.notes["listings_error"] = str(exc)
    try:
        result.written += fetch_screener(root, today)
    except Exception as exc:  # the sector labels are not worth a failed night
        log.warning("us: screener fetch failed (%s)", exc)
        result.notes["screener_error"] = str(exc)
    universe = with_indices(U.price_universe(root, listing_day))
    prices, counts = fetch_prices(root, today, universe)
    result.notes.update(counts)
    seen = {w.path for w in result.written} | {w.path for w in prices}
    result.written += prices
    result.written += [w for w in register_existing(root) if w.path not in seen]
    result.written += build(root)
    return result
