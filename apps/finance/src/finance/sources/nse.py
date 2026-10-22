"""The nse leg: NSE cm and full bhavcopy, NSE indices, MCX futures; then equity, index and commodity
Parquet for the days that are complete.

Everything the old poller learned about these sources by probing is here as code: the cm bhavcopy's
two layouts and their date ranges, the previous session's file served on non-trading days, the HTML
page from niftyindices.com, the Excel workbook at a .csv URL, MCX behind Akamai. Each is a test.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Iterable
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from finance.build import mcx as M
from finance.build import nse as N
from finance.files import parquet_rel, raw_rel, read_raw, sha256, write_if_changed, write_raw
from finance.http import NseArchives, Transient, looks_html, retry
from finance.settings import BACKFILL_START, OLD_CM_END, REQUEST_PAUSE, UDIFF_START
from finance.sources import Ledger, LegResult, NoData, Written
from finance.sources.daily import Fetcher, Got, NoFile
from finance.sources.mcx import McxBhav

log = logging.getLogger("finance.nse")


# ---- fetchers ---------------------------------------------------------------------------------


class NseCm:
    """cm bhavcopy (has ISIN). UDiFF layout where available, the old layout before it; a zip."""

    name, ext, host = "nse_cm", "zip", "nsearchives"
    UDIFF = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    OLD = (
        "https://nsearchives.nseindia.com/content/historical/EQUITIES/{d:%Y}/{mon}/"
        "cm{d:%d}{mon}{d:%Y}bhav.csv.zip"
    )

    def __init__(self, client: NseArchives) -> None:
        self.c = client

    def urls(self, day: date) -> list[str]:
        urls: list[str] = []
        if day >= UDIFF_START:
            urls.append(self.UDIFF.format(d=day))
        if day <= OLD_CM_END:
            urls.append(self.OLD.format(d=day, mon=f"{day:%b}".upper()))
        return urls

    def fetch(self, day: date) -> Got | NoFile:
        for url in self.urls(day):
            r = self.c.get(url)
            if r.status_code == 404:
                continue
            if r.status_code != 200:
                raise Transient(f"HTTP {r.status_code}")
            if r.content[:2] != b"PK":
                raise Transient("not a zip" + (" (html page)" if looks_html(r.content) else ""))
            found = N.cm_date(r.content)
            if found != day:
                return NoFile(f"stale:{found}")
            return Got(r.content)
        return NoFile("404")


class NseFull:
    """sec_bhavdata_full: OHLC plus delivery, no ISIN. The date inside is checked because NSE serves
    the previous session's file with 200 on non-trading days. Kept as served, Excel included."""

    name, ext, host = "nse_full", "csv.gz", "nsearchives"
    URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"

    def __init__(self, client: NseArchives) -> None:
        self.c = client

    def fetch(self, day: date) -> Got | NoFile:
        r = self.c.get(self.URL.format(d=day))
        if r.status_code == 404:
            return NoFile("404")
        if r.status_code != 200:
            raise Transient(f"HTTP {r.status_code}")
        if looks_html(r.content):
            raise Transient("html page instead of csv")
        found = N.full_date(r.content)
        if found != day:
            return NoFile(f"stale:{found}")
        return Got(r.content)


class NseIndex:
    """ind_close_all from niftyindices.com. An HTML page on non-trading days."""

    name, ext, host = "nse_index", "csv.gz", "niftyindices"
    URL = "https://www.niftyindices.com/Daily_Snapshot/ind_close_all_{d:%d%m%Y}.csv"

    def __init__(self, client: NseArchives) -> None:
        self.c = client

    def fetch(self, day: date) -> Got | NoFile:
        r = self.c.get(self.URL.format(d=day), headers={"referer": "https://www.nseindia.com"})
        if r.status_code == 404:
            return NoFile("404")
        if r.status_code != 200:
            raise Transient(f"HTTP {r.status_code}")
        if looks_html(r.content):
            return NoFile("html")
        found = N.index_date(r.content)
        if found != day:
            return NoFile(f"stale:{found}")
        return Got(r.content)


def fetchers() -> list[Fetcher]:
    nse = NseArchives()
    return [NseCm(nse), NseFull(nse), NseIndex(nse), McxBhav()]


# ---- the leg ----------------------------------------------------------------------------------

Sleep = Callable[[float], None]


def days(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def written_file(root: Path, rel: str, kind: str, source: str, day: date) -> Written:
    p = root / rel
    return Written(rel, kind, source, day, p.stat().st_size, sha256(p))


def fetch_sources(
    root: Path, today: date, ledger: Ledger, srcs: list[Fetcher], sleep: Sleep = time.sleep
) -> LegResult:
    """Every calendar day from the backfill start to today, per source: a file on disk is
    registered, a trusted no-data day is skipped, anything else is fetched. Weekends are never
    skipped by rule; NSE has Saturday sessions. Sequential, a pause after every request."""
    out = LegResult()
    requests = 0
    for src in srcs:
        pause = REQUEST_PAUSE.get(src.host, 0.5)
        for day in days(BACKFILL_START, today):
            rel = raw_rel(src.name, day, src.ext)
            if (root / rel).exists():
                out.written.append(written_file(root, rel, "raw", src.name, day))
                continue
            if (src.name, day) in ledger.no_data:
                continue
            try:
                got = retry(functools.partial(src.fetch, day), what=f"{src.name} {day}")
            except (
                Exception
            ) as exc:  # one bad day must not stop the rest; it is tried again tomorrow
                log.error("%s %s: giving up: %s", src.name, day, exc)
                out.notes.setdefault("errors", []).append(f"{src.name} {day}: {exc}")
                continue
            finally:
                requests += 1
                sleep(pause)
            if isinstance(got, Got):
                write_raw(root / rel, got.content)
                out.written.append(written_file(root, rel, "raw", src.name, day))
            else:
                out.no_data.append(NoData(src.name, day))
    out.notes["requests"] = requests
    return out


def _raw_days(result: LegResult, source: str) -> set[date]:
    return {w.day for w in result.written if w.source == source and w.kind == "raw" and w.day}


def build_tables(root: Path, result: LegResult, ledger: Ledger) -> None:
    """Parquet for every day whose inputs are complete. Equity needs the cm file and either the
    delivery file or the knowledge that there is none, so a recent day waits for delivery data.
    A Parquet file that exists is registered, never rebuilt."""
    cm = _raw_days(result, "nse_cm")
    full = _raw_days(result, "nse_full")
    full_none = {d for s, d in ledger.no_data if s == "nse_full"} | {
        nd.day for nd in result.no_data if nd.source == "nse_full"
    }
    plan: list[tuple[str, date]] = []
    plan += [("equity", d) for d in sorted(cm) if d in full or d in full_none]
    plan += [("index", d) for d in sorted(_raw_days(result, "nse_index"))]
    plan += [("commodity", d) for d in sorted(_raw_days(result, "mcx"))]
    built = 0
    for table, day in plan:
        rel = parquet_rel(table, day)
        if not (root / rel).exists():
            try:
                df = build_day(root, table, day)
            except Exception as exc:  # one bad day must not stop the rest
                log.error("%s %s: %s", table, day, exc)
                result.notes.setdefault("errors", []).append(f"{table} {day}: {exc}")
                continue
            if df.is_empty():
                log.warning("%s %s: parsed 0 rows, not written", table, day)
                continue
            write_if_changed(root / rel, N.parquet_bytes(df))
            built += 1
        result.written.append(written_file(root, rel, "parquet", table, day))
    result.notes["built"] = built
    waiting = sorted(d for d in cm if d not in full and d not in full_none)
    if waiting:
        result.notes["equity_waiting"] = [d.isoformat() for d in waiting[-5:]]


def build_day(root: Path, table: str, day: date) -> pl.DataFrame:
    if table == "equity":
        cm = N.parse_cm(read_raw(root / raw_rel("nse_cm", day, "zip")))
        fp = root / raw_rel("nse_full", day, "csv.gz")
        full = N.parse_full(read_raw(fp)) if fp.exists() else None
        df = N.equity_day(cm, full)
    elif table == "index":
        df = N.parse_index(read_raw(root / raw_rel("nse_index", day, "csv.gz")))
    elif table == "commodity":
        df = M.parse_mcx(read_raw(root / raw_rel("mcx", day, "json.gz")))
    else:
        raise ValueError(table)
    if not df.is_empty():
        N.check_day(df, day, table)
    return df


def leg(root: Path, today: date, ledger: Ledger) -> LegResult:
    result = fetch_sources(root, today, ledger, fetchers())
    build_tables(root, result, ledger)
    return result
