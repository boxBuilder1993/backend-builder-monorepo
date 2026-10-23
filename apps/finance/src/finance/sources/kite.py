"""The kite leg: one-minute candles from Zerodha's Kite Connect, one raw file per trading day.

Kite serves about three years of candles, at most 60 days per request, three requests a second, and
only with an access token that expires every morning. The login cannot be automated within Zerodha's
terms; `make kite-login` in the old repo does it by hand, writing ~/.config/ifin/kite_token.json. A
run without a valid token is an outcome, "login needed", and the days wait for the next run, which
fills them because Kite serves history. The universe is the liquid stocks of the daily store plus a
few ETFs and indices; the instrument dump is a reference table.

Raw layout: raw/kite_1m/YYYY/kite_1m_YYYYMMDD.jsonl.gz, one line per instrument,
{"symbol", "token", "candles": [[ts, o, h, l, c, v], ...]} exactly as returned.
"""

# ruff: noqa: E501
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

import gzip
import json
import logging
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from finance.build import kite as K
from finance.build.nse import parquet_bytes
from finance.files import marker_rel, parquet_rel, raw_rel, reference_rel, sha256, write_if_changed
from finance.settings import (
    KITE_CONFIG,
    KITE_EXTRA,
    KITE_INDICES,
    KITE_INTERVAL,
    KITE_MIN_TURNOVER,
    KITE_PAUSE,
    KITE_RANGE_DAYS,
    KITE_START,
    KITE_TOKEN,
)
from finance.sources import Ledger, LegResult, NoData, Written

log = logging.getLogger("finance.kite")
NAME, EXT, TABLE = "kite_1m", "jsonl.gz", "equity_1m"


class KiteLoginNeeded(RuntimeError):
    pass


def client() -> Any:
    """A KiteConnect client with today's access token, or KiteLoginNeeded."""
    from kiteconnect import KiteConnect

    if not KITE_CONFIG.exists():
        raise KiteLoginNeeded(f"no {KITE_CONFIG}: put api_key and api_secret there")
    cfg = tomllib.loads(KITE_CONFIG.read_text())
    if not KITE_TOKEN.exists():
        raise KiteLoginNeeded("no access token: run `make kite-login`")
    tok = json.loads(KITE_TOKEN.read_text())
    if tok.get("date") != date.today().isoformat():
        raise KiteLoginNeeded(f"access token is from {tok.get('date')}: run `make kite-login`")
    kite = KiteConnect(api_key=cfg["api_key"])
    kite.set_access_token(tok["access_token"])
    return kite


@dataclass(frozen=True)
class Instrument:
    symbol: str
    token: int


def _written(root: Path, rel: str, kind: str, source: str, day: date | None) -> Written:
    p = root / rel
    return Written(rel, kind, source, day, p.stat().st_size, sha256(p))


def save_instruments(root: Path, kite: Any) -> Written:
    """Kite's NSE instrument dump as a reference table, written only when it changes."""
    rows = kite.instruments("NSE")
    df = (
        pl.DataFrame(rows)
        .select(
            "instrument_token",
            "exchange_token",
            "tradingsymbol",
            "name",
            "segment",
            "instrument_type",
            "exchange",
            "tick_size",
            "lot_size",
        )  # fmt: skip
        .sort("instrument_token")
    )
    rel = reference_rel("kite_instruments")
    write_if_changed(root / rel, parquet_bytes(df))
    return _written(root, rel, "reference", "kite_instruments", None)


def universe(root: Path) -> tuple[list[Instrument], list[str]]:
    """Liquid stocks as of the daily store's latest session, plus the configured ETFs and indices,
    mapped to instrument tokens. Returns (instruments, symbols without a token)."""
    glob = (root / "parquet" / "equity" / "*" / "*.parquet").as_posix()
    con = duckdb.connect()
    try:
        liquid = [
            r[0]
            for r in con.execute(
                f"""
            with e as (select * from read_parquet('{glob}', hive_partitioning=true) where series = 'EQ'),
            last as (select max(date) d from e),
            w as (select isin, symbol, turnover, row_number() over (partition by isin order by date desc) rn
                  from e where date > (select d from last) - interval 400 day)
            select last(symbol order by rn desc) symbol from w where rn <= 250 group by isin
            having median(turnover) >= {KITE_MIN_TURNOVER} and count(*) >= 120 order by 1"""
            ).fetchall()
        ]
    finally:
        con.close()
    inst = pl.read_parquet(root / reference_rel("kite_instruments"))
    eq = {
        r["tradingsymbol"]: r["instrument_token"]
        for r in inst.filter(pl.col("segment") == "NSE").iter_rows(named=True)
    }
    idx = {
        r["tradingsymbol"]: r["instrument_token"]
        for r in inst.filter(pl.col("segment") == "INDICES").iter_rows(named=True)
    }
    out: list[Instrument] = []
    missing: list[str] = []
    for s in list(dict.fromkeys(liquid + list(KITE_EXTRA))):
        (out.append(Instrument(s, eq[s])) if s in eq else missing.append(s))
    for s in KITE_INDICES:
        (out.append(Instrument(s, idx[s])) if s in idx else missing.append(s))
    return out, missing


def ranges(days: list[date]) -> list[tuple[date, date]]:
    """Contiguous calendar ranges of at most KITE_RANGE_DAYS covering the days."""
    out: list[tuple[date, date]] = []
    for d in sorted(days):
        if out and (d - out[-1][0]).days < KITE_RANGE_DAYS and (d - out[-1][1]).days <= 7:
            out[-1] = (out[-1][0], d)
        else:
            out.append((d, d))
    return out


Fetch = Callable[[Any, Instrument, date, date], list[dict[str, Any]]]


def fetch_candles(kite: Any, inst: Instrument, a: date, b: date) -> list[dict[str, Any]]:
    for attempt in range(4):
        try:
            time.sleep(KITE_PAUSE)
            return kite.historical_data(
                inst.token,
                datetime.combine(a, datetime.min.time()),
                datetime.combine(b, datetime.max.time().replace(microsecond=0)),
                KITE_INTERVAL,
            )
        except Exception as exc:
            name = type(exc).__name__
            if name in ("TokenException", "PermissionException"):
                raise KiteLoginNeeded(str(exc)) from exc
            if attempt == 3:
                raise
            log.warning("kite %s %s..%s: %s, retrying", inst.symbol, a, b, name)
            time.sleep(2 * (attempt + 1))
    return []


class DayWriters:
    """One gzip JSON-lines writer per trading day, created on first use, finalised atomically."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[date, gzip.GzipFile] = {}

    def write(self, d: date, line: dict[str, Any]) -> None:
        if d not in self.files:
            p = self.root / raw_rel(NAME, d, EXT)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.files[d] = gzip.GzipFile(p.with_name(p.name + ".tmp"), "wb", mtime=0)
        self.files[d].write((json.dumps(line, separators=(",", ":")) + "\n").encode())

    def close(self) -> set[date]:
        done: set[date] = set()
        for d, f in self.files.items():
            f.close()
            p = self.root / raw_rel(NAME, d, EXT)
            Path(str(p) + ".tmp").replace(p)
            done.add(d)
        return done


def missing_days(root: Path, today: date, end: date, ledger: Ledger) -> list[date]:
    out: list[date] = []
    d = KITE_START
    while d <= end:
        if (
            not (root / raw_rel(NAME, d, EXT)).exists()
            and (NAME, d) not in ledger.no_data
            and not (root / marker_rel(NAME, d)).exists()
        ):
            out.append(d)
        d += timedelta(days=1)
    return out


def fetch(
    root: Path,
    today: date,
    ledger: Ledger,
    kite: Any,
    insts: list[Instrument],
    fetcher: Fetch = fetch_candles,
) -> tuple[list[Written], list[NoData], dict[str, Any]]:
    """Every missing trading day of candles for the universe, in 60-day ranges. Today is included
    after 16:00, when the session's candles are complete."""
    end = today if datetime.now().hour >= 16 else today - timedelta(days=1)
    missing = missing_days(root, today, end, ledger)
    written: list[Written] = []
    no_data: list[NoData] = []
    requests = 0
    for a, b in ranges(missing):
        w = DayWriters(root)
        for inst in insts:
            candles = fetcher(kite, inst, a, b)
            requests += 1
            by_day: dict[date, list[list[Any]]] = {}
            for c in candles:
                ts = c["date"]
                d = ts.date() if hasattr(ts, "date") else datetime.fromisoformat(str(ts)).date()
                if not (a <= d <= b):
                    raise ValueError(f"kite returned {d} for {inst.symbol} outside {a}..{b}")
                by_day.setdefault(d, []).append(
                    [
                        ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                        c["open"],
                        c["high"],
                        c["low"],
                        c["close"],
                        c["volume"],
                    ]
                )
            for d, rows in by_day.items():
                w.write(d, {"symbol": inst.symbol, "token": inst.token, "candles": rows})
        done = w.close()
        for d in done:
            written.append(_written(root, raw_rel(NAME, d, EXT), "raw", NAME, d))
        for d in missing:
            if a <= d <= b and d not in done:
                no_data.append(NoData(NAME, d))
        log.info("kite: %s..%s saved %d days, %d requests so far", a, b, len(done), requests)
    return written, no_data, {"kite_requests": requests, "kite_instruments": len(insts)}


def register_existing(root: Path) -> list[Written]:
    out: list[Written] = []
    for p in sorted((root / "raw" / NAME).glob("*/kite_1m_*.jsonl.gz")):
        day = date(int(p.name[8:12]), int(p.name[12:14]), int(p.name[14:16]))
        out.append(_written(root, raw_rel(NAME, day, EXT), "raw", NAME, day))
    inst = root / reference_rel("kite_instruments")
    if inst.exists():
        out.append(
            _written(root, reference_rel("kite_instruments"), "reference", "kite_instruments", None)
        )
    return out


def build(root: Path) -> list[Written]:
    """equity_1m per day for every raw day without a Parquet; existing files registered, not rebuilt."""
    out: list[Written] = []
    for p in sorted((root / "raw" / NAME).glob("*/kite_1m_*.jsonl.gz")):
        day = date(int(p.name[8:12]), int(p.name[12:14]), int(p.name[14:16]))
        rel = parquet_rel(TABLE, day)
        if not (root / rel).exists():
            df = K.parse(p.read_bytes(), day)
            if df.is_empty():
                continue
            write_if_changed(root / rel, parquet_bytes(df))
        out.append(_written(root, rel, "parquet", TABLE, day))
    return out


def leg(root: Path, today: date, ledger: Ledger) -> LegResult:
    result = LegResult()
    try:
        kite = client()
        result.written.append(save_instruments(root, kite))
        insts, missing = universe(root)
        if missing:
            result.notes["no_token"] = missing[:20]
        written, no_data, counts = fetch(root, today, ledger, kite, insts)
        result.written += written
        result.no_data += no_data
        result.notes.update(counts)
    except KiteLoginNeeded as exc:
        result.notes["kite"] = f"login needed: {exc}"
    seen = {w.path for w in result.written}
    result.written += [w for w in register_existing(root) if w.path not in seen]
    result.written += build(root)
    return result
