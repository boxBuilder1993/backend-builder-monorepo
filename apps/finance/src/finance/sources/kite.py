"""The kite leg: one-minute candles from Zerodha's Kite Connect, one raw file per trading day.

Kite serves about three years of candles, at most 60 days per request, three requests a second, and
only with an access token that expires every morning. The login cannot be automated within Zerodha's
terms; `make kite-login` in the old repo does it by hand, writing ~/.config/ifin/kite_token.json. A
run without a valid token is an outcome, "login needed", and the days wait for the next run, which
fills them because Kite serves history. The universe is the liquid stocks of the daily store, every
member of the Nifty 500 (sector labels come with them), and every stock fetched in the last year, plus
a few ETFs and indices; the instrument dump is a reference table.

Raw layout: raw/kite_1m/YYYY/kite_1m_YYYYMMDD.jsonl.gz, one line per instrument,
{"symbol", "token", "candles": [[ts, o, h, l, c, v], ...]} exactly as returned. An instrument that
joins the universe later gets its history as supplement files beside the days' files,
kite_1m_YYYYMMDD_s<stamp>.jsonl.gz, and the day's Parquet is built from all of them; kite_history is
the reference table of instruments whose history has been asked for once.
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
    KITE_STAY_SESSIONS,
    KITE_TOKEN,
    KITE_UNIVERSE_INDEX,
)
from finance.sources import BUDGETS, Ledger, LegResult, NoData, Written

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


def liquid_symbols(root: Path) -> list[str]:
    """EQ stocks whose median daily turnover over the last 250 sessions (at least 120 of them) is
    KITE_MIN_TURNOVER or more, as of the daily store's latest session."""
    glob = (root / "parquet" / "equity" / "*" / "*.parquet").as_posix()
    con = duckdb.connect()
    try:
        return [
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


def index_symbols(root: Path, index: str = KITE_UNIVERSE_INDEX) -> list[str]:
    """The members of the index in the newest constituent snapshot; empty before the first one."""
    p = root / reference_rel("index_constituents")
    if not p.exists():
        return []
    df = pl.read_parquet(p).filter(pl.col("index") == index)
    if df.is_empty():
        return []
    latest = df["as_of"].max()
    return sorted(df.filter(pl.col("as_of") == latest)["symbol"].unique().to_list())


def recent_symbols(root: Path, sessions: int = KITE_STAY_SESSIONS) -> list[str]:
    """Every stock with minute bars on any of the last ``sessions`` days held: once in, a stock stays
    in the universe for a year, so a name hovering around the turnover bar has no holes."""
    files = sorted((root / "parquet" / TABLE).glob("year=*/*.parquet"))[-sessions:]
    if not files:
        return []
    con = duckdb.connect()
    try:
        paths = ", ".join(f"'{f.as_posix()}'" for f in files)
        return [
            r[0]
            for r in con.execute(
                f"select distinct symbol from read_parquet([{paths}]) where symbol not like 'NIFTY %' order by 1"
            ).fetchall()
        ]
    finally:
        con.close()


def universe(root: Path) -> tuple[list[Instrument], list[str], dict[str, int]]:
    """The stocks to fetch: the liquid ones by the turnover rule, every member of KITE_UNIVERSE_INDEX,
    and every stock fetched in the last KITE_STAY_SESSIONS sessions; plus the configured ETFs and
    indices; mapped to instrument tokens. Returns (instruments, symbols without a token, counts)."""
    liquid, members, recent = liquid_symbols(root), index_symbols(root), recent_symbols(root)
    counts = {"liquid": len(liquid), "index": len(members), "recent": len(recent)}
    symbols = list(dict.fromkeys(liquid + members + recent + list(KITE_EXTRA)))
    counts["universe"] = len(symbols)
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
    for s in symbols:
        (out.append(Instrument(s, eq[s])) if s in eq else missing.append(s))
    for s in KITE_INDICES:
        (out.append(Instrument(s, idx[s])) if s in idx else missing.append(s))
    return out, missing, counts


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


def supplement_rel(d: date, stamp: str) -> str:
    """A day's supplement file: candles of instruments that joined the universe after the day's file
    was written. The day's raw file is never rewritten; a supplement sits beside it, named by the run
    that wrote it, and the day's Parquet is built from the file and all its supplements."""
    base = raw_rel(NAME, d, EXT)
    return base[: -len(EXT) - 1] + f"_s{stamp}.{EXT}"


def day_of_raw(p: Path) -> date:
    """kite_1m_YYYYMMDD... -> the day, for the day's file and its supplements alike."""
    return date(int(p.name[8:12]), int(p.name[12:14]), int(p.name[14:16]))


class DayWriters:
    """One gzip JSON-lines writer per trading day, created on first use, finalised atomically. With a
    ``stamp`` the writers produce the day's supplement files instead of the day's file."""

    def __init__(self, root: Path, stamp: str | None = None) -> None:
        self.root = root
        self.stamp = stamp
        self.files: dict[date, gzip.GzipFile] = {}

    def _rel(self, d: date) -> str:
        return supplement_rel(d, self.stamp) if self.stamp else raw_rel(NAME, d, EXT)

    def write(self, d: date, line: dict[str, Any]) -> None:
        if d not in self.files:
            p = self.root / self._rel(d)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.files[d] = gzip.GzipFile(p.with_name(p.name + ".tmp"), "wb", mtime=0)
        self.files[d].write((json.dumps(line, separators=(",", ":")) + "\n").encode())

    def close(self) -> set[date]:
        """Finalise every day. A day's file is moved into place; a supplement that exists already from
        this run is appended to as a further gzip member (a run's supplement for a day is one file,
        whatever the number of instruments it covered), so a run of 140 instruments over 750 sessions
        leaves 750 files, not 100,000."""
        done: set[date] = set()
        for d, f in self.files.items():
            f.close()
            p = self.root / self._rel(d)
            tmp = Path(str(p) + ".tmp")
            if self.stamp and p.exists():
                with p.open("ab") as out:
                    out.write(tmp.read_bytes())
                tmp.unlink()
            else:
                tmp.replace(p)
            done.add(d)
        self.files = {}
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
    deadline: float | None = None,
) -> tuple[list[Written], list[NoData], dict[str, Any]]:
    """Every missing trading day of candles for the universe, in 60-day ranges. Today is included
    after 16:00, when the session's candles are complete. ``deadline`` is a time.monotonic() value:
    a range is started only if it is expected to finish before it (one request per instrument at
    KITE_PAUSE each, with margin), so a long backfill ends cleanly, reports the ranges left, and
    continues next run; a range cut from outside would be lost."""
    end = today if datetime.now().hour >= 16 else today - timedelta(days=1)
    missing = missing_days(root, today, end, ledger)
    written: list[Written] = []
    no_data: list[NoData] = []
    requests = 0
    todo = ranges(missing)
    wanted = set(missing)
    ranges_left = 0
    for i, (a, b) in enumerate(todo):
        if (
            deadline is not None
            and time.monotonic() + len(insts) * KITE_PAUSE * 1.5 + 60 > deadline
        ):
            ranges_left = len(todo) - i
            log.info("kite: out of time, %d ranges left for the next run", ranges_left)
            break
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
                if d in wanted:  # a range spans held days too; those files are never rewritten
                    w.write(d, {"symbol": inst.symbol, "token": inst.token, "candles": rows})
        done = w.close()
        for d in done:
            written.append(_written(root, raw_rel(NAME, d, EXT), "raw", NAME, d))
        for d in missing:
            if a <= d <= b and d not in done:
                no_data.append(NoData(NAME, d))
        log.info("kite: %s..%s saved %d days, %d requests so far", a, b, len(done), requests)
    counts: dict[str, Any] = {"kite_requests": requests, "kite_instruments": len(insts)}
    if ranges_left:
        counts["kite_ranges_left"] = ranges_left
    return written, no_data, counts


HISTORY = "kite_history"
"""Reference table: symbol, backfilled_from, done_on. A symbol listed here has had every session from
``backfilled_from`` to its first held day asked of Kite once; it is not asked again."""


def coverage(root: Path) -> dict[str, tuple[date, date]]:
    """symbol -> (first day, last day) with minute bars held, from the Parquet."""
    files = sorted((root / "parquet" / TABLE).glob("year=*/*.parquet"))
    if not files:
        return {}
    con = duckdb.connect()
    try:
        paths = ", ".join(f"'{f.as_posix()}'" for f in files)
        rows = con.execute(
            f"select symbol, min(ts::date), max(ts::date) from read_parquet([{paths}]) group by 1"
        ).fetchall()
    finally:
        con.close()
    return {r[0]: (r[1], r[2]) for r in rows}


def history(root: Path) -> set[str]:
    p = root / reference_rel(HISTORY)
    return set(pl.read_parquet(p)["symbol"].to_list()) if p.exists() else set()


def record_history(root: Path, symbols: list[str], today: date) -> Written:
    p = root / reference_rel(HISTORY)
    old = (
        pl.read_parquet(p)
        if p.exists()
        else pl.DataFrame(
            schema={"symbol": pl.Utf8, "backfilled_from": pl.Date, "done_on": pl.Date}
        )
    )
    add = pl.DataFrame(
        {
            "symbol": symbols,
            "backfilled_from": [KITE_START] * len(symbols),
            "done_on": [today] * len(symbols),
        }
    ).cast({"backfilled_from": pl.Date, "done_on": pl.Date})
    df = pl.concat([old, add]).unique(subset=["symbol"], keep="first").sort("symbol")
    write_if_changed(p, parquet_bytes(df))
    return _written(root, reference_rel(HISTORY), "reference", HISTORY, None)


def backfill_symbols(
    root: Path,
    today: date,
    kite: Any,
    insts: list[Instrument],
    fetcher: Fetch = fetch_candles,
    deadline: float | None = None,
) -> tuple[list[Written], dict[str, Any]]:
    """History for instruments that joined the universe after the days were fetched: every session
    held (a day with a raw file) from KITE_START to the day before the instrument's first held bar,
    asked of Kite in 60-day ranges and written as supplement files, one per day for the run (every
    instrument's candles for the day appended as the instrument finishes). An instrument is done when
    all its ranges are, and recorded in kite_history; the deadline stops between instruments, so a run
    cut short resumes with the next instrument next time."""
    done_before = history(root)
    cov = coverage(root)
    sessions = sorted(
        day_of_raw(p) for p in (root / "raw" / NAME).glob("*/kite_1m_????????.jsonl.gz")
    )
    todo: list[tuple[Instrument, list[date]]] = []
    for inst in insts:
        if inst.symbol in done_before:
            continue
        first = cov.get(inst.symbol, (None, None))[0]
        days = [d for d in sessions if d >= KITE_START and (first is None or d < first)]
        if days:
            todo.append((inst, days))
        else:
            done_before.add(inst.symbol)  # covered from the start: nothing to ask
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    written: list[Written] = []
    finished: list[str] = []
    days_touched: set[date] = set()
    requests = 0
    left = 0
    for i, (inst, days) in enumerate(todo):
        rs = ranges(days)
        if deadline is not None and time.monotonic() + len(rs) * KITE_PAUSE * 1.5 + 30 > deadline:
            left = len(todo) - i
            log.info("kite backfill: out of time, %d instruments left for the next run", left)
            break
        w = DayWriters(root, stamp=stamp)
        for a, b in rs:
            candles = fetcher(kite, inst, a, b)
            requests += 1
            for c in candles:
                ts = c["date"]
                d = ts.date() if hasattr(ts, "date") else datetime.fromisoformat(str(ts)).date()
                if not (a <= d <= b):
                    raise ValueError(f"kite returned {d} for {inst.symbol} outside {a}..{b}")
                w.write(
                    d,
                    {
                        "symbol": inst.symbol,
                        "token": inst.token,
                        "candles": [
                            [
                                ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                                c["open"],
                                c["high"],
                                c["low"],
                                c["close"],
                                c["volume"],
                            ]
                        ],
                    },
                )
        days_touched.update(w.close())
        finished.append(inst.symbol)
        log.info(
            "kite backfill: %s done, %d ranges, %d requests so far", inst.symbol, len(rs), requests
        )
    for d in sorted(days_touched):  # the run's supplements, one per day, registered once complete
        written.append(_written(root, supplement_rel(d, stamp), "raw", NAME, d))
    new_done = [s for s in done_before if s not in history(root)] + finished
    if new_done:
        written.append(record_history(root, sorted(set(new_done)), today))
    counts: dict[str, Any] = {"backfill_requests": requests, "backfill_symbols": len(finished)}
    if left:
        counts["backfill_left"] = left
    return written, counts


def register_existing(root: Path) -> list[Written]:
    out: list[Written] = []
    for p in sorted((root / "raw" / NAME).glob("*/kite_1m_*.jsonl.gz")):
        day = day_of_raw(p)
        out.append(_written(root, f"raw/{NAME}/{p.parent.name}/{p.name}", "raw", NAME, day))
    for name in ("kite_instruments", HISTORY):
        if (root / reference_rel(name)).exists():
            out.append(_written(root, reference_rel(name), "reference", name, None))
    return out


def build(root: Path) -> list[Written]:
    """equity_1m per day from the day's raw file and its supplements. A day is built when it has no
    Parquet or a supplement newer than its Parquet; otherwise the existing file is registered."""
    out: list[Written] = []
    by_day: dict[date, list[Path]] = {}
    for p in sorted((root / "raw" / NAME).glob("*/kite_1m_*.jsonl.gz")):
        by_day.setdefault(day_of_raw(p), []).append(p)
    for day, files in sorted(by_day.items()):
        rel = parquet_rel(TABLE, day)
        target = root / rel
        stale = target.exists() and any(f.stat().st_mtime > target.stat().st_mtime for f in files)
        if not target.exists() or stale:
            frames = [K.parse(f.read_bytes(), day) for f in files]
            df = (
                pl.concat(frames)
                .unique(subset=["symbol", "ts"], keep="last")
                .sort(["symbol", "ts"])
            )
            if df.is_empty():
                continue
            write_if_changed(target, parquet_bytes(df))
        out.append(_written(root, rel, "parquet", TABLE, day))
    return out


def leg(root: Path, today: date, ledger: Ledger) -> LegResult:
    result = LegResult()
    deadline = time.monotonic() + BUDGETS["kite"] * 0.9
    try:
        kite = client()
        result.written.append(save_instruments(root, kite))
        insts, missing, counts = universe(root)
        result.notes["universe"] = counts
        if missing:
            result.notes["no_token"] = missing[:20]
        written, no_data, counts = fetch(root, today, ledger, kite, insts, deadline=deadline)
        result.written += written
        result.no_data += no_data
        result.notes.update(counts)
        written, counts = backfill_symbols(root, today, kite, insts, deadline=deadline)
        result.written += written
        result.notes.update(counts)
    except KiteLoginNeeded as exc:
        result.notes["kite"] = f"login needed: {exc}"
    seen = {w.path for w in result.written}
    result.written += [w for w in register_existing(root) if w.path not in seen]
    result.written += build(root)
    return result
