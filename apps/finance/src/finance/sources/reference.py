"""The reference leg: NSE corporate actions per year, the holiday master, symbol changes and the
equity master as raw files under raw/nse_meta/, then the reference Parquet tables. Small files; the
ones that grow are refreshed every run and written only when their bytes change, so an unchanged
table is not republished. Past years' corporate actions are fetched once."""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from finance.build import nse as N
from finance.build import reference as R
from finance.files import reference_rel, sha256, write_if_changed
from finance.http import NseApi, NseArchives, Transient, looks_html, retry
from finance.settings import BACKFILL_START, CORP_URL, EQUITY_L_URL, HOLIDAY_URL, SYMBOLCHANGE_URL
from finance.sources import Ledger, LegResult, Written

log = logging.getLogger("finance.reference")

META = "raw/nse_meta"


def _gz(content: bytes) -> bytes:
    return gzip.compress(content, mtime=0)


def _written(root: Path, rel: str, kind: str, source: str) -> Written:
    p = root / rel
    return Written(rel, kind, source, None, p.stat().st_size, sha256(p))


def fetch_meta(root: Path, today: date, api: NseApi, arch: NseArchives) -> list[Written]:
    """Every raw metadata file, fetched or found: corp_actions_<year>.json.gz (past years kept),
    holidays_<year>.json.gz, symbolchange.csv.gz, equity_l.csv.gz."""
    out: list[Written] = []
    (root / META).mkdir(parents=True, exist_ok=True)
    for y in range(BACKFILL_START.year, today.year + 1):
        rel = f"{META}/corp_actions_{y}.json.gz"
        if not (root / rel).exists() or y >= today.year:
            rows: Any = retry(
                lambda y=y: api.get_json(CORP_URL.format(y=y)), what=f"corp actions {y}"
            )
            if not isinstance(rows, list):
                raise Transient(f"corp actions {y}: unexpected payload {str(rows)[:80]}")
            write_if_changed(root / rel, _gz(json.dumps(rows).encode()))
        out.append(_written(root, rel, "raw", "nse_meta"))
    rel = f"{META}/holidays_{today.year}.json.gz"
    j: Any = retry(lambda: api.get_json(HOLIDAY_URL), what="holidays")
    write_if_changed(root / rel, _gz(json.dumps(j).encode()))
    out.append(_written(root, rel, "raw", "nse_meta"))
    for url, name in ((SYMBOLCHANGE_URL, "symbolchange.csv.gz"), (EQUITY_L_URL, "equity_l.csv.gz")):

        def _get(u: str = url) -> bytes:
            r = arch.get(u)
            if r.status_code != 200 or looks_html(r.content):
                raise Transient(f"HTTP {r.status_code}")
            return r.content

        content = retry(_get, what=name)
        rel = f"{META}/{name}"
        write_if_changed(root / rel, _gz(content))
        out.append(_written(root, rel, "raw", "nse_meta"))
    for p in sorted((root / META).glob("holidays_*.json.gz")):  # earlier years, kept and registered
        rel = f"{META}/{p.name}"
        if rel not in {w.path for w in out}:
            out.append(_written(root, rel, "raw", "nse_meta"))
    return out


def build_reference(root: Path) -> list[Written]:
    """Whole-table Parquet for every reference table whose raw input exists, plus securities."""
    out: list[Written] = []
    meta = root / META

    def write(name: str, df: pl.DataFrame) -> None:
        rel = reference_rel(name)
        write_if_changed(root / rel, N.parquet_bytes(df))
        out.append(_written(root, rel, "reference", name))

    corp = sorted(meta.glob("corp_actions_*.json.gz"))
    if corp:
        write("corp_actions", R.parse_corp_actions(corp))
    hol = sorted(meta.glob("holidays_*.json.gz"))
    if hol:
        write("holidays", R.parse_holidays(hol))
    if (meta / "symbolchange.csv.gz").exists():
        write("symbol_changes", R.parse_symbol_changes(meta / "symbolchange.csv.gz"))
    if (meta / "equity_l.csv.gz").exists():
        write("equity_master", R.parse_equity_master(meta / "equity_l.csv.gz"))
    securities = R.build_securities(root)
    if securities is not None:
        write("securities", securities)
    return out


def leg(root: Path, today: date, ledger: Ledger) -> LegResult:
    result = LegResult()
    result.written += fetch_meta(root, today, NseApi(), NseArchives())
    result.written += build_reference(root)
    result.notes["tables"] = [w.source for w in result.written if w.kind == "reference"]
    return result
