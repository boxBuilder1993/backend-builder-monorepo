# ruff: noqa: E501
"""The reference tables: corporate actions, holidays, symbol changes, the equity master, and the
securities table that links ISINs across face-value splits. Each is a whole-table Parquet, written on a
total sort order so a rebuild of unchanged input is byte-identical."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from finance.build.nse import as_date, as_float, as_int, as_str, csv_frame
from finance.files import read_raw
from finance.settings import LINK_MAX_GAP_DAYS

BONUS_RE = re.compile(r"bonus\s*-?\s*(\d+)\s*:\s*(\d+)", re.I)
SPLIT_RE = re.compile(
    r"from\s+r[es]\.?\s*(\d+(?:\.\d+)?)\s*/?-?\s*(?:per\s+share)?\s*to\s+r[es]\.?\s*(\d+(?:\.\d+)?)",
    re.I,
)


def classify_action(subject: str) -> tuple[str, float | None]:
    """(kind, price factor). Prices before ex_date are multiplied by the factor to compare with prices
    after it. Bonus a:b -> b/(a+b). Split from Rs X to Rs Y -> Y/X. Unparsed split-like subjects are
    all NCRPS or debenture issues, which do not affect equity prices."""
    s = subject.strip()
    low = s.lower()
    factor, kinds = 1.0, []
    for a, b in BONUS_RE.findall(s):
        a_, b_ = int(a), int(b)
        if a_ and b_:
            factor *= b_ / (a_ + b_)
            kinds.append("bonus")
    m = SPLIT_RE.search(s)
    if m and ("split" in low or "sub-division" in low or "consolidation" in low):
        old, new = float(m.group(1)), float(m.group(2))
        if old and new:
            factor *= new / old
            kinds.append("split" if new < old else "consolidation")
    if kinds:
        return "+".join(dict.fromkeys(kinds)), factor
    if "dividend" in low:
        return "dividend", None
    if "rights" in low:
        return "rights", None
    if "buy" in low and "back" in low:
        return "buyback", None
    return "other", None


CORP_SCHEMA = {
    "ex_date": pl.String, "record_date": pl.String, "isin": pl.String, "symbol": pl.String,
    "series": pl.String, "company": pl.String, "face_value": pl.String, "subject": pl.String,
    "kind": pl.String, "factor": pl.Float64,
}  # fmt: skip


def parse_corp_actions(paths: list[Path]) -> pl.DataFrame:
    recs: list[dict[str, object]] = []
    for p in paths:
        for r in json.loads(read_raw(p)):
            kind, factor = classify_action(r.get("subject") or "")
            recs.append(
                {
                    "ex_date": r.get("exDate"),
                    "record_date": r.get("recDate"),
                    "isin": r.get("isin"),
                    "symbol": r.get("symbol"),
                    "series": r.get("series"),
                    "company": r.get("comp"),
                    "face_value": r.get("faceVal"),
                    "subject": (r.get("subject") or "").strip(),
                    "kind": kind,
                    "factor": factor,
                }  # fmt: skip
            )
    df = pl.DataFrame(recs, schema=CORP_SCHEMA)
    return (
        df.with_columns(
            as_date("ex_date", "ex_date", "%d-%b-%Y"),
            as_date("record_date", "record_date", "%d-%b-%Y"),
            as_float("face_value", "face_value"),
        )
        .unique(subset=["symbol", "series", "ex_date", "subject"], keep="first")
        .sort(["ex_date", "symbol", "series", "subject"])
    )


def parse_holidays(paths: list[Path]) -> pl.DataFrame:
    recs: list[dict[str, object]] = []
    for p in paths:
        for segment, rows in json.loads(read_raw(p)).items():
            for r in rows or []:
                recs.append(
                    {
                        "date": r.get("tradingDate"),
                        "segment": segment,
                        "description": r.get("description"),
                    }
                )
    df = pl.DataFrame(
        recs, schema={"date": pl.String, "segment": pl.String, "description": pl.String}
    )
    return (
        df.with_columns(as_date("date", "date", "%d-%b-%Y"))
        .unique()
        .sort(["date", "segment", "description"])
    )


def parse_symbol_changes(path: Path) -> pl.DataFrame:
    recs: list[dict[str, str]] = []
    for line in read_raw(path).decode("utf-8", "replace").splitlines():
        parts = line.rsplit(",", 3)
        if len(parts) != 4:
            continue
        company, old, new, dt = (x.strip() for x in parts)
        recs.append({"company": company, "old_symbol": old, "new_symbol": new, "change_date": dt})
    df = pl.DataFrame(
        recs,
        schema={
            "company": pl.String,
            "old_symbol": pl.String,
            "new_symbol": pl.String,
            "change_date": pl.String,
        },
    )
    return df.with_columns(as_date("change_date", "change_date", "%d-%b-%Y")).sort(
        ["change_date", "old_symbol", "new_symbol"]
    )


def parse_equity_master(path: Path) -> pl.DataFrame:
    df = csv_frame(read_raw(path))
    return df.select(
        as_str("SYMBOL", "symbol"),
        as_str("NAME OF COMPANY", "company"),
        as_str("SERIES", "series"),
        as_date("DATE OF LISTING", "listing_date", "%d-%b-%Y"),
        as_float("PAID UP VALUE", "paid_up_value"),
        as_int("MARKET LOT", "market_lot"),
        as_str("ISIN NUMBER", "isin"),
        as_float("FACE VALUE", "face_value"),
    ).sort(["symbol", "series", "isin"])


INDEX_SNAPSHOT = re.compile(r"nse_indices_(\d{8})_(.+)\.csv\.gz$")


def parse_index_constituents(paths: list[Path], files: dict[str, str]) -> pl.DataFrame:
    """Every snapshot of every index file into one table: index, symbol, isin, company, industry,
    series, as_of (the snapshot's day). ``files`` maps the index name to the archive file name, as
    settings.INDEX_FILES; a snapshot of a file not in the map is skipped."""
    by_file = {v: k for k, v in files.items()}
    frames: list[pl.DataFrame] = []
    for path in paths:
        m = INDEX_SNAPSHOT.search(path.name)
        if not m or m.group(2) not in by_file:
            continue
        as_of = date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:]))
        df = csv_frame(read_raw(path))
        frames.append(
            df.select(
                pl.lit(by_file[m.group(2)]).alias("index"),
                as_str("Symbol", "symbol"),
                as_str("ISIN Code", "isin"),
                as_str("Company Name", "company"),
                as_str("Industry", "industry"),
                as_str("Series", "series"),
                pl.lit(as_of).alias("as_of"),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "index": pl.Utf8,
                "symbol": pl.Utf8,
                "isin": pl.Utf8,
                "company": pl.Utf8,
                "industry": pl.Utf8,
                "series": pl.Utf8,
                "as_of": pl.Date,
            }
        )
    return pl.concat(frames).sort(["index", "as_of", "symbol", "isin"])


def build_securities(root: Path) -> pl.DataFrame | None:
    """ISINs that belong to the same security get one stable sid, the earliest ISIN in the chain. NSE
    issues a new ISIN on a face-value split, so the same symbol switches ISIN from one day to the
    next; those get linked. Renames keep the ISIN and need nothing. None when no equity is built."""
    glob = (root / "parquet" / "equity" / "*" / "*.parquet").as_posix()
    if not any((root / "parquet" / "equity").glob("year=*/*.parquet")):
        return None
    con = duckdb.connect()
    src = f"read_parquet('{glob}', hive_partitioning=true)"
    edges = con.execute(
        f"""
        with s as (
            select date, symbol, series, isin,
                   lag(isin) over w as prev_isin, lag(date) over w as prev_date
            from {src} where isin is not null and isin <> ''
            window w as (partition by symbol, series order by date))
        select distinct prev_isin, isin from s
        where prev_isin is not null and prev_isin <> isin and date - prev_date <= {LINK_MAX_GAP_DAYS}
    """
    ).fetchall()
    spans = con.execute(
        f"""
        select isin, min(date), max(date), arg_max(symbol, date), count(*)
        from {src} where isin is not null and isin <> '' group by isin
    """
    ).fetchall()
    con.close()
    first: dict[str, date] = {isin: d0 for isin, d0, _, _, _ in spans}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    for a, b in edges:
        if a in first and b in first:
            parent[find(a)] = find(b)
    members: dict[str, list[str]] = defaultdict(list)
    for isin in first:
        members[find(isin)].append(isin)
    sid_of: dict[str, str] = {}
    for group in members.values():
        sid = min(group, key=lambda i: (first[i], i))
        for i in group:
            sid_of[i] = sid
    return pl.DataFrame(
        {
            "isin": [r[0] for r in spans],
            "sid": [sid_of[r[0]] for r in spans],
            "first_date": [r[1] for r in spans],
            "last_date": [r[2] for r in spans],
            "last_symbol": [r[3] for r in spans],
            "n_days": [r[4] for r in spans],
        }
    ).sort(["sid", "first_date", "isin"])
