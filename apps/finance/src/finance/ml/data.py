"""Rows for fitting and inference: DuckDB over the mirror, the query with its named params, then the
frame coerced to a schema and encoded. Runtime-free; tested on frames.

The same views the research code used, rebuilt over the mirror's Parquet: equity_raw, equity (EQ/BE
with a stable sid), equity_adj (split and bonus adjusted), index_daily, commodity_raw and
commodity_continuous, us_equity, us_listings, equity_1m, and the reference tables.
"""

# ruff: noqa: E501  (the view SQL reads better unwrapped)
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
# polars and duckdb type loosely; the frames are checked by the tests.

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from finance.ml.schema import Feature, Schema, Target

DEFAULT_SERIES = ("EQ", "BE")
DUCKDB_MEMORY = os.environ.get("FINANCE_DUCKDB_MEMORY", "12GB")
DUCKDB_TEMP_MAX = os.environ.get("FINANCE_DUCKDB_TEMP_MAX", "80GB")
"""The most a rows build may spill to disk before it fails instead of filling the drive: a build that
needs more has the wrong shape (one did, at 201 GB, on 2026-10-24)."""
"""DuckDB's cap for a rows build; beyond it the build spills to the work dir. DuckDB's own default is
80% of the box, which leaves Postgres and the worker nothing while twenty million rows are built."""
ROLL_DAYS = 5


def work_dir() -> Path:
    """Scratch for frames passed between reactions and the model cache: $XDG_CACHE_HOME/finance."""
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "finance"


# ---- the views -------------------------------------------------------------------------------------


def _glob(root: Path, table: str) -> str:
    return (root / "parquet" / table / "*" / "*.parquet").as_posix()


def _ref(root: Path, name: str) -> str:
    return (root / "parquet" / "reference" / f"{name}.parquet").as_posix()


def _has_days(root: Path, table: str) -> bool:
    return any((root / "parquet" / table).glob("year=*/*.parquet"))


def _has_ref(root: Path, name: str) -> bool:
    return Path(_ref(root, name)).exists()


def tune(
    con: duckdb.DuckDBPyConnection, threads: int | None = None, memory: str = DUCKDB_MEMORY
) -> None:
    """Settings for a rows build: the memory cap, a spill directory under the work dir with a size
    cap, and no insertion-order preservation (the rows are sorted at the end anyway)."""
    tmp = work_dir() / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = [
        f"set memory_limit='{memory}'",
        f"set temp_directory='{tmp.as_posix()}'",
        f"set max_temp_directory_size='{DUCKDB_TEMP_MAX}'",
        "set preserve_insertion_order=false",
    ]
    if threads:
        parts.append(f"set threads={threads}")
    con.execute("; ".join(parts))


def connect(root: Path) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with every view the mirror can support."""
    con = duckdb.connect()
    install_views(con, root)
    return con


def install_views(con: duckdb.DuckDBPyConnection, root: Path) -> list[str]:
    made: list[str] = []

    def view(name: str, sql: str) -> None:
        con.execute(f"create or replace view {name} as {sql}")
        made.append(name)

    for ref in (
        "corp_actions",
        "holidays",
        "symbol_changes",
        "equity_master",
        "securities",
        "index_constituents",
        "kite_instruments",
        "us_sectors",
    ):
        if _has_ref(root, ref):
            view(ref, f"select * from read_parquet('{_ref(root, ref)}')")
    if _has_days(root, "equity"):
        view(
            "equity_raw",
            f"select * exclude (year) from read_parquet('{_glob(root, 'equity')}', hive_partitioning=true)",
        )
        series = ", ".join(f"'{s}'" for s in DEFAULT_SERIES)
        if _has_ref(root, "securities"):
            view(
                "equity",
                f"select e.*, coalesce(s.sid, e.isin) as sid from equity_raw e "
                f"left join securities s using (isin) where e.series in ({series})",
            )
        else:
            view(
                "equity",
                f"select e.*, e.isin as sid from equity_raw e where e.series in ({series})",
            )
        if _has_ref(root, "corp_actions"):
            if _has_ref(root, "securities"):
                view(
                    "adj_factors",
                    """
                    with c as (select * from corp_actions where factor is not null and ex_date is not null),
                    by_isin as (select c.ex_date, c.symbol, c.subject, c.kind, c.factor, s.sid
                                from c join securities s using (isin)),
                    rest as (select c.* from c anti join securities s using (isin)),
                    days as (select symbol, date, any_value(sid) as sid from equity group by 1, 2),
                    by_symbol as (select r.ex_date, r.symbol, r.subject, r.kind, r.factor, e.sid
                                  from rest r asof left join days e
                                  on r.symbol = e.symbol and r.ex_date <= e.date)
                    select * from by_isin union all select * from by_symbol""",
                )
            else:
                view(
                    "adj_factors",
                    """
                    with days as (select symbol, date, any_value(sid) as sid from equity group by 1, 2)
                    select c.ex_date, c.symbol, c.subject, c.kind, c.factor, e.sid
                    from (select * from corp_actions where factor is not null and ex_date is not null) c
                    asof left join days e on c.symbol = e.symbol and c.ex_date <= e.date""",
                )
            view(
                "equity_adj",
                """
                with agg as (select sid, ex_date, product(factor) as f from adj_factors
                             where sid is not null group by 1, 2),
                cum as (select sid, ex_date,
                               exp(sum(ln(f)) over (partition by sid order by ex_date desc
                                                    rows between unbounded preceding and current row)) as cum
                        from agg)
                select e.*, coalesce(c.cum, 1.0) as adj_factor,
                       e.open * coalesce(c.cum, 1.0) as adj_open, e.high * coalesce(c.cum, 1.0) as adj_high,
                       e.low * coalesce(c.cum, 1.0) as adj_low, e.close * coalesce(c.cum, 1.0) as adj_close,
                       (e.volume / coalesce(c.cum, 1.0))::bigint as adj_volume
                from equity e asof left join cum c on e.sid = c.sid and e.date < c.ex_date""",
            )
    for table, name in (
        ("equity_1m", "equity_1m"),
        ("us_equity", "us_equity"),
        ("us_listings", "us_listings"),
        ("index", "index_daily"),
    ):
        if _has_days(root, table):
            view(
                name,
                f"select * exclude (year) from read_parquet('{_glob(root, table)}', hive_partitioning=true)",
            )
    if _has_ref(root, "us_symbols"):
        view("us_symbols", f"select * from read_parquet('{_ref(root, 'us_symbols')}')")
    if _has_days(root, "commodity"):
        view(
            "commodity_raw",
            f"select * exclude (year) from read_parquet('{_glob(root, 'commodity')}', hive_partitioning=true)",
        )
        con.execute(
            """
            create or replace macro commodity_near(roll_days) as table
            select * from commodity_raw
            where instrument = 'FUTCOM' and expiry - date >= roll_days
            qualify row_number() over (partition by date, symbol order by expiry) = 1"""
        )
        view("commodity_continuous", f"select * from commodity_near({ROLL_DAYS})")
    return made


# ---- the query -------------------------------------------------------------------------------------

_PARAM = re.compile(r"(?<!:):([A-Za-z_]\w*)")


def param_names(sql: str) -> list[str]:
    """The :name parameters a query uses, in order of first appearance; ``::`` casts are not params."""
    seen: list[str] = []
    for m in _PARAM.finditer(sql):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def resolve_params(declared: dict[str, Any], given: dict[str, Any] | None) -> dict[str, Any]:
    """Declared defaults overlaid with what the run passed; a name not declared is an error."""
    extra = set(given or {}) - set(declared)
    if extra:
        raise ValueError(f"params not declared by the Fit: {sorted(extra)}")
    return {**declared, **(given or {})}


def relation(
    con: duckdb.DuckDBPyConnection, sql: str, params: dict[str, Any]
) -> duckdb.DuckDBPyRelation:
    """The query with :name params bound by name, unexecuted. Every param the query uses must be given."""
    missing = [p for p in param_names(sql) if p not in params]
    if missing:
        raise ValueError(f"query uses params without values: {missing}")
    return con.sql(_PARAM.sub(r"$\1", sql), params=params)


def run_query(con: duckdb.DuckDBPyConnection, sql: str, params: dict[str, Any]) -> pl.DataFrame:
    """Run SQL with :name params bound by name, into a frame."""
    return relation(con, sql, params).pl()


CHUNK_MONTHS = 3
CHUNK_WARMUP_DAYS = 45
"""A rows query with ``start`` and ``end`` params is run one quarter at a time, each chunk asked from
45 calendar days before its start (the longest intraday lookback is 20 sessions of relative volume)
and kept from its start: the per-minute windows over the whole universe (peers, breadth) sort every
row of the range, and three years at once ran DuckDB out of 18 GB; a quarter fits in a few."""


def chunks(start: date, end: date, months: int) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    a = start
    while a <= end:
        y, m = a.year + (a.month - 1 + months) // 12, (a.month - 1 + months) % 12 + 1
        b = min(date(y, m, 1) - timedelta(days=1), end)
        out.append((a, b))
        a = b + timedelta(days=1)
    return out


def _as_date(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v)[:10])


def write_rows(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    params: dict[str, Any],
    out: Path,
    chunk_months: int = CHUNK_MONTHS,
    warmup_days: int = CHUNK_WARMUP_DAYS,
) -> int:
    """The query straight to Parquet, never through a Python-side frame: DuckDB streams it. With
    ``start`` and ``end`` params the range is built in chunks (see CHUNK_MONTHS), each written as a
    part and the parts concatenated into ``out``; a row's chunk is decided by its ``ts``. Returns the
    row count. The forecast rows are tens of millions; a frame of them would not fit beside the fit."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if not ("start" in params and "end" in params) or not chunk_months:
        relation(con, sql, params).write_parquet(out.as_posix(), compression="zstd")
        return int(
            con.sql(f"select count(*) from read_parquet('{out.as_posix()}')").fetchall()[0][0]
        )
    start, end = _as_date(params["start"]), min(_as_date(params["end"]), date.today())
    parts = out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    for q in parts.glob("*.parquet"):
        q.unlink()
    for i, (a, b) in enumerate(chunks(start, end, chunk_months)):
        chunk = {
            **params,
            "start": (a - timedelta(days=warmup_days)).isoformat(),
            "end": b.isoformat(),
        }
        rel = relation(con, sql, chunk)
        if "ts" in rel.columns:
            rel = rel.filter(f"ts >= TIMESTAMP '{a.isoformat()} 00:00:00'")
        rel.write_parquet((parts / f"{i:04d}.parquet").as_posix(), compression="zstd")
    con.sql(
        f"copy (select * from read_parquet('{parts.as_posix()}/*.parquet')) "
        f"to '{out.as_posix()}' (format parquet, compression zstd)"
    )
    for q in parts.glob("*.parquet"):
        q.unlink()
    parts.rmdir()
    return int(con.sql(f"select count(*) from read_parquet('{out.as_posix()}')").fetchall()[0][0])


def file_hash(path: Path) -> str:
    """A stable, order-independent hash of a Parquet file's content and column types: the record of
    what a fit read, computed by DuckDB without loading the file into Python."""
    con = duckdb.connect()
    try:
        src = f"read_parquet('{path.as_posix()}')"
        cols = con.sql(f"describe select * from {src}").fetchall()
        n, x = con.sql(f"select count(*), bit_xor(hash(*COLUMNS(*))) from {src}").fetchall()[0]
    finally:
        con.close()
    h = hashlib.sha256()
    h.update(",".join(f"{c[0]}:{c[1]}" for c in cols).encode())
    h.update(f"{n}:{x}".encode())
    return h.hexdigest()


def read_rows(path: Path, schema: Schema, fitting: bool) -> pl.DataFrame:
    """The rows for a schema: numeric features and regression targets as Float32 while reading, the
    key, categoricals and any extra columns as they are. Half the memory of reading everything as
    Float64 and casting after."""
    lf = pl.scan_parquet(path)
    names = set(lf.collect_schema().names())
    numeric = {f.name for f in schema.features if f.kind == "numeric"}
    if schema.target.task == "regression":
        numeric |= set(schema.target.columns)
    casts = [pl.col(c).cast(pl.Float32) for c in sorted(numeric & names)]
    return lf.with_columns(casts).collect()


FIT_BATCH_ROWS = int(os.environ.get("FINANCE_FIT_BATCH_ROWS", "1000000"))
"""Rows per chunk when a fit streams its file: a million rows of 99 float32 features is 400 MB."""


def complete_schema(path: Path, schema: Schema) -> Schema:
    """The declared schema completed from the file in one aggregate pass: the sorted distinct
    levels of each categorical feature and, for classification, the target's classes. What
    ``prepare(..., fitting=True)`` learns from a whole frame, without holding the frame."""
    cats = [f.name for f in schema.features if f.kind != "numeric"]
    tcols = list(schema.target.columns) if schema.target.task != "regression" else []
    names = set(pl.scan_parquet(path).collect_schema().names())
    want = list(schema.key) + list(schema.feature_names) + list(schema.target.columns)
    missing = [c for c in want if c not in names]
    if missing:
        raise ValueError(f"query did not yield the columns the schema needs: {missing}")
    levels: dict[str, list[str]] = {}
    if cats or tcols:
        con = duckdb.connect()
        try:
            agg = ", ".join(
                f"list(distinct {_q(c)}::varchar order by {_q(c)}::varchar) "
                f"filter (where {_q(c)} is not null)"
                for c in cats + tcols
            )
            row = con.execute(f"select {agg} from read_parquet(?)", [str(path)]).fetchone()
            got: tuple[Any, ...] = tuple(row) if row is not None else ()
            levels = {
                c: [str(v) for v in (vals or [])] for c, vals in zip(cats + tcols, got, strict=True)
            }
        finally:
            con.close()
    feats = tuple(
        f if f.kind == "numeric" else Feature(f.name, f.kind, tuple(levels[f.name]))
        for f in schema.features
    )
    target = schema.target
    if tcols:
        seen = sorted({v for c in tcols for v in levels[c]})
        if target.task == "binary" and len(seen) != 2:
            raise ValueError(f"a binary target has exactly two classes, saw {seen}")
        target = Target(target.columns, target.task, tuple(seen))
    return Schema(schema.key, feats, target, schema.version)


def _q(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def count_rows(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def iter_batches(path: Path, schema: Schema, batch_rows: int = FIT_BATCH_ROWS) -> Any:
    """(X, Y) float32 chunks of a rows file encoded against a completed schema, in file order; the
    schema's levels and classes are final, so a chunk never learns anything. ``Prepared`` per chunk
    is built the same way as for a whole frame.

    Each chunk is a polars ``slice`` of the scan, which reads only the row groups it covers and
    frees them: ten chunks of a 20-million-row file held 1.9 GB throughout. pyarrow's batch
    iterator over the same file retained its buffers and grew to 6.7 GB by the tenth chunk."""
    numeric = {f.name for f in schema.features if f.kind == "numeric"}
    if schema.target.task == "regression":
        numeric |= set(schema.target.columns)
    cols = list(schema.key) + list(schema.feature_names) + list(schema.target.columns)
    total = count_rows(path)
    offset = 0
    while offset < total:
        df = pl.scan_parquet(path).select(cols).slice(offset, batch_rows).collect()
        offset += batch_rows
        if df.height == 0:
            break
        casts = [pl.col(c).cast(pl.Float32) for c in sorted(numeric & set(df.columns))]
        p = prepare(df.with_columns(casts), schema, fitting=False)
        if p.Y is None:
            raise ValueError("fitting needs the target columns")
        if schema.target.task != "regression" and (p.Y < 0).any():
            raise ValueError("target has null labels")
        yield p.X, p.Y


def frame_hash(df: pl.DataFrame, key: tuple[str, ...]) -> str:
    """A stable hash of a frame's content in key order: the record of what a fit read."""
    ordered = df.sort(list(key)) if key else df
    h = hashlib.sha256()
    h.update(",".join(f"{c}:{ordered.schema[c]}" for c in ordered.columns).encode())
    h.update(ordered.hash_rows().to_numpy().tobytes())
    return h.hexdigest()


# ---- coercion and encoding -------------------------------------------------------------------------


@dataclass
class Prepared:
    """What an estimator sees. X is float32 (n, features), categoricals as integer codes with NaN for
    null and unknown; Y is (n, targets) float32 for regression, class codes for classification."""

    X: np.ndarray
    Y: np.ndarray | None
    keys: pl.DataFrame
    truth: pl.DataFrame | None
    schema: Schema
    unknown_levels: dict[str, int] = field(default_factory=dict[str, int])
    extra: pl.DataFrame | None = None
    """Columns the query yielded that are neither key, feature nor target: carried into the
    inference table for the metrics (the forecast's raw return and its scale, for instance)."""


def check_columns(df: pl.DataFrame, schema: Schema, need_target: bool) -> None:
    want = list(schema.key) + list(schema.feature_names)
    if need_target:
        want += list(schema.target.columns)
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise ValueError(f"query did not yield the columns the schema needs: {missing}")


def prepare(df: pl.DataFrame, schema: Schema, fitting: bool) -> Prepared:
    """Coerce and encode a frame to a schema. Fitting completes the schema with the levels and classes
    seen; inference uses the schema's levels and counts what it had not seen."""
    has_target = all(c in df.columns for c in schema.target.columns)
    check_columns(df, schema, need_target=fitting)
    if not has_target and fitting:
        raise ValueError("fitting needs the target columns")
    df = df.sort(list(schema.key))
    cols: list[np.ndarray] = []
    feats: list[Feature] = []
    unknown: dict[str, int] = {}
    for f in schema.features:
        s = df[f.name]
        if f.kind == "numeric":
            try:
                arr = s.cast(pl.Float32, strict=True).to_numpy()
            except Exception as exc:
                raise ValueError(f"feature {f.name!r} is not numeric: {exc}") from None
            feats.append(f)
        else:
            text = s.cast(pl.String)
            if fitting:
                levels = tuple(sorted(text.drop_nulls().unique().to_list()))
            else:
                if f.levels is None:
                    raise ValueError(f"feature {f.name!r}: schema has no levels; was it fitted?")
                levels = f.levels
            index = {v: i for i, v in enumerate(levels)}
            vals = text.to_list()
            codes = np.array(
                [index.get(v, -1) if v is not None else -1 for v in vals], dtype="float32"
            )
            if not fitting:
                unknown[f.name] = int(sum(1 for v in vals if v is not None and v not in index))
            codes[codes < 0] = np.nan
            arr = codes
            feats.append(Feature(f.name, f.kind, levels))
        cols.append(arr)
    X = np.column_stack(cols) if cols else np.empty((df.height, 0), dtype="float32")
    target = schema.target
    y_mat: np.ndarray | None = None
    truth: pl.DataFrame | None = None
    if has_target:
        truth = df.select(list(target.columns))
        if target.task == "regression":
            y_mat = np.column_stack(
                [truth[c].cast(pl.Float32, strict=True).to_numpy() for c in target.columns]
            )
        else:
            labels = truth.select([pl.col(c).cast(pl.String) for c in target.columns])
            if fitting:
                seen = sorted(
                    {v for c in target.columns for v in labels[c].drop_nulls().unique().to_list()}
                )
                if target.task == "binary" and len(seen) != 2:
                    raise ValueError(f"a binary target has exactly two classes, saw {seen}")
                target = Target(target.columns, target.task, tuple(seen))
            classes = target.classes or ()
            index = {v: i for i, v in enumerate(classes)}
            y_mat = np.column_stack(
                [
                    np.array(
                        [index.get(v, -1) if v is not None else -1 for v in labels[c].to_list()],
                        dtype="float32",
                    )
                    for c in target.columns
                ]
            )
            if fitting and (y_mat < 0).any():
                raise ValueError("target has null labels")
    out = Schema(schema.key, tuple(feats), target, schema.version)
    used = set(schema.key) | set(schema.feature_names) | set(schema.target.columns)
    rest = [c for c in df.columns if c not in used]
    extra = df.select(rest) if rest else None
    return Prepared(X, y_mat, df.select(list(schema.key)), truth, out, unknown, extra)
