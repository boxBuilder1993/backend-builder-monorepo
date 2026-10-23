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
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from finance.ml.schema import Feature, Schema, Target

DEFAULT_SERIES = ("EQ", "BE")
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

    for ref in ("corp_actions", "holidays", "symbol_changes", "equity_master", "securities"):
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


def run_query(con: duckdb.DuckDBPyConnection, sql: str, params: dict[str, Any]) -> pl.DataFrame:
    """Run SQL with :name params bound by name. Every param the query uses must be given."""
    missing = [p for p in param_names(sql) if p not in params]
    if missing:
        raise ValueError(f"query uses params without values: {missing}")
    return con.sql(_PARAM.sub(r"$\1", sql), params=params).pl()


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
                arr = s.cast(pl.Float64, strict=True).to_numpy().astype("float32")
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
                [
                    truth[c].cast(pl.Float64, strict=True).to_numpy().astype("float32")
                    for c in target.columns
                ]
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
    return Prepared(X, y_mat, df.select(list(schema.key)), truth, out, unknown)
