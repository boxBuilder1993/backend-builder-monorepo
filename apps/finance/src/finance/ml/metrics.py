"""Named metrics over a prediction table: truth columns, pred_<col> columns, params. Generic ones
and the finance ones. An Evaluation names metrics by these keys."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

Metric = Callable[[np.ndarray, np.ndarray, dict[str, Any]], float]
REGISTRY: dict[str, Metric] = {}


def metric(name: str) -> Callable[[Metric], Metric]:
    def deco(fn: Metric) -> Metric:
        REGISTRY[name] = fn
        return fn

    return deco


@metric("mae")
def mae(y: np.ndarray, p: np.ndarray, _: dict[str, Any]) -> float:
    return float(np.mean(np.abs(y - p)))


@metric("rmse")
def rmse(y: np.ndarray, p: np.ndarray, _: dict[str, Any]) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


@metric("r2")
def r2(y: np.ndarray, p: np.ndarray, _: dict[str, Any]) -> float:
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss if ss else 0.0


@metric("hit_rate")
def hit_rate(y: np.ndarray, p: np.ndarray, params: dict[str, Any]) -> float:
    """Share of rows the model picked (prediction above threshold) whose truth was above it too."""
    t = float(params.get("threshold", 0.0))
    picked = p > t
    return float(np.mean(y[picked] > t)) if picked.any() else 0.0


@metric("net_return_per_trade")
def net_return_per_trade(y: np.ndarray, p: np.ndarray, params: dict[str, Any]) -> float:
    """Mean truth over the rows the model picked, minus round-trip costs in basis points. Truth is a
    return net of nothing; costs come from params: cost_bps (default 25) and, for a flat fee per
    trade,    fee and principal."""
    t = float(params.get("threshold", 0.0))
    picked = p > t
    if not picked.any():
        return 0.0
    cost = float(params.get("cost_bps", 25.0)) / 1e4
    fee, principal = float(params.get("fee", 0.0)), float(params.get("principal", 0.0))
    if fee and principal:
        cost += fee / principal
    return float(np.mean(y[picked])) - cost


@metric("trades")
def trades(y: np.ndarray, p: np.ndarray, params: dict[str, Any]) -> float:
    return float(np.sum(p > float(params.get("threshold", 0.0))))


@metric("accuracy")
def accuracy(y: np.ndarray, p: np.ndarray, _: dict[str, Any]) -> float:
    return float(np.mean(y == p))


# ---- table metrics: they need more than the two columns ------------------------------------------

TableMetric = Callable[[pl.DataFrame, str, dict[str, Any]], float]
"""(table, target column, params) -> value. The table has the key, the truth, pred_<col> and the
query's extra columns, so a metric can group by the decision point or read a raw return."""
TABLE_REGISTRY: dict[str, TableMetric] = {}


def table_metric(name: str) -> Callable[[TableMetric], TableMetric]:
    def deco(fn: TableMetric) -> TableMetric:
        TABLE_REGISTRY[name] = fn
        return fn

    return deco


def _per_point(table: pl.DataFrame, col: str, params: dict[str, Any]) -> pl.Series:
    """Spearman correlation of truth and prediction within each decision point (params: group,
    default ts; min_cross_section, default 20 rows for a point to count)."""
    group = str(params.get("group", "ts"))
    least = int(params.get("min_cross_section", 20))
    if group not in table.columns:
        raise ValueError(f"ic needs the column {group!r} in the table")
    per = (
        table.select([pl.col(group), pl.col(col), pl.col(f"pred_{col}")])
        .drop_nulls()
        .group_by(group)
        .agg(
            pl.corr(pl.col(col).rank(), pl.col(f"pred_{col}").rank()).alias("c"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= least)
        .drop_nulls("c")
    )
    return per["c"]


@table_metric("ic")
def ic(table: pl.DataFrame, col: str, params: dict[str, Any]) -> float:
    """Mean over decision points of the within-point Spearman correlation."""
    c = _per_point(table, col, params)
    return float(c.to_numpy().mean()) if c.len() else 0.0


@table_metric("ic_se")
def ic_se(table: pl.DataFrame, col: str, params: dict[str, Any]) -> float:
    c = _per_point(table, col, params)
    arr = c.to_numpy().astype("float64")
    return float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


@table_metric("ic_points")
def ic_points(table: pl.DataFrame, col: str, params: dict[str, Any]) -> float:
    return float(_per_point(table, col, params).len())


def _pair(table: pl.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    pair = table.select([pl.col(col), pl.col(f"pred_{col}")]).drop_nulls()
    return pair[col].to_numpy().astype("float64"), pair[f"pred_{col}"].to_numpy().astype("float64")


@table_metric("calibration_slope")
def calibration_slope(table: pl.DataFrame, col: str, _: dict[str, Any]) -> float:
    """Slope of truth on prediction: 1 means the predicted magnitudes are right, below 1 they are
    too bold, above 1 too timid. The consumer rescales by it when it is off."""
    y, p = _pair(table, col)
    v = float(np.var(p))
    return float(np.cov(p, y, bias=True)[0, 1] / v) if v > 0 else 0.0


@table_metric("calibration_monotone")
def calibration_monotone(table: pl.DataFrame, col: str, params: dict[str, Any]) -> float:
    """Share of adjacent prediction deciles whose realised mean rises: 1 is monotone."""
    y, p = _pair(table, col)
    bins = int(params.get("bins", 10))
    if len(p) < bins * 2:
        return 0.0
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)[1:-1])
    which = np.searchsorted(edges, p, side="right")
    means = np.array([y[which == b].mean() for b in range(bins) if (which == b).any()])
    steps = np.diff(means)
    return float(np.mean(steps > 0)) if len(steps) else 0.0


def _raw(table: pl.DataFrame, col: str, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The raw return and the predicted raw move: pred_<col> times the scale column (params:
    return_column, default r; scale_column, default scale), times ``rescale`` (default 1; the
    calibration slope of a validation, so a tree model's exaggerated magnitudes are brought to size
    before any threshold). For a target that is already a raw return, set scale_column to null."""
    rc = str(params.get("return_column", "r"))
    sc = params.get("scale_column", "scale")
    need = [rc, f"pred_{col}"] + ([str(sc)] if sc else [])
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise ValueError(f"raw metrics need the columns {missing} in the table")
    sub = table.select([pl.col(c) for c in need]).drop_nulls()
    r = sub[rc].to_numpy().astype("float64")
    p = sub[f"pred_{col}"].to_numpy().astype("float64")
    if sc:
        p = p * sub[str(sc)].to_numpy().astype("float64")
    return r, p * float(params.get("rescale", 1.0))


@table_metric("net_return_raw")
def net_return_raw(table: pl.DataFrame, col: str, params: dict[str, Any]) -> float:
    """Mean raw return over the rows whose predicted raw move exceeds the threshold (params:
    threshold, default 0.001), minus cost_bps (default 10) as a fraction."""
    r, p = _raw(table, col, params)
    picked = p > float(params.get("threshold", 0.001))
    if not picked.any():
        return 0.0
    return float(np.mean(r[picked])) - float(params.get("cost_bps", 10.0)) / 1e4


@table_metric("trades_raw")
def trades_raw(table: pl.DataFrame, col: str, params: dict[str, Any]) -> float:
    _, p = _raw(table, col, params)
    return float(np.sum(p > float(params.get("threshold", 0.001))))


def _topk(table: pl.DataFrame, col: str, params: dict[str, Any]) -> pl.DataFrame:
    """The rows a cross-sectional consumer would take: the ``top_k`` (default 5) highest predictions
    within each decision point (``group``, default ts), among the points with at least
    ``min_cross_section`` rows. Independent of calibration: a ranking, not a threshold."""
    group = str(params.get("group", "ts"))
    k = int(params.get("top_k", 5))
    least = int(params.get("min_cross_section", 20))
    rc = str(params.get("return_column", "r"))
    need = [group, rc, f"pred_{col}"]
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise ValueError(f"top-k metrics need the columns {missing} in the table")
    return (
        table.select(need)
        .drop_nulls()
        .with_columns(
            pl.col(f"pred_{col}").rank(method="ordinal", descending=True).over(group).alias("_rk"),
            pl.len().over(group).alias("_n"),
        )
        .filter((pl.col("_n") >= least) & (pl.col("_rk") <= k))
    )


@table_metric("net_return_topk")
def net_return_topk(table: pl.DataFrame, col: str, params: dict[str, Any]) -> float:
    """Mean raw return of the top-k rows per decision point, minus cost_bps (default 10)."""
    picked = _topk(table, col, params)
    if picked.is_empty():
        return 0.0
    rc = str(params.get("return_column", "r"))
    return (
        float(picked[rc].to_numpy().astype("float64").mean())
        - float(params.get("cost_bps", 10.0)) / 1e4
    )


@table_metric("trades_topk")
def trades_topk(table: pl.DataFrame, col: str, params: dict[str, Any]) -> float:
    return float(_topk(table, col, params).height)


@table_metric("spread_topk")
def spread_topk(table: pl.DataFrame, col: str, params: dict[str, Any]) -> float:
    """Top-k mean raw return minus bottom-k mean raw return per decision point, averaged: the gross
    long-short edge of the ranking before costs."""
    rc = str(params.get("return_column", "r"))
    top = _topk(table, col, params)
    bottom = _topk(table.with_columns((-pl.col(f"pred_{col}")).alias(f"pred_{col}")), col, params)
    if top.is_empty() or bottom.is_empty():
        return 0.0
    return float(
        top[rc].to_numpy().astype("float64").mean() - bottom[rc].to_numpy().astype("float64").mean()
    )


def _groups(table: pl.DataFrame, by: str) -> list[tuple[str, pl.DataFrame]]:
    """``month`` groups by the month of ts; any other value is a column of the table."""
    if by == "month":
        if "ts" not in table.columns:
            raise ValueError("by=month needs a ts column")
        table = table.with_columns(pl.col("ts").dt.strftime("%Y-%m").alias("__month"))
        by = "__month"
    elif by not in table.columns:
        raise ValueError(f"by={by!r} is not a column of the table")
    out: list[tuple[str, pl.DataFrame]] = []
    for key, sub in sorted(table.partition_by(by, as_dict=True).items()):
        out.append((str(key[0]), sub.drop(by) if by == "__month" else sub))
    return out


def evaluate(
    table: pl.DataFrame, target_columns: list[str], names: list[str], params: dict[str, Any]
) -> dict[str, float]:
    """{"<metric>.<column>": value} for every named metric and target column in the table; with
    params["by"] (``month`` or a column), also "<metric>.<column>@<group>" per group."""
    unknown = [n for n in names if n not in REGISTRY and n not in TABLE_REGISTRY]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; known: {sorted(REGISTRY | TABLE_REGISTRY)}")
    out: dict[str, float] = {}
    by = params.get("by")
    parts: list[tuple[str | None, pl.DataFrame]] = [(None, table)]
    if by:
        parts += _groups(table, str(by))
    for label, part in parts:
        suffix = f"@{label}" if label is not None else ""
        for col in target_columns:
            if col not in part.columns or f"pred_{col}" not in part.columns:
                raise ValueError(
                    f"table lacks {col!r} or pred_{col!r}; was the target in the query?"
                )
            pair = part.select([pl.col(col), pl.col(f"pred_{col}")]).drop_nulls()
            y = pair[col].to_numpy()
            p = pair[f"pred_{col}"].to_numpy()
            if y.dtype.kind not in "fiub":
                y = y.astype(str)
                p = p.astype(str)
            else:
                y = y.astype("float64")
                p = p.astype("float64")
            for n in names:
                if n in REGISTRY:
                    out[f"{n}.{col}{suffix}"] = REGISTRY[n](y, p, params)
                else:
                    out[f"{n}.{col}{suffix}"] = TABLE_REGISTRY[n](part, col, params)
    return out
