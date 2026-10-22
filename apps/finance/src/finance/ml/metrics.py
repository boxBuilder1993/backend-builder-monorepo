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


def evaluate(
    table: pl.DataFrame, target_columns: list[str], names: list[str], params: dict[str, Any]
) -> dict[str, float]:
    """{"<metric>.<column>": value} for every named metric and target column in the table."""
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; known: {sorted(REGISTRY)}")
    out: dict[str, float] = {}
    for col in target_columns:
        if col not in table.columns or f"pred_{col}" not in table.columns:
            raise ValueError(f"table lacks {col!r} or pred_{col!r}; was the target in the query?")
        pair = table.select([pl.col(col), pl.col(f"pred_{col}")]).drop_nulls()
        y = pair[col].to_numpy()
        p = pair[f"pred_{col}"].to_numpy()
        if y.dtype.kind not in "fiub":
            y = y.astype(str)
            p = p.astype(str)
        else:
            y = y.astype("float64")
            p = p.astype("float64")
        for n in names:
            out[f"{n}.{col}"] = REGISTRY[n](y, p, params)
    return out
