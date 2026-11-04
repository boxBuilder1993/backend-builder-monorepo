"""How a walk-forward window's forecasts are scored. See docs/finance-portfolio.md §5 (revision 5).

The ranking score is the Spearman IC within each decision point (the date, and the hold duration
``h`` when the rows carry one), averaged. The magnitude scores sit next to it, because a portfolio
sizes positions and compares a forecast against costs, which a rank correlation cannot inform:

- ``pearson``: the Pearson IC within each decision point, averaged;
- ``hit``: the share of rows whose forecast has the sign of the realised return;
- ``slope`` and ``r2``: realised on predicted after removing each decision point's mean (so the
  market's level, which no cross-sectional model forecasts, is left out). A slope of 1 means a stock
  forecast two points above the day's average did two points better than average; ``r2`` is against
  predicting the day's average, using the forecast's own scale, so a badly scaled forecast scores
  below zero even when it ranks well;
- ``bias``: mean forecast minus mean realised return, the level error the slope leaves out;
- ``deciles``: the rows cut into ten by forecast, each with its mean forecast, mean realised return
  and share of rows that rose. Monotone realised returns with gaps matching the forecast's gaps are
  what a calibrated model looks like.

Slope, bias and deciles are in the target's units: return points for a raw-return target, scaled
units for a symmetric one. Rows without a target (the last h sessions) are left out, as are decision
points with fewer than ``MIN_POINT`` rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

MIN_POINT = 20
"""Rows a decision point needs before its correlation counts."""

SCORE_KEYS = ("ic", "pearson", "hit", "slope", "r2", "bias")
"""The scalar scores, overall and per duration."""


def _f(v: Any, nd: int = 4) -> float | None:
    return round(float(v), nd) if isinstance(v, int | float) and v == v else None


def window_scores(path: Path, target: str, key: list[str]) -> dict[str, Any]:
    """The scores of one inference table (``target`` and ``pred_<target>`` columns): overall and,
    when the rows carry ``h``, per duration under ``by_h``."""
    df = pl.read_parquet(path)
    pred = f"pred_{target}"
    df = df.filter(pl.col(target).is_not_null() & pl.col(pred).is_not_null())
    group = [k for k in key if k != "symbol"]
    if "h" in df.columns and "h" not in group:
        group.append("h")
    out: dict[str, Any] = {**frame_scores(df, target, pred, group), "rows": df.height}
    if "h" in df.columns and df.height:
        out["by_h"] = {
            str(h): frame_scores(part, target, pred, group)
            for (h,), part in sorted(df.partition_by("h", as_dict=True).items())
        }
    return out


def frame_scores(df: pl.DataFrame, target: str, pred: str, group: list[str]) -> dict[str, Any]:
    """The scores of one frame, its decision points given by ``group``."""
    per = (
        df.group_by(group)
        .agg(
            pl.corr(pl.col(target).rank(), pl.col(pred).rank()).alias("c"),
            pl.corr(pl.col(target), pl.col(pred)).alias("p"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= MIN_POINT)
        .filter(pl.col("c").is_not_null() & pl.col("c").is_not_nan())
    )
    out: dict[str, Any] = {"ic": None, "pearson": None, "points": per.height}
    if not per.height:
        return out
    out["ic"] = _f(per["c"].mean())
    out["pearson"] = _f(per["p"].mean())
    kept = df.join(per.select(group), on=group, how="semi")
    y = pl.col(target)
    p = pl.col(pred)
    dm = kept.with_columns(
        (y - y.mean().over(group)).alias("_y"), (p - p.mean().over(group)).alias("_p")
    )
    py = dm["_p"].cast(pl.Float64).to_numpy()
    yy = dm["_y"].cast(pl.Float64).to_numpy()
    ss_p = float((py * py).sum())
    ss_y = float((yy * yy).sum())
    slope = float((py * yy).sum()) / ss_p if ss_p > 0 else None
    r2 = 1 - float(((yy - py) ** 2).sum()) / ss_y if ss_y > 0 else None
    bias = float(kept[pred].cast(pl.Float64).to_numpy().mean())
    bias -= float(kept[target].cast(pl.Float64).to_numpy().mean())
    out.update(
        hit=_f(((kept[target] > 0) == (kept[pred] > 0)).mean()),
        slope=_f(slope),
        r2=_f(r2),
        bias=_f(bias),
        deciles=_deciles(kept, target, pred),
    )
    return out


def _deciles(df: pl.DataFrame, target: str, pred: str) -> list[dict[str, Any]]:
    """Ten bins by forecast, lowest first: mean forecast, mean realised return, share that rose."""
    if df.height < 10 * MIN_POINT:
        return []
    binned = df.with_columns(
        pl.col(pred).rank(method="ordinal").mul(10).sub(1).floordiv(df.height).alias("_d")
    )
    table = (
        binned.group_by("_d")
        .agg(
            pl.col(pred).mean().alias("pred"),
            pl.col(target).mean().alias("real"),
            (pl.col(target) > 0).mean().alias("hit"),
            pl.len().alias("n"),
        )
        .sort("_d")
    )
    return [
        {"pred": _f(r["pred"]), "real": _f(r["real"]), "hit": _f(r["hit"], 3), "n": int(r["n"])}
        for r in table.iter_rows(named=True)
    ]


def mean_scores(scores: list[dict[str, Any]]) -> dict[str, float | None]:
    """The scalar scores averaged over windows, those a window lacks left out."""
    out: dict[str, float | None] = {}
    for k in SCORE_KEYS:
        xs = [s[k] for s in scores if isinstance(s.get(k), int | float)]
        out[k] = round(sum(xs) / len(xs), 4) if xs else None
    return out
