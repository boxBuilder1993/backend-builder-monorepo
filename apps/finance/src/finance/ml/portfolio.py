"""Prices and model-free scores for the backtest, from the mirror through DuckDB.

Prices are NSE EQ-series daily rows from ``equity_adj``: unadjusted open and close (what a trade
pays), the cumulative adjustment factor (splits and bonuses scale held shares), and the 20-session
median rupee turnover before each day (liquidity and the position cap). Baseline scores are computed
on adjusted closes so a split is not a crash. See docs/finance-portfolio.md §4 and §6.
"""

# ruff: noqa: E501  (the SQL reads better unwrapped)

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from finance.ml import data
from finance.ml.backtest import Market, Rules
from finance.ml.forecast import DURATIONS

LIQUID_TURNOVER = 1e7
"""₹1 crore: a stock is in the universe on a day when its 20-session median turnover is at least this."""
HISTORY_DAYS = 400
"""Calendar days of prices loaded before a period starts, so 250-session features are complete."""
ETF = "NIFTYBEES"

PRICES_SQL = """
with e as (
    select date, symbol, open, close, adj_factor, adj_close, turnover
    from equity_adj
    where series = 'EQ' and date between :start_data and :end and open > 0 and close > 0
    qualify row_number() over (partition by symbol, date order by turnover desc) = 1
)
select date, symbol, open, close, adj_factor, adj_close,
       median(turnover) over (partition by symbol order by date rows between 20 preceding and 1 preceding) as med20,
       ln(adj_close / lag(adj_close, 250) over w) - ln(adj_close / lag(adj_close, 21) over w) as mom_12_1,
       stddev_samp(r) over (partition by symbol order by date rows between 119 preceding and current row) as vol_120d,
       count(*) over (partition by symbol order by date rows between 249 preceding and current row) as hist
from (select *, ln(adj_close / lag(adj_close) over (partition by symbol order by date)) as r from e)
window w as (partition by symbol order by date)
order by date, symbol
"""

BASELINES = ("momentum_12_1", "low_vol", "equal_weight", "niftybees")


def load_prices(root: Any, start: date, end: date) -> pl.DataFrame:
    """Every EQ row from ``HISTORY_DAYS`` before ``start`` to ``end``, with the baseline columns."""
    con = data.connect(root)
    try:
        data.tune(con)
        start_data = date.fromordinal(start.toordinal() - HISTORY_DAYS)
        return data.run_query(
            con, PRICES_SQL, {"start_data": start_data.isoformat(), "end": end.isoformat()}
        )
    finally:
        con.close()


def universe(prices: pl.DataFrame) -> pl.DataFrame:
    """The rows of stocks that are ever liquid, plus the ETF benchmark: a smaller market to simulate."""
    ever = prices.filter(pl.col("med20") >= LIQUID_TURNOVER)["symbol"].unique().to_list()
    return prices.filter(pl.col("symbol").is_in([*ever, ETF]))


def baseline_scores(
    name: str, prices: pl.DataFrame, market: Market, draws: int = 1, seed: int = 0
) -> list[np.ndarray]:
    """Score arrays for a model-free baseline: one array, or ``draws`` arrays for the random one.
    A stock outside the liquid universe on a day has no score that day."""
    liquid = prices.filter((pl.col("med20") >= LIQUID_TURNOVER) & (pl.col("symbol") != ETF))
    if name == "momentum_12_1":
        f = liquid.filter(pl.col("hist") >= 250).select(
            "date", "symbol", pl.col("mom_12_1").alias("score")
        )
        return [market.scores(f.drop_nulls())]
    if name == "low_vol":
        f = liquid.filter(pl.col("hist") >= 120).select(
            "date", "symbol", (-pl.col("vol_120d")).alias("score")
        )
        return [market.scores(f.drop_nulls())]
    if name == "equal_weight":
        base = market.scores(liquid.select("date", "symbol", pl.lit(1.0).alias("score")))
        rng = np.random.default_rng(seed)
        return [np.where(np.isnan(base), np.nan, rng.random(base.shape)) for _ in range(draws)]
    if name == "niftybees":
        f = prices.filter(pl.col("symbol") == ETF).select(
            "date", "symbol", pl.lit(1.0).alias("score")
        )
        return [market.scores(f)]
    raise ValueError(f"unknown baseline {name!r}; known: {BASELINES}")


def nearest_duration(rebalance: int, durations: tuple[int, ...] = DURATIONS) -> int:
    """The hold duration the model was asked about that is closest to the rebalance interval; the
    shorter one on a tie."""
    return min(durations, key=lambda h: (abs(h - rebalance), h))


def walkforward_scores(
    root: Any, predictions_path: str, market: Market, rebalance: int
) -> np.ndarray:
    """A walk-forward's forecasts as a score array: the forecast at the duration nearest the
    rebalance interval, since an account that decides every ``rebalance`` sessions holds for about
    that long. Rows are keyed by the decision day; the first ``pred_*`` column is the score."""
    h = nearest_duration(rebalance)
    lf = pl.scan_parquet(Path(root) / predictions_path).filter(pl.col("h") == h)
    cols = lf.collect_schema().names()
    pred = next(c for c in cols if c.startswith("pred_"))
    day = pl.col("day") if "day" in cols else pl.col("ts").dt.date()
    f = lf.select(day.alias("date"), "symbol", pl.col(pred).alias("score")).collect()
    return market.scores(f.drop_nulls())


def source_name(source: dict[str, Any]) -> str:
    if "walk_forward_id" in source:
        return f"walk_forward {str(source['walk_forward_id'])[:8]}"
    return str(source["baseline"])


def source_scores(
    root: Any,
    source: dict[str, Any],
    prices: pl.DataFrame,
    market: Market,
    rules: Rules,
    draws: int = 1,
    seed: int = 0,
) -> list[np.ndarray]:
    """Score arrays for a grid source: a walk-forward's forecasts (``{"walk_forward_id",
    "predictions_path"}``, one array per rebalance interval) or a baseline."""
    if "walk_forward_id" in source:
        return [walkforward_scores(root, str(source["predictions_path"]), market, rules.rebalance)]
    return baseline_scores(str(source["baseline"]), prices, market, draws=draws, seed=seed)


def niftybees_rules_override() -> dict[str, Any]:
    """NIFTYBEES is bought once with everything and never sold: one stock, no rebalance."""
    return {"top_n": 1, "rebalance": 10**6, "exit": "band", "band": 1.0, "min_hold": 0}


def index_returns(
    root: Any, start: date, end: date, names: tuple[str, ...] = ("Nifty 50", "Nifty 500")
) -> dict[str, float]:
    """Price-index return over the period, in percent, from the first close on or after ``start``
    to the last close on or before ``end``; no costs, no dividends."""
    con: duckdb.DuckDBPyConnection = data.connect(root)
    try:
        out: dict[str, float] = {}
        for n in names:
            row = con.execute(
                "select first(close order by date), last(close order by date) from index_daily "
                "where index_name = ? and date between ? and ?",
                [n, start, end],
            ).fetchone()
            if row and row[0] and row[1]:
                out[n] = round((row[1] / row[0] - 1) * 100, 2)
        return out
    finally:
        con.close()
