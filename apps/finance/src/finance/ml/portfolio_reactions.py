"""The backtest as reactions: a grid of rule combinations per source and capital, one account in
full, and the summary of a labelled set of grids. See docs/finance-portfolio.md §4 and §6.

``RunBacktestGrid`` returns in seconds and defers one ``BacktestGridSource`` per (source × capital),
so no parent run holds a transaction while hundreds of accounts are simulated (issue #69). Each
child loads prices once, runs every combination in-process, writes one Parquet row per combination
into the mirror and publishes it.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from backend_builder import Context, Reaction

from finance import reactions as poller
from finance.ml import backtest as B
from finance.ml import costs as C
from finance.ml import portfolio as P
from finance.ml.entities import BacktestGrid, BacktestRun
from finance.ml.reactions import PublishArtifact, PullStore

GRID_BUDGET = 3 * 3600
DEFAULT_COSTS = C.DELIVERY.name


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _day(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


class RunBacktestGrid(Reaction):
    """Input: {"label", "sources": [{"baseline": ...}], "capitals": [...], "start", "end",
    "grid": {rule: [values]}, "costs"?, "draws"?: random draws for equal_weight (default 20)}.
    One BacktestGrid row per (source × capital), each run by a deferred BacktestGridSource."""

    async def calculate(self, ctx: Context) -> None:
        costs = str(ctx.input.get("costs", DEFAULT_COSTS))
        if costs not in C.SCHEDULES:
            raise ValueError(f"unknown cost schedule {costs!r}")
        grid = dict(ctx.input["grid"])
        # an unknown rule name or bad value fails here, before anything is queued
        B.rules_grid(grid)
        # the mirror is brought to the store's revision once, here; the children only read it, so
        # concurrent children never write the same StoreSync row
        pull = await ctx.trigger(PullStore, {})
        assert pull is not None
        if pull.vars.get("status") == "failed":
            raise RuntimeError(f"store unavailable: {pull.vars.get('error')}")
        ids: list[str] = []
        for source in ctx.input["sources"]:
            name = source.get("baseline")
            if name not in P.BASELINES:
                raise ValueError(f"unknown source {source!r}; baselines: {P.BASELINES}")
            for capital in ctx.input["capitals"]:
                g = ctx.create(
                    BacktestGrid,
                    label=str(ctx.input["label"]),
                    source=dict(source),
                    capital=float(capital),
                    costs=costs,
                    start_date=_day(ctx.input["start"]),
                    end_date=_day(ctx.input["end"]),
                    grid=grid,
                    draws=int(ctx.input.get("draws", 20)) if name == "equal_weight" else 1,
                )
                await ctx.trigger(
                    BacktestGridSource, {"grid_id": g.id}, deferred=True, timeout=GRID_BUDGET
                )
                ids.append(g.id)
        ctx.save("grids", ids)


class BacktestGridSource(Reaction):
    """Deferred, one per BacktestGrid. Input: {"grid_id"}. Every combination, one results file.
    Reads the mirror as the parent left it (the parent ran PullStore)."""

    async def calculate(self, ctx: Context) -> None:
        g = await ctx.get_by_id(BacktestGrid, ctx.input["grid_id"])
        if g is None:
            raise LookupError(f"no BacktestGrid {ctx.input['grid_id']}")
        try:
            rel = f"parquet/backtests/grid_{g.id}.parquet"
            out = poller.data_root() / rel
            table = await asyncio.to_thread(_grid, poller.data_root(), g, out)
        except Exception as exc:  # noqa: BLE001 - the grid's outcome
            g.status, g.error = "failed", _err(exc)
            ctx.save("error", g.error)
            return
        await ctx.trigger(
            PublishArtifact, {"paths": [rel], "kind": "parquet", "source": "backtests"}
        )
        best = table.sort("end_value", descending=True).row(0, named=True)
        g.results_path, g.combinations, g.best, g.status = rel, table.height, _plain(best), "done"
        ctx.save("combinations", table.height)
        ctx.save("best", g.best)


def _grid(root: Path, g: BacktestGrid, out: Path) -> pl.DataFrame:
    start, end = _day(g.start_date), _day(g.end_date)
    prices = P.universe(P.load_prices(root, start, end))
    market = B.Market.from_frame(prices)
    name = str(g.source["baseline"])
    draws = P.baseline_scores(name, prices, market, draws=g.draws, seed=0)
    grid = B.rules_grid(dict(g.grid))
    if name == "niftybees":
        base = grid[0]
        grid = [B.Rules(**{**asdict(base), **P.niftybees_rules_override()})]
    i0, i1 = market.index(start), market.index(end, after=True) - 1
    table = B.run_grid(market, draws, grid, g.capital, C.SCHEDULES[g.costs], i0, i1)
    table = table.with_columns(
        pl.lit(g.id).alias("grid_id"),
        pl.lit(name).alias("source"),
        pl.lit(g.capital).alias("capital"),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    return table


class RunBacktest(Reaction):
    """Input: {"label", "source", "rules", "capital", "start", "end", "costs"?}. One account with
    every trade and its daily equity written to the mirror and published; the summary in the row,
    with the Nifty 50 and Nifty 500 price returns over the same dates for comparison."""

    async def calculate(self, ctx: Context) -> None:
        source = dict(ctx.input["source"])
        rules = B.Rules(**dict(ctx.input["rules"]))
        costs = str(ctx.input.get("costs", DEFAULT_COSTS))
        run = ctx.create(
            BacktestRun,
            label=str(ctx.input["label"]),
            source=source,
            rules=asdict(rules),
            costs=costs,
            capital=float(ctx.input["capital"]),
            start_date=_day(ctx.input["start"]),
            end_date=_day(ctx.input["end"]),
        )
        pull = await ctx.trigger(PullStore, {})
        assert pull is not None
        stem = f"parquet/backtests/run_{run.id}"
        try:
            summary = await asyncio.to_thread(_account, poller.data_root(), run, rules, stem)
        except Exception as exc:  # noqa: BLE001
            run.status, run.error = "failed", _err(exc)
            ctx.save("error", run.error)
            return
        paths = [f"{stem}_equity.parquet", f"{stem}_trades.parquet"]
        await ctx.trigger(
            PublishArtifact, {"paths": paths, "kind": "parquet", "source": "backtests"}
        )
        run.equity_path, run.trades_path = paths
        run.summary, run.status = summary, "done"
        ctx.save("summary", summary)


def _account(root: Path, run: BacktestRun, rules: B.Rules, stem: str) -> dict[str, Any]:
    start, end = _day(run.start_date), _day(run.end_date)
    prices = P.universe(P.load_prices(root, start, end))
    market = B.Market.from_frame(prices, rules.max_turnover_share)
    name = str(run.source["baseline"])
    scores = P.baseline_scores(name, prices, market, draws=1, seed=0)[0]
    if name == "niftybees":
        rules = B.Rules(**{**asdict(rules), **P.niftybees_rules_override()})
    i0, i1 = market.index(start), market.index(end, after=True) - 1
    res = B.simulate(
        market, scores, rules, run.capital, C.SCHEDULES[run.costs], i0, i1, record=True
    )
    eq = pl.DataFrame({"date": res.dates, "equity": res.equity, "cash": res.cash})
    trades = pl.DataFrame(res.log) if res.log else pl.DataFrame(schema={"date": pl.Date})
    (root / stem).parent.mkdir(parents=True, exist_ok=True)
    eq.write_parquet(root / f"{stem}_equity.parquet")
    trades.write_parquet(root / f"{stem}_trades.parquet")
    summary = B.summarise(res, run.capital)
    summary["index_return_pct"] = P.index_returns(root, res.dates[0], res.dates[-1])
    return _plain(summary)


class SummarizeBacktestGrids(Reaction):
    """Input: {"label"}. Reads the done BacktestGrids with the label and saves, in vars, the best
    combination per (source, capital) by end value, and for the equal-weight source the charges
    per year over (top_n × rebalance) per capital: the capital study's answer in markdown."""

    async def calculate(self, ctx: Context) -> None:
        label = str(ctx.input["label"])
        grids = await ctx.query(
            BacktestGrid,
            "select * from backtest_grid where label = :l and status = 'done'",
            l=label,
        )
        if not grids:
            raise LookupError(f"no finished BacktestGrid labelled {label!r}")
        root = poller.data_root()
        frames = [pl.read_parquet(root / g.results_path) for g in grids if g.results_path]
        table = pl.concat(frames, how="diagonal_relaxed")
        ctx.save("grids", len(grids))
        ctx.save("rows", table.height)
        ctx.save("table_best", _best_table(table))
        for capital in sorted(table["capital"].unique().to_list()):
            ew = table.filter((pl.col("source") == "equal_weight") & (pl.col("capital") == capital))
            if ew.height:
                ctx.save(f"table_costs_{int(capital)}", _cost_table(ew))


def _best_table(t: pl.DataFrame) -> str:
    best = (
        t.sort("end_value", descending=True)
        .group_by("source", "capital", maintain_order=True)
        .first()
        .sort("capital", "end_value", descending=[False, True])
    )
    cols = ["capital", "source", "top_n", "rebalance", "exit", "band", "min_hold", "end_value",
            "cagr_pct", "max_drawdown_pct", "charges_pct_per_year", "cash_drag_pct",
            "trades"]  # fmt: skip
    return _md(best.select([c for c in cols if c in best.columns]))


def _cost_table(t: pl.DataFrame) -> str:
    """Charges per year (% of capital) for the band exit, top_n down, rebalance across."""
    band = t.filter(pl.col("exit") == "band")
    if band.height == 0:
        band = t
    pivot = (
        band.group_by("top_n", "rebalance")
        .agg(pl.col("charges_pct_per_year").mean().round(2))
        .pivot(on="rebalance", index="top_n", values="charges_pct_per_year")
        .sort("top_n")
    )
    return _md(pivot)


def _md(df: pl.DataFrame) -> str:
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    lines = [head, "|" + "---|" * len(df.columns)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def _plain(v: Any) -> Any:
    """Numpy scalars and nested containers as plain Python, for a JSON column."""
    if isinstance(v, dict):
        items: dict[Any, Any] = v  # pyright: ignore[reportUnknownVariableType]
        return {str(k): _plain(x) for k, x in items.items()}
    if isinstance(v, list | tuple):
        seq: list[Any] = list(v)  # pyright: ignore[reportUnknownArgumentType]
        return [_plain(x) for x in seq]
    if isinstance(v, np.integer):
        return int(v)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(v, np.floating):
        return float(v)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(v, np.bool_):
        return bool(v)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(v, date):
        return v.isoformat()
    return v
