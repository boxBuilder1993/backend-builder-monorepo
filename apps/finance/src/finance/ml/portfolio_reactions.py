"""The backtest as reactions: a grid of rule combinations per source and capital, one account in
full, and the summary of a labelled set of grids. See docs/finance-portfolio.md §4 and §6.

``RunBacktestGrid`` returns in seconds and defers one ``BacktestGridSource`` per (source × capital),
so no parent run holds a transaction while hundreds of accounts are simulated (issue #69). Each
child loads prices once, runs every combination in-process, writes one Parquet row per combination
into the mirror and publishes it.
"""

from __future__ import annotations

import asyncio
import uuid
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
from finance.ml.entities import BacktestGrid, BacktestRun, WalkForward
from finance.ml.reactions import PublishArtifact
from finance.reactions import PullStore

GRID_BUDGET = 3 * 3600
DEFAULT_COSTS = C.DELIVERY.name


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _day(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


async def resolve_source(ctx: Context, source: dict[str, Any]) -> dict[str, Any]:
    """A grid or run source checked and completed: a walk-forward must have closed with its
    predictions file, whose path is copied in so the simulation reads the mirror without the row;
    a baseline must be a known one."""
    if "walk_forward_id" in source:
        wf = await ctx.get_by_id(WalkForward, str(source["walk_forward_id"]))
        if wf is None or not wf.predictions_path:
            raise ValueError(f"walk-forward {source['walk_forward_id']} has no predictions yet")
        return {
            "walk_forward_id": wf.id,
            "predictions_path": wf.predictions_path,
            "label": wf.label,
        }
    if source.get("baseline") not in P.BASELINES:
        raise ValueError(f"unknown source {source!r}; baselines: {P.BASELINES}")
    return source


class RunBacktestGrid(Reaction):
    """Input: {"label", "sources": [{"baseline": ...} | {"walk_forward_id": ...}], "capitals",
    "start", "end", "grid": {rule: [values]}, "costs"?, "draws"?: random draws for equal_weight
    (default 20)}. One BacktestGrid row per (source × capital), each run by a deferred
    BacktestGridSource. A walk-forward source must have closed with a predictions file; its path
    is copied into the grid's source so the child reads the mirror without the row."""

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
        for raw in ctx.input["sources"]:
            source = await resolve_source(ctx, dict(raw))
            name = source.get("baseline")
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
    name = P.source_name(g.source)
    grid = B.rules_grid(dict(g.grid))
    if name == "niftybees":
        base = grid[0]
        grid = [B.Rules(**{**asdict(base), **P.niftybees_rules_override()})]
    i0, i1 = market.index(start), market.index(end, after=True) - 1
    # a model is asked for the hold nearest each rebalance interval: one score array per interval
    groups: dict[int, list[B.Rules]] = {}
    for r in grid:
        groups.setdefault(r.rebalance if "walk_forward_id" in g.source else 0, []).append(r)
    parts: list[pl.DataFrame] = []
    for rules in groups.values():
        draws = P.source_scores(root, g.source, prices, market, rules[0], g.draws, 0)
        parts.append(B.run_grid(market, draws, rules, g.capital, C.SCHEDULES[g.costs], i0, i1))
    table = pl.concat(parts, how="diagonal_relaxed").with_columns(
        pl.lit(g.id).alias("grid_id"),
        pl.lit(name).alias("source"),
        pl.lit(str(g.source.get("walk_forward_id") or "")).alias("walk_forward_id"),
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
        source = await resolve_source(ctx, dict(ctx.input["source"]))
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
        ctx.save("run_id", run.id)
        ctx.save("equity_path", run.equity_path)


def _account(root: Path, run: BacktestRun, rules: B.Rules, stem: str) -> dict[str, Any]:
    start, end = _day(run.start_date), _day(run.end_date)
    prices = P.universe(P.load_prices(root, start, end))
    market = B.Market.from_frame(prices, rules.max_turnover_share)
    name = P.source_name(run.source)
    scores = P.source_scores(root, run.source, prices, market, rules, 1, 0)[0]
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


RULE_COLUMNS = list(asdict(B.Rules(top_n=1, rebalance=1)))
SELECT_COLUMNS = [
    "rank", "grid_id", "source", "walk_forward_id", *RULE_COLUMNS, "return_pct", "return_pct_ew",
    "excess_pct", "sharpe", "max_drawdown_pct", "charges_pct_per_year", "trades", "avg_hold_days",
    "ruined_share",
]  # fmt: skip


class SelectPortfolio(Reaction):
    """Input: {"label", "grid_ids": [...], "top"?: 20}. The grids of one selection period and
    capital: the model sources and the one equal_weight grid among them. Every configuration's
    return less the equal-weight return under the same rules (the matching baseline) ranks the
    candidates; the best and the top N go to vars and the full ranking to
    parquet/backtests/select_<id>.parquet, published. Then the best and its baseline run as full
    accounts over the same period for the t-statistic of the monthly excess return. Everything here
    is decided on the selection period alone; the confirmation on the held-out year is a separate
    RunBacktest with the rules saved here, unchanged."""

    async def calculate(self, ctx: Context) -> None:
        label = str(ctx.input["label"])
        top_n = int(ctx.input.get("top", 20))
        grids: list[BacktestGrid] = []
        for gid in ctx.input["grid_ids"]:
            g = await ctx.get_by_id(BacktestGrid, str(gid))
            if g is None or g.status != "done" or not g.results_path:
                raise ValueError(f"grid {gid} is not done")
            grids.append(g)
        periods = {(g.capital, g.start_date, g.end_date, g.costs) for g in grids}
        if len(periods) != 1:
            raise ValueError(f"the grids must share capital, period and costs; got {periods}")
        ew = [g for g in grids if g.source.get("baseline") == "equal_weight"]
        if len(ew) != 1:
            raise ValueError("exactly one equal_weight grid is the matching baseline")
        candidates = [g for g in grids if g is not ew[0]]
        root = poller.data_root()
        sel_id = uuid.uuid4().hex
        rel = f"parquet/backtests/select_{sel_id}.parquet"
        ranking = await asyncio.to_thread(_rank, root, candidates, ew[0], rel)
        await ctx.trigger(
            PublishArtifact, {"paths": [rel], "kind": "parquet", "source": "backtests"}
        )
        best = _plain(ranking.row(0, named=True))
        top = [_plain(r) for r in ranking.head(top_n).iter_rows(named=True)]
        ctx.save("ranking_path", rel)
        ctx.save("configurations", ranking.height)
        ctx.save("baseline_grid_id", ew[0].id)
        ctx.save("best", best)
        ctx.save("top", top)
        ctx.save("table_top", _md(ranking.head(top_n).select(SELECT_COLUMNS)))
        # the chosen account and its baseline in full, for the t-statistic over months
        g0 = ew[0]
        rules = {k: best[k] for k in RULE_COLUMNS}
        chosen = next(g for g in candidates if g.id == best["grid_id"])
        common = {"rules": rules, "capital": g0.capital, "costs": g0.costs,
                  "start": g0.start_date.isoformat(), "end": g0.end_date.isoformat()}  # fmt: skip
        a = await ctx.trigger(
            RunBacktest, {"label": f"{label} best", "source": chosen.source, **common}
        )
        b = await ctx.trigger(
            RunBacktest, {"label": f"{label} equal_weight", "source": g0.source, **common}
        )
        assert a is not None and b is not None
        if a.vars.get("equity_path") and b.vars.get("equity_path"):
            stat = await asyncio.to_thread(
                _monthly_excess, root, str(a.vars["equity_path"]), str(b.vars["equity_path"])
            )
            ctx.save("monthly_excess", stat)
        ctx.save("best_run_id", a.vars.get("run_id"))
        ctx.save("baseline_run_id", b.vars.get("run_id"))


def _rank(root: Path, candidates: list[BacktestGrid], ew: BacktestGrid, rel: str) -> pl.DataFrame:
    base = pl.read_parquet(root / str(ew.results_path)).select(
        *RULE_COLUMNS, pl.col("return_pct").alias("return_pct_ew")
    )
    tables = [pl.read_parquet(root / str(g.results_path)) for g in candidates]
    joined = (
        pl.concat(tables, how="diagonal_relaxed")
        .join(base, on=RULE_COLUMNS, how="inner")
        .with_columns((pl.col("return_pct") - pl.col("return_pct_ew")).alias("excess_pct"))
        .sort(["excess_pct", "sharpe"], descending=[True, True], nulls_last=True)
        .with_row_index("rank", offset=1)
    )
    if joined.height == 0:
        raise ValueError("no configuration of the candidates matches a rule set of the baseline")
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(root / rel)
    return joined


def _monthly_excess(root: Path, best_path: str, ew_path: str) -> dict[str, Any]:
    """Monthly returns of the two accounts from their equity curves; the mean excess, its standard
    deviation and the t-statistic over the months."""

    def monthly(path: str) -> pl.DataFrame:
        eq = pl.read_parquet(root / path).sort("date")
        ends = eq.group_by(pl.col("date").dt.truncate("1mo").alias("month")).agg(
            pl.col("equity").last()
        )
        return ends.sort("month").with_columns(pl.col("equity").pct_change().alias("ret"))

    m = monthly(best_path).join(monthly(ew_path), on="month", suffix="_ew").drop_nulls()
    d = (m["ret"] - m["ret_ew"]).to_numpy()
    n = int(len(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    t = float(np.mean(d) / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None
    return {
        "months": n,
        "mean_excess_pct": float(np.mean(d) * 100) if n else None,
        "sd_excess_pct": sd * 100 if n > 1 else None,
        "t_stat": t,
        "months_beating_baseline": int((d > 0).sum()) if n else 0,
    }


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
