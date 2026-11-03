# ruff: noqa: E501  (test fixtures read better on one line)

"""The cost schedules and the account simulation on synthetic prices with known answers, and the
backtest reactions on Postgres with the price loader faked. See docs/finance-portfolio.md §9."""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from backend_builder import PostgresStore, Worker, run
from finance import mirror
from finance.ml import backtest as B
from finance.ml import costs as C
from finance.ml import portfolio as P
from finance.ml.entities import BacktestGrid, BacktestRun
from finance.ml.portfolio_reactions import (
    RunBacktest,
    RunBacktestGrid,
    SelectPortfolio,
    SummarizeBacktestGrids,
)

ZERO = C.Schedule("zero", 0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_costs_match_the_published_schedule_by_hand() -> None:
    # delivery, ₹10,000 bought: STT 10 + exchange 0.307 + SEBI 0.01 + stamp 1.5 + GST 18% of
    # (0.307 + 0.01) = 0.05706 → 11.87406
    assert C.order_charges(C.DELIVERY, "buy", 10_000) == pytest.approx(11.87406, abs=1e-6)
    # delivery sale: STT 10 + exchange 0.307 + SEBI 0.01 + GST 0.05706, no stamp; DP separately
    assert C.order_charges(C.DELIVERY, "sell", 10_000) == pytest.approx(10.37406, abs=1e-6)
    assert C.dp_charge(C.DELIVERY) == pytest.approx(18.1012)
    # intraday, ₹10,000 sold: brokerage 3 (0.03% under the ₹20 cap) + STT 2.5 + exchange 0.307
    # + SEBI 0.01 + GST 18% of 3.317 = 0.59706 → 6.41406
    assert C.order_charges(C.INTRADAY, "sell", 10_000) == pytest.approx(6.41406, abs=1e-6)
    # intraday brokerage is capped at ₹20 per order: ₹1 lakh pays 20, not 30
    big = C.order_charges(C.INTRADAY, "buy", 100_000)
    assert big == pytest.approx(20 + 3.07 + 0.1 + 3.0 + 0.18 * (20 + 3.07 + 0.1), abs=1e-6)
    # futures sale of ₹5 lakh: brokerage 20, STT 0.05% = 250, exchange 9.15, SEBI 0.5
    fut = C.order_charges(C.FUTURES, "sell", 500_000)
    assert fut == pytest.approx(20 + 250 + 9.15 + 0.5 + 0.18 * (20 + 9.15 + 0.5), abs=1e-6)
    assert C.order_charges(C.DELIVERY, "buy", 0) == 0.0
    with pytest.raises(ValueError, match="side"):
        C.order_charges(C.DELIVERY, "hold", 100)
    # a ₹10,000 delivery round trip: 11.87 + 10.37 + 18.10 = 40.35 bps
    assert C.round_trip_bps(C.DELIVERY, 10_000) == pytest.approx(40.349, abs=1e-3)


def market(
    n_days: int, paths: dict[str, list[float]], factors: dict[str, list[float]] | None = None
) -> B.Market:
    """Daily bars where open equals the previous close and close follows the given path."""
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows: list[dict[str, Any]] = []
    for sym, closes in paths.items():
        for i, c in enumerate(closes):
            if np.isnan(c):
                continue
            o = closes[i - 1] if i and not np.isnan(closes[i - 1]) else c
            f = (factors or {}).get(sym, [1.0] * n_days)[i]
            rows.append(dict(date=days[i], symbol=sym, open=o, close=c, adj_factor=f, med20=1e12))  # fmt: skip
    return B.Market.from_frame(pl.DataFrame(rows))


def const_scores(m: B.Market, ranks: dict[str, float]) -> np.ndarray:
    out = np.full((len(m.dates), len(m.symbols)), np.nan)
    for s, v in ranks.items():
        out[:, m.symbols.index(s)] = v
    return out


def test_known_outcome_top_stock_rising_one_percent_a_day() -> None:
    n = 6
    m = market(n, {"UP": [100 * 1.01**i for i in range(n)], "FLAT": [100.0] * n})
    sc = const_scores(m, {"UP": 2.0, "FLAT": 1.0})
    r = B.simulate(m, sc, B.Rules(top_n=1, rebalance=100, slippage_bps=0), 1_000, ZERO, 0, n - 1)
    # decided at day 0's close, bought at day 1's open (= day 0's close, 100): 10 shares
    assert r.log == [] and r.trades == 1
    assert r.equity[0] == 1_000  # nothing held yet
    assert r.equity[-1] == pytest.approx(10 * 100 * 1.01**5)
    # with delivery charges the buy pays STT, stamp and fees and buys one share fewer if needed
    rc = B.simulate(
        m, sc, B.Rules(top_n=1, rebalance=100, slippage_bps=0), 1_000, C.DELIVERY, 0, n - 1
    )
    assert rc.charges == pytest.approx(C.order_charges(C.DELIVERY, "buy", 900.0))
    assert rc.equity[-1] == pytest.approx(9 * 100 * 1.01**5 + 1_000 - 900 - rc.charges)


def test_fills_at_the_next_open_never_the_decision_close() -> None:
    m = market(3, {"A": [100.0, 120.0, 130.0]})
    r = B.simulate(
        m, const_scores(m, {"A": 1}), B.Rules(1, 100, slippage_bps=0), 1_000, ZERO, 0, 2, True
    )
    buy = r.log[0]
    assert buy["date"] == m.dates[1] and buy["price"] == 100.0  # day 1's open is day 0's close


def test_band_rule_keeps_a_holding_inside_the_band_and_sells_below_it() -> None:
    n, syms = 8, [f"S{i:02d}" for i in range(30)]
    m = market(n, {s: [100.0] * n for s in syms})
    sc = np.full((n, 30), np.nan)
    sc[:, :] = np.arange(30, 0, -1)  # S00 best
    # at day 2 S00 falls to rank 15 (held, band 20); at day 4 to rank 25 (sold)
    sc[2:, 0] = 30 - 14.5
    sc[4:, 0] = 30 - 24.5
    rules = B.Rules(top_n=10, rebalance=2, band=2.0, slippage_bps=0)
    r = B.simulate(m, sc, rules, 10_000, ZERO, 0, n - 1, record=True)
    s00 = [x for x in r.log if x["symbol"] == "S00"]
    assert [x["side"] for x in s00] == ["buy", "sell"]
    assert s00[1]["date"] == m.dates[5]  # decided at day 4's close, sold at day 5's open
    # fixed: the same move to rank 15 is enough to sell
    rf = B.simulate(
        m, sc, B.Rules(10, 2, exit="fixed", slippage_bps=0), 10_000, ZERO, 0, n - 1, True
    )
    assert [x["date"] for x in rf.log if x["symbol"] == "S00" and x["side"] == "sell"] == [
        m.dates[3]
    ]


def test_min_hold_delays_a_sale_by_one_decision() -> None:
    n = 8
    m = market(n, {"A": [100.0] * n, "B": [100.0] * n})
    sc = const_scores(m, {"A": 2.0, "B": 1.0})
    sc[2:, 0] = 0.5  # A falls below B at the first decision after buying
    base = B.Rules(top_n=1, rebalance=2, exit="fixed", slippage_bps=0)

    def sold(r: B.Result) -> list[date]:
        return [x["date"] for x in r.log if x["symbol"] == "A" and x["side"] == "sell"]

    assert sold(B.simulate(m, sc, base, 1_000, ZERO, 0, n - 1, True)) == [m.dates[3]]
    held = B.Rules(top_n=1, rebalance=2, exit="fixed", slippage_bps=0, min_hold=1)
    assert sold(B.simulate(m, sc, held, 1_000, ZERO, 0, n - 1, True)) == [m.dates[5]]


def test_a_split_doubles_shares_and_keeps_value() -> None:
    n = 5
    # a 1:2 split effective day 3: the price halves, the factor goes from 0.5 to 1
    m = market(n, {"A": [100.0, 100.0, 100.0, 50.0, 50.0]}, {"A": [0.5, 0.5, 0.5, 1.0, 1.0]})
    r = B.simulate(
        m, const_scores(m, {"A": 1}), B.Rules(1, 100, slippage_bps=0), 1_000, ZERO, 0, n - 1
    )
    assert r.equity[2] == pytest.approx(1_000)
    assert r.equity[3] == pytest.approx(1_000) and r.equity[4] == pytest.approx(1_000)


def test_unaffordable_stock_is_passed_over_and_cash_carries() -> None:
    n = 4
    m = market(n, {"DEAR": [5_000.0] * n, "CHEAP": [300.0] * n})
    sc = const_scores(m, {"DEAR": 2.0, "CHEAP": 1.0})
    r = B.simulate(m, sc, B.Rules(1, 100, slippage_bps=0), 1_000, ZERO, 0, n - 1, True)
    assert [x["symbol"] for x in r.log] == ["CHEAP"] and r.log[0]["shares"] == 3
    assert r.cash[-1] == pytest.approx(100.0)
    assert r.affordable == [0.5]


def test_stop_rule_sells_after_a_close_beyond_the_stop() -> None:
    n = 6
    m = market(n, {"A": [100.0, 100.0, 95.0, 91.0, 91.0, 91.0]})
    rules = B.Rules(1, 100, exit="stop", stop_loss=0.08, slippage_bps=0)
    r = B.simulate(m, const_scores(m, {"A": 1}), rules, 1_000, ZERO, 0, n - 1, True)
    sells = [x for x in r.log if x["side"] == "sell"]
    assert len(sells) == 1 and sells[0]["date"] == m.dates[4]  # 91 ≤ 100 × 0.92 at day 3's close


def test_delisted_stock_is_sold_at_its_last_close() -> None:
    n = 5
    m = market(n, {"A": [100.0, 100.0, 110.0, np.nan, np.nan], "B": [10.0] * n})
    r = B.simulate(
        m,
        const_scores(m, {"A": 2.0, "B": 1.0}),
        B.Rules(1, 100, slippage_bps=0),
        1_000,
        ZERO,
        0,
        n - 1,
        True,
    )
    assert [(x["side"], x["why"], x["price"]) for x in r.log if x["symbol"] == "A"] == [
        ("buy", "rule", 100.0), ("sell", "delisted", 110.0)
    ]  # fmt: skip


def test_grid_rows_equal_single_runs_and_grid_dedupes() -> None:
    n = 30
    rng = np.random.default_rng(1)
    paths = {
        f"S{i}": [float(x) for x in 100 * np.cumprod(1 + rng.normal(0, 0.01, n))] for i in range(12)
    }
    m = market(n, paths)
    sc = rng.random((n, 12))
    grid = B.rules_grid(
        {"top_n": [2, 3], "rebalance": [5], "exit": ["band", "fixed"], "band": [1.5, 2.0]}
    )
    assert len(grid) == 2 * (2 + 1)  # fixed ignores band
    table = B.run_grid(m, [sc], grid, 10_000, C.DELIVERY, 0, n - 1)
    for rules, row in zip(grid, table.iter_rows(named=True), strict=True):
        single = B.summarise(B.simulate(m, sc, rules, 10_000, C.DELIVERY, 0, n - 1), 10_000)
        assert row["end_value"] == pytest.approx(single["end_value"])
    with pytest.raises(TypeError):
        B.rules_grid({"nonsense": [1], "top_n": [1], "rebalance": [1]})


# ---- the reactions ----------------------------------------------------------------------------


def synthetic_prices(start: date, n_days: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, Any]] = []
    syms = [f"S{i}" for i in range(15)] + [P.ETF]
    for s in syms:
        px = 100.0
        for i in range(n_days):
            d = start - timedelta(days=60) + timedelta(days=i)
            o = px
            px *= 1 + rng.normal(0.0005, 0.01)
            rows.append(dict(date=d, symbol=s, open=o, close=px, adj_factor=1.0, adj_close=px,
                             med20=5e7, mom_12_1=float(rng.normal()), vol_120d=0.01 + rng.random() / 100,
                             hist=300))  # fmt: skip
    return pl.DataFrame(rows)


@pytest.fixture
def faked_prices(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    start = date(2025, 3, 1)
    prices = synthetic_prices(start)

    def load(root: Any, s: date, e: date) -> pl.DataFrame:
        return prices

    def index_returns(root: Any, s: date, e: date) -> dict[str, float]:
        return {"Nifty 50": 1.0}

    def revision(repo: str, token: str) -> str:
        return "rev1"

    def no_pull(
        repo: str, token: str, revision: str, patterns: list[str], root: Path
    ) -> mirror.Pulled:
        return mirror.Pulled(0, 0, 0.0)

    monkeypatch.setattr(P, "load_prices", load)
    monkeypatch.setattr(P, "index_returns", index_returns)
    monkeypatch.setenv("SECRET_HF_TOKEN", "hf_test_token_000000000000")
    monkeypatch.setattr(mirror, "store_revision", revision)
    monkeypatch.setattr(mirror, "pull", no_pull)
    return data_dir


async def test_grid_reactions_end_to_end(
    store: PostgresStore, faked_prices: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from finance import settings

    monkeypatch.setattr(settings, "PUBLISH", False)
    spec = {
        "label": "capital study test",
        "sources": [
            {"baseline": "momentum_12_1"},
            {"baseline": "equal_weight"},
            {"baseline": "niftybees"},
        ],
        "capitals": [10_000, 100_000],
        "start": "2025-03-01",
        "end": "2025-04-30",
        "grid": {"top_n": [2, 4], "rebalance": [5, 10], "exit": ["band"], "band": [2.0]},
        "draws": 3,
    }
    ctx = await run(store, RunBacktestGrid, spec)
    assert len(ctx.vars["grids"]) == 6
    from finance import app

    worker = Worker(app, store)
    while await worker.run_once():
        pass
    grids = await store.query(
        BacktestGrid, "select * from backtest_grid where label = :l", l=spec["label"]
    )
    assert {g.status for g in grids} == {"done"}, [g.error for g in grids]
    by_source = {(g.source["baseline"], g.capital): g for g in grids}
    assert by_source[("momentum_12_1", 10_000.0)].combinations == 4
    assert by_source[("niftybees", 10_000.0)].combinations == 1  # bought once, one rule
    assert by_source[("equal_weight", 10_000.0)].draws == 3
    table = pl.read_parquet(faked_prices / str(by_source[("momentum_12_1", 10_000.0)].results_path))
    assert {"top_n", "rebalance", "end_value", "charges_pct_per_year", "affordable_pct"} <= set(
        table.columns
    )
    summary = await run(store, SummarizeBacktestGrids, {"label": spec["label"]})
    assert "| capital | source |" in summary.vars["table_best"]
    assert "table_costs_10000" in summary.vars and "table_costs_100000" in summary.vars
    one = await run(store, RunBacktest, {"label": "one", "source": {"baseline": "momentum_12_1"},
                                         "rules": {"top_n": 3, "rebalance": 5}, "capital": 10_000,
                                         "start": "2025-03-01", "end": "2025-04-30"})  # fmt: skip
    runs = await store.query(BacktestRun, "select * from backtest_run where label = 'one'")
    assert runs[0].status == "done", runs[0].error
    assert one.vars["summary"]["index_return_pct"] == {"Nifty 50": 1.0}
    trades = pl.read_parquet(faked_prices / str(runs[0].trades_path))
    assert set(trades["side"].to_list()) <= {"buy", "sell"} and trades.height > 0


def test_an_account_eaten_by_charges_stops_at_zero() -> None:
    n = 40
    syms = [f"S{i:02d}" for i in range(40)]
    # ₹5 stocks in ₹50 positions: each sale's fixed ₹18 DP charge is a third of the position
    m = market(n, {s: [5.0] * n for s in syms})
    rng = np.random.default_rng(3)
    sc = rng.random((n, 40))  # a new random ranking of 40 stocks every day: maximum churn
    rules = B.Rules(top_n=20, rebalance=1, exit="fixed", slippage_bps=0)
    r = B.simulate(m, sc, rules, 1_000, C.DELIVERY, 0, n - 1)
    assert r.ruined and r.equity.min() >= 0 and r.equity[-1] == 0
    assert B.summarise(r, 1_000)["ruined"] is True


def test_a_ruin_early_in_one_year_summarises_the_years_after() -> None:
    days = [date(2024, 12, 20) + timedelta(days=i) for i in range(30)]
    syms = [f"S{i:02d}" for i in range(40)]
    rows = [
        dict(date=d, symbol=s, open=5.0, close=5.0, adj_factor=1.0, med20=1e12)
        for d in days
        for s in syms
    ]
    m = B.Market.from_frame(pl.DataFrame(rows))
    sc = np.random.default_rng(3).random((len(days), 40))
    r = B.simulate(
        m,
        sc,
        B.Rules(top_n=20, rebalance=1, exit="fixed", slippage_bps=0),
        1_000,
        C.DELIVERY,
        0,
        len(days) - 1,
    )
    assert r.ruined and r.dates[0].year == 2024 and r.dates[-1].year == 2025
    s = B.summarise(r, 1_000)
    assert s["ruined"] and s["end_value"] == 0 and s["by_year"]["2025"] in (0.0, -100.0)


def test_nearest_duration_prefers_the_closer_then_the_shorter() -> None:
    assert [P.nearest_duration(r) for r in (1, 3, 5, 7, 8, 15, 30, 50, 90, 100, 200)] == [
        1, 1, 5, 5, 10, 10, 21, 42, 63, 126, 126
    ]  # fmt: skip


def test_walkforward_scores_take_the_forecast_at_the_duration_of_the_rebalance(
    tmp_path: Path,
) -> None:
    prices = synthetic_prices(date(2025, 3, 1))
    market = B.Market.from_frame(prices)
    d0 = market.dates[5]
    rows: list[dict[str, Any]] = []
    for sym in ("S0", "S1"):
        for h, v in ((5, 1.0), (21, -1.0)):
            rows.append({"symbol": sym, "ts": datetime.combine(d0, time(15, 30)), "h": h,
                         "day": d0, "pred_y": v if sym == "S0" else -v, "y": 0.0})  # fmt: skip
    (tmp_path / "parquet" / "walkforward").mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(tmp_path / "parquet/walkforward/wf.parquet")
    s0, s1 = market.symbols.index("S0"), market.symbols.index("S1")
    short = P.walkforward_scores(tmp_path, "parquet/walkforward/wf.parquet", market, rebalance=5)
    assert (short[5, s0], short[5, s1]) == (1.0, -1.0) and np.isnan(short[6, s0])
    monthly = P.walkforward_scores(tmp_path, "parquet/walkforward/wf.parquet", market, rebalance=21)
    assert (monthly[5, s0], monthly[5, s1]) == (-1.0, 1.0)
    # the source helper routes to it, and a baseline still goes to the baselines
    rules = B.Rules(top_n=2, rebalance=5)
    src = {"walk_forward_id": "x", "predictions_path": "parquet/walkforward/wf.parquet"}
    assert np.array_equal(
        P.source_scores(tmp_path, src, prices, market, rules)[0], short, equal_nan=True
    )
    assert (
        len(P.source_scores(tmp_path, {"baseline": "equal_weight"}, prices, market, rules, draws=3))
        == 3
    )


async def test_select_portfolio_ranks_models_against_the_matching_equal_weight(
    store: PostgresStore, faked_prices: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from finance import app, settings
    from finance.ml.entities import WalkForward

    monkeypatch.setattr(settings, "PUBLISH", False)
    prices = synthetic_prices(date(2025, 3, 1))
    market = B.Market.from_frame(prices)
    # a "model" that knows the next 5 and 21 sessions: a perfect forecast, scaled, one row per day
    rng = np.random.default_rng(1)
    rows: list[dict[str, Any]] = []
    closes = {s: prices.filter(pl.col("symbol") == s).sort("date") for s in market.symbols}
    for d in market.dates:
        for s in market.symbols:
            if s == P.ETF:
                continue
            c = closes[s]["close"].to_numpy()
            j = closes[s]["date"].to_list().index(d)
            for h in (5, 21):
                if j + h + 1 < len(c):
                    fwd = float(np.log(c[j + h] / c[j + 1]))
                    rows.append({"symbol": s, "ts": datetime.combine(d, time(15, 30)), "h": h,
                                 "day": d, "pred_y": fwd + rng.normal(0, 0.001), "y": fwd})  # fmt: skip
    rel = "parquet/walkforward/test.parquet"
    (faked_prices / "parquet" / "walkforward").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(faked_prices / rel)
    wf = await store.create(
        WalkForward, fit_id="f", algorithm="oracle", params={}, feature_set="all",
        label="oracle", status="done", windows=1, predictions_path=rel,
    )  # fmt: skip
    spec = {
        "label": "select test",
        "sources": [{"walk_forward_id": wf.id}, {"baseline": "equal_weight"}],
        "capitals": [100_000],
        "start": "2025-03-01",
        "end": "2025-04-30",
        "grid": {"top_n": [2, 3], "rebalance": [5, 21], "exit": ["band"], "band": [2.0]},
        "draws": 2,
    }
    ctx = await run(store, RunBacktestGrid, spec)
    worker = Worker(app, store)
    while await worker.run_once():
        pass
    grids = await store.query(
        BacktestGrid, "select * from backtest_grid where label = :l", l="select test"
    )
    assert {g.status for g in grids} == {"done"}, [g.error for g in grids]
    model = next(g for g in grids if "walk_forward_id" in g.source)
    assert model.source["predictions_path"] == rel and model.combinations == 4
    table = pl.read_parquet(faked_prices / str(model.results_path))
    assert (
        set(table["walk_forward_id"].to_list()) == {wf.id}
        and table["source"].str.starts_with("walk_forward").all()
    )
    sel = await run(
        store, SelectPortfolio, {"label": "select test", "grid_ids": ctx.vars["grids"], "top": 3}
    )
    best, top = sel.vars["best"], sel.vars["top"]
    assert sel.vars["configurations"] == 4 and len(top) == 3 and top[0] == best
    assert (
        best["excess_pct"] == best["return_pct"] - best["return_pct_ew"] and best["excess_pct"] > 0
    )
    assert best["grid_id"] == model.id and set(top[0]) >= {"rank", "top_n", "rebalance", "sharpe"}
    assert "| rank | grid_id |" in sel.vars["table_top"]
    ranking = pl.read_parquet(faked_prices / str(sel.vars["ranking_path"]))
    assert ranking["rank"].to_list() == [1, 2, 3, 4] and ranking["excess_pct"].is_sorted(
        descending=True
    )
    stat = sel.vars["monthly_excess"]
    assert (
        stat["months"] >= 1
        and "t_stat" in stat
        and sel.vars["best_run_id"] != sel.vars["baseline_run_id"]
    )
    runs = await store.query(
        BacktestRun, "select * from backtest_run where label like 'select test %'"
    )
    assert {r.status for r in runs} == {"done"} and {r.label for r in runs} == {
        "select test best",
        "select test equal_weight",
    }
    # a source that has no predictions yet is refused before anything is queued
    with pytest.raises(ValueError, match="no predictions"):
        await run(
            store, RunBacktestGrid, {**spec, "sources": [{"walk_forward_id": str(uuid.uuid4())}]}
        )
