"""A long-only delivery account, simulated day by day: scores in, trades and daily equity out.

Pure: no runtime, no database. The reactions in ``reactions.py`` load prices and scores through
DuckDB, build a ``Market`` and call ``simulate`` or ``run_grid``. See docs/finance-portfolio.md §6.

Mechanics, in the order they happen on each trading day t:

1. Corporate actions: a held stock whose adjustment factor changed since yesterday has its shares
   multiplied by factor(t) / factor(t-1) (a 1:2 split doubles them), its cost price divided by it.
2. At t's open: yesterday's sell orders fill (open × (1 - slippage)), then the buy orders, each
   getting the cash left divided by the buys still to do, capped at a share of the stock's recent
   turnover; whole shares; charges from the cost schedule, the DP charge once per stock sold.
   An order for a stock with no price at t's open waits (sells) or is dropped (buys).
3. At t's close: a stock whose last row in the data is t (delisted) is sold at the close. Under the
   ``stop`` rule, a close beyond the stop-loss or take-profit queues a sell for t+1's open.
4. If t is a decision day: stocks are ranked by score at t (higher is better); holdings are kept or
   queued for sale by the exit rule; the open slots are queued for purchase from the best-ranked
   stocks not held, among those whose single share is affordable within equity ÷ N.
5. The account is marked at the close (a stock with no close keeps its last price).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from itertools import product
from typing import Any

import numpy as np
import polars as pl

from finance.ml import costs as C

TRADING_DAYS = 248
RISK_FREE = 0.065


@dataclass(frozen=True)
class Rules:
    top_n: int
    rebalance: int
    """Sessions between decisions; the first decision is on the first day of the period."""
    exit: str = "band"
    """``band``: keep while rank <= ceil(band × N). ``fixed``: keep only the top N (band = 1).
    ``score``: keep while the score is above zero, buy only above zero (rank ignored).
    ``stop``: band, plus a sale after a close at or beyond the stop-loss or take-profit."""
    band: float = 2.0
    min_hold: int = 0
    """Decisions a new holding survives before it may be sold by rank (stops excepted)."""
    slippage_bps: float = 5.0
    stop_loss: float = 0.08
    take_profit: float = 0.20
    max_turnover_share: float = 0.02
    """A position is at most this share of the stock's 20-session median rupee turnover."""

    def keep_rank(self) -> int:
        if self.exit == "fixed":
            return self.top_n
        return max(self.top_n, math.ceil(self.band * self.top_n))


@dataclass
class Market:
    """Dense daily arrays over (date, symbol): unadjusted open and close, the adjustment factor,
    and the most a position may be in rupees (NaN where there is no row)."""

    dates: list[date]
    symbols: list[str]
    open: np.ndarray
    close: np.ndarray
    factor: np.ndarray
    cap: np.ndarray
    last_row: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        has = ~np.isnan(self.close)
        idx = np.where(has, np.arange(len(self.dates))[:, None], -1)
        self.last_row = idx.max(axis=0)

    @classmethod
    def from_frame(cls, prices: pl.DataFrame, max_turnover_share: float = 0.02) -> Market:
        """``prices``: date, symbol, open, close, adj_factor, med20 (the 20-session median rupee
        turnover before the day)."""
        dates = sorted(prices["date"].unique().to_list())
        symbols = sorted(prices["symbol"].unique().to_list())
        di = {d: i for i, d in enumerate(dates)}
        si = {s: i for i, s in enumerate(symbols)}
        shape = (len(dates), len(symbols))
        arrays = {k: np.full(shape, np.nan) for k in ("open", "close", "factor", "cap")}
        r = prices.select("date", "symbol", "open", "close", "adj_factor", "med20")
        rows = np.array([di[d] for d in r["date"].to_list()])
        cols = np.array([si[s] for s in r["symbol"].to_list()])
        arrays["open"][rows, cols] = r["open"].to_numpy()
        arrays["close"][rows, cols] = r["close"].to_numpy()
        arrays["factor"][rows, cols] = r["adj_factor"].fill_null(1.0).to_numpy()
        arrays["cap"][rows, cols] = r["med20"].to_numpy() * max_turnover_share
        return cls(dates, symbols, arrays["open"], arrays["close"], arrays["factor"], arrays["cap"])

    def scores(self, frame: pl.DataFrame) -> np.ndarray:
        """A (date, symbol, score) frame as an array aligned to this market, NaN where absent."""
        di = {d: i for i, d in enumerate(self.dates)}
        si = {s: i for i, s in enumerate(self.symbols)}
        out = np.full((len(self.dates), len(self.symbols)), np.nan)
        f = frame.filter(pl.col("date").is_in(self.dates) & pl.col("symbol").is_in(self.symbols))
        rows = np.array([di[d] for d in f["date"].to_list()], dtype=int)
        cols = np.array([si[s] for s in f["symbol"].to_list()], dtype=int)
        if len(rows):
            out[rows, cols] = f["score"].to_numpy()
        return out

    def index(self, d: date, after: bool = False) -> int:
        """The position of the first trading day on or after ``d`` (``after``: strictly after)."""
        for i, x in enumerate(self.dates):
            if x > d or (x == d and not after):
                return i
        return len(self.dates)


@dataclass
class Holding:
    shares: float
    cost: float
    factor: float
    bought_t: int
    bought_decision: int


@dataclass
class Result:
    dates: list[date]
    equity: np.ndarray
    cash: np.ndarray
    charges: float
    traded: float
    trades: int
    hold_days: list[int]
    affordable: list[float]
    """At each decision, the share of scored stocks whose single share fit the allocation."""
    log: list[dict[str, Any]]
    ruined: bool = False
    """True when the account reached zero and stopped trading."""


def simulate(
    market: Market,
    scores: np.ndarray,
    rules: Rules,
    capital: float,
    schedule: C.Schedule,
    start: int,
    end: int,
    record: bool = False,
) -> Result:
    """One account from the trading day at index ``start`` to ``end`` inclusive."""
    slip = rules.slippage_bps / 1e4
    cash = float(capital)
    held: dict[int, Holding] = {}
    last_px: dict[int, float] = {}
    sells: set[int] = set()
    buys: list[int] = []
    charges = traded = 0.0
    trades = 0
    decision = 0
    hold_days: list[int] = []
    affordable: list[float] = []
    log: list[dict[str, Any]] = []
    n_days = end - start + 1
    equity = np.zeros(n_days)
    cash_series = np.zeros(n_days)
    keep_rank = rules.keep_rank()
    ruined = False
    buy_rate = schedule.stt_buy + schedule.stamp_buy + schedule.exchange * (1 + schedule.gst)

    def sell(s: int, t: int, price: float, why: str) -> None:
        nonlocal cash, charges, traded, trades
        h = held.pop(s)
        value = h.shares * price
        c = C.order_charges(schedule, "sell", value) + C.dp_charge(schedule)
        cash += value - c
        charges += c
        traded += value
        trades += 1
        hold_days.append(t - h.bought_t)
        if record:
            log.append(_trade(market, t, s, "sell", h.shares, price, value, c, why))

    for k, t in enumerate(range(start, end + 1)):
        # 1. corporate actions
        for s, h in held.items():
            f = market.factor[t, s]
            if not np.isnan(f) and f != h.factor:
                ratio = f / h.factor
                h.shares *= ratio
                h.cost /= ratio
                h.factor = f
        # 2. fills at the open: sells, then buys
        for s in sorted(sells):
            o = market.open[t, s]
            if s in held and not np.isnan(o):
                sell(s, t, o * (1 - slip), "rule")
        sells = {s for s in sells if s in held}
        todo = list(buys)
        buys = []
        for j, s in enumerate(todo):
            o = market.open[t, s]
            if np.isnan(o) or s in held:
                continue
            price = o * (1 + slip)
            alloc = cash / (len(todo) - j)
            cap = market.cap[t, s]
            if not np.isnan(cap):
                alloc = min(alloc, cap)
            shares = math.floor(alloc / (price * (1 + buy_rate)))
            if shares < 1:
                continue
            value = shares * price
            c = C.order_charges(schedule, "buy", value)
            while shares >= 1 and value + c > cash:
                shares -= 1
                value = shares * price
                c = C.order_charges(schedule, "buy", value)
            if shares < 1:
                continue
            cash -= value + c
            charges += c
            traded += value
            trades += 1
            # the decision that queued this buy is the one before the counter moved on
            held[s] = Holding(shares, price, market.factor[t, s], t, decision - 1)
            if record:
                log.append(_trade(market, t, s, "buy", shares, price, value, c, "rule"))
        # 3. at the close: delistings and stops
        for s in list(held):
            c_t = market.close[t, s]
            if market.last_row[s] == t and t < len(market.dates) - 1:
                sell(s, t, c_t, "delisted")
                continue
            if rules.exit == "stop" and not np.isnan(c_t):
                h = held[s]
                if c_t <= h.cost * (1 - rules.stop_loss) or c_t >= h.cost * (1 + rules.take_profit):
                    sells.add(s)
        # 4. a decision
        if k % rules.rebalance == 0 and t < end:
            sc = scores[t]
            close_t = market.close[t]
            valid = ~np.isnan(sc) & ~np.isnan(close_t)
            idx = np.flatnonzero(valid)
            order = idx[np.argsort(-sc[idx], kind="stable")]
            rank = {int(s): i + 1 for i, s in enumerate(order)}
            for s, h in held.items():
                if s in sells:
                    continue
                if rules.exit == "score":
                    bad = not sc[s] > 0
                else:
                    bad = rank.get(s, 10**9) > keep_rank
                if bad and decision - h.bought_decision > rules.min_hold:
                    sells.add(s)
            slots = rules.top_n - (len(held) - len(sells))
            value_now = cash + sum(h.shares * _px(market, t, s, last_px) for s, h in held.items())
            target = value_now / rules.top_n
            fits = close_t[idx] * (1 + slip) * (1 + buy_rate) <= target
            affordable.append(float(fits.mean()) if len(idx) else 0.0)
            if slots > 0:
                for s in order:
                    s = int(s)
                    if len(buys) == slots:
                        break
                    if s in held or s in sells:
                        continue
                    if rules.exit == "score" and not sc[s] > 0:
                        break
                    if close_t[s] * (1 + slip) * (1 + buy_rate) <= target:
                        buys.append(s)
            decision += 1
        # 5. mark at the close
        for s in held:
            c_t = market.close[t, s]
            if not np.isnan(c_t):
                last_px[s] = c_t
        equity[k] = cash + sum(h.shares * last_px.get(s, h.cost) for s, h in held.items())
        cash_series[k] = cash
        if equity[k] <= 0:
            # ruined: charges on positions too small to carry them (a fixed ₹18 DP charge on a
            # ₹500 sale) have taken everything; a real account stops here at zero
            equity[k:] = 0.0
            cash_series[k:] = 0.0
            ruined = True
            break
    return Result(market.dates[start : end + 1], equity, cash_series, charges, traded, trades,
                  hold_days, affordable, log, ruined)  # fmt: skip


def _trade(
    market: Market,
    t: int,
    s: int,
    side: str,
    shares: float,
    price: float,
    value: float,
    charges: float,
    why: str,
) -> dict[str, Any]:
    return {
        "date": market.dates[t],
        "symbol": market.symbols[s],
        "side": side,
        "shares": shares,
        "price": price,
        "value": value,
        "charges": charges,
        "why": why,
    }


def _px(market: Market, t: int, s: int, last_px: dict[int, float]) -> float:
    c = market.close[t, s]
    return float(c) if not np.isnan(c) else last_px.get(s, 0.0)


def summarise(result: Result, capital: float) -> dict[str, Any]:
    """The account's numbers: end value, CAGR, volatility, Sharpe, drawdown, charges, turnover, cash
    drag, holding period, trades, and the return of each calendar year."""
    eq = result.equity
    n = len(eq)
    if n == 0:
        return {}
    prev = np.r_[capital, eq][:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(np.r_[capital, eq]) / np.where(prev > 0, prev, 1), 0.0)
    years = n / TRADING_DAYS
    end = float(eq[-1])
    cagr = (end / capital) ** (1 / years) - 1 if end > 0 and years > 0 else -1.0
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=-1.0)
    vol = float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    peak = np.maximum.accumulate(np.r_[capital, eq])[1:]
    dd = float((eq / peak - 1).min())
    by_year: dict[str, float] = {}
    prev = capital
    years_seen = sorted({d.year for d in result.dates})
    for y in years_seen:
        last = max(i for i, d in enumerate(result.dates) if d.year == y)
        by_year[str(y)] = round((float(eq[last]) / prev - 1) * 100, 2)
        prev = float(eq[last])
    avg_equity = float(np.mean(eq))
    per_year = 1 / years if years > 0 else 0.0
    hold = float(np.mean(result.hold_days)) if result.hold_days else 0.0
    afford = float(np.mean(result.affordable)) if result.affordable else 0.0
    return {
        "end_value": round(end, 2),
        "return_pct": round((end / capital - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "sharpe": round((cagr - RISK_FREE) / vol, 2) if vol > 0 else 0.0,
        "max_drawdown_pct": round(dd * 100, 2),
        "charges": round(float(result.charges), 2),
        "charges_pct_per_year": round(result.charges / avg_equity * per_year * 100, 2),
        "turnover_per_year": round(result.traded / avg_equity * per_year, 2),
        "cash_drag_pct": round(float(np.mean(result.cash / np.where(eq > 0, eq, 1))) * 100, 2),
        "trades": result.trades,
        "avg_hold_days": round(hold, 1),
        "affordable_pct": round(afford * 100, 1),
        "days": n,
        "ruined": result.ruined,
        "by_year": by_year,
    }


def rules_grid(grid: dict[str, list[Any]], base: Rules | None = None) -> list[Rules]:
    """Every combination of the listed rule values; ``band`` only varies for the band and stop exits
    (fixed and score ignore it), so no combination is counted twice."""
    keys = sorted(grid)
    seen: set[Rules] = set()
    out: list[Rules] = []
    for values in product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, values, strict=True))
        r = replace(base, **kw) if base else Rules(**kw)
        if r.exit in ("fixed", "score"):
            r = replace(r, band=1.0)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def run_grid(
    market: Market,
    score_draws: list[np.ndarray],
    grid: list[Rules],
    capital: float,
    schedule: C.Schedule,
    start: int,
    end: int,
) -> pl.DataFrame:
    """Every rule combination on one source. Several score arrays (draws) are averaged: one for a
    model or a deterministic baseline, many for the random equal-weight baseline."""
    rows: list[dict[str, Any]] = []
    for rules in grid:
        sums: list[dict[str, Any]] = []
        for sc in score_draws:
            res = simulate(market, sc, rules, capital, schedule, start, end)
            sums.append(summarise(res, capital))
        row: dict[str, Any] = {k: v for k, v in asdict(rules).items()}
        row["draws"] = len(sums)
        row["ruined_share"] = float(np.mean([s["ruined"] for s in sums]))
        for key in ("end_value", "return_pct", "cagr_pct", "vol_pct", "sharpe", "max_drawdown_pct",
                    "charges", "charges_pct_per_year", "turnover_per_year", "cash_drag_pct",
                    "trades", "avg_hold_days", "affordable_pct"):  # fmt: skip
            row[key] = float(np.mean([s[key] for s in sums]))
        for y in sums[0]["by_year"]:
            row[f"y{y}_pct"] = float(np.mean([s["by_year"][y] for s in sums]))
        rows.append(row)
    return pl.DataFrame(rows)
