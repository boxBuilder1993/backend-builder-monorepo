"""The window study: for every stock, the return between any two points of a 15-minute grid, inside
one session or into the next, per year, with hit rates and consistency. The session study's single
window (15:14 to 09:29) is one of 1,250; the question is which window of each stock paid, how
consistently, and whether the nights that span a weekend or a holiday paid more than plain nights.

Grid: the close of the 09:29 bar and every 15 minutes to 15:14, plus the 15:29 close (25 points).
The 09:15 open is left out on purpose: it is the auction print, not a fill. Same-day windows are the
300 pairs t_in < t_out; cross-session windows are every pair t_in today, t_out next session (625),
split by what the night spans: ``night`` when the next session is the next calendar day, ``break``
when a weekend or holiday lies between (``across``). Returns are log returns of the long; a window
whose return is reliably negative is a short.

Summaries are polars over the study table: the market-wide picture (median across stocks of the
yearly totals per window), the weekend question, and per stock the best window chosen on earlier
years and scored on the last, which is the only honest ranking with 1,250 windows a stock.
"""

# ruff: noqa: E501  (the SQL reads better unwrapped)

from __future__ import annotations

from typing import Any

import polars as pl

GRID = tuple(f"{9 + (29 + 15 * k) // 60:02d}:{(29 + 15 * k) % 60:02d}" for k in range(24)) + (
    "15:29",
)
"""09:29, 09:44, ..., 15:14, 15:29."""

STUDY_SQL = """
with g as (
    select symbol, ts::date as d, strftime(ts, '%H:%M') as t, close,
           case when symbol like 'NIFTY %' then 'index' else 'stock' end as kind
    from equity_1m
    where strftime(ts, '%H:%M') in (__GRID__) and close > 0
      and ts >= :start::timestamp and ts < (:end::date + 1)::timestamp
    qualify row_number() over (partition by symbol, ts order by close) = 1
),
liq as (
    select symbol from (select symbol, ts::date as d, sum(close * volume) as tv from equity_1m
                        where ts >= :liq_from::timestamp group by 1, 2)
    group by 1 having median(tv) >= :min_turnover
),
u as (select g.* from g where g.kind = 'index' or g.symbol in (select symbol from liq)),
days as (
    select symbol, d, lead(d) over (partition by symbol order by d) as nd
    from (select distinct symbol, d from u)
),
same as (
    select a.symbol, a.kind, a.d, a.t as t_in, b.t as t_out, 'none' as across, ln(b.close / a.close) as r
    from u a join u b on a.symbol = b.symbol and a.d = b.d and a.t < b.t
),
nxt as (
    select a.symbol, a.kind, a.d, a.t as t_in, b.t as t_out,
           case when days.nd - a.d = 1 then 'night' else 'break' end as across,
           ln(b.close / a.close) as r
    from u a join days on days.symbol = a.symbol and days.d = a.d
    join u b on b.symbol = a.symbol and b.d = days.nd
    where days.nd is not null and days.nd - a.d <= 5
),
r as (select * from same union all select * from nxt),
per_year as (
    select symbol, kind, t_in, t_out, across, year(d) as y, count(*) as sessions,
           sum(r) as total, avg(r) as mean, avg(case when r > 0 then 1.0 else 0.0 end) as hit,
           stddev_samp(r) as sd
    from r group by all having count(*) >= :min_sessions
)
select * from per_year order by kind desc, symbol, across, t_in, t_out, y
"""


def study_sql() -> str:
    return STUDY_SQL.replace("__GRID__", ", ".join(f"'{t}'" for t in GRID))


DEFAULTS: dict[str, Any] = {
    "start": "2023-10-01",
    "end": "2030-12-31",
    "liq_from": "2024-01-01",
    "min_turnover": 5e7,
    "min_sessions": 20,
}
"""Minute bars begin in October 2023; liquidity floor ₹5 crore a day (median since 2024); a
window-year needs 20 sessions (the break windows have about 50 a year)."""


def _md(df: pl.DataFrame, pct: tuple[str, ...] = ()) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows(named=True):
        cells: list[str] = []
        for c in cols:
            v = row[c]
            if v is None:
                cells.append("")
            elif c in pct:
                cells.append(f"{float(v) * 100:+.0f}%")
            elif isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _yearly(rows: pl.DataFrame, kind: str = "stock") -> pl.DataFrame:
    return rows.filter(pl.col("kind") == kind)


def market_table(rows: pl.DataFrame, across: str, top: int = 12) -> str:
    """Per window, the median across stocks of the yearly total, per year; the windows with the
    highest worst year (longs) and the lowest best year (shorts). Across: none, night or break."""
    r = _yearly(rows).filter(pl.col("across") == across)
    years = sorted(r["y"].unique().to_list())
    med = (
        r.group_by("t_in", "t_out", "y")
        .agg(pl.col("total").median().alias("med"), pl.len().alias("stocks"))
        .pivot(on="y", index=["t_in", "t_out"], values="med")
    )
    ycols = [str(y) for y in years]
    med = med.with_columns(
        pl.min_horizontal(ycols).alias("worst"), pl.max_horizontal(ycols).alias("best")
    )
    longs = med.sort("worst", descending=True).head(top).with_columns(pl.lit("long").alias("side"))
    shorts = med.sort("best").head(top).with_columns(pl.lit("short").alias("side"))
    out = pl.concat([longs, shorts]).select(["side", "t_in", "t_out", *ycols, "worst", "best"])
    return _md(out, pct=tuple(ycols) + ("worst", "best"))


def weekend_table(rows: pl.DataFrame, t_in: str = "15:14", t_out: str = "09:29") -> str:
    """The overnight window into a plain night against one that spans a weekend or holiday: per
    year, the median across stocks of the mean return per session (bps) and the hit rate, and the
    indices one by one."""
    w = rows.filter(
        (pl.col("t_in") == t_in) & (pl.col("t_out") == t_out) & (pl.col("across") != "none")
    )
    stocks = (
        _yearly(w)
        .group_by("y", "across")
        .agg(
            pl.len().alias("stocks"),
            (pl.col("mean").median() * 1e4).round(1).alias("median bps per session"),
            (pl.col("hit").median() * 100).round(0).alias("median hit %"),
            pl.col("sessions").median().alias("sessions"),
        )
        .sort("y", "across")
    )
    idx = (
        _yearly(w, "index")
        .select(
            "symbol",
            "y",
            "across",
            "sessions",
            (pl.col("mean") * 1e4).round(1).alias("bps per session"),
            (pl.col("total") * 100).round(1).alias("total %"),
            (pl.col("hit") * 100).round(0).alias("hit %"),
        )
        .sort("symbol", "y", "across")
    )
    return "Stocks (median across stocks):\n\n" + _md(stocks) + "\n\nIndices:\n\n" + _md(idx)


def stock_table(
    rows: pl.DataFrame, pick_years: tuple[int, ...], test_year: int, top: int = 25
) -> str:
    """Per stock, the window with the highest worst total over the pick years (positive in each),
    then that same window's total in the test year, which it was not chosen on. Ranked by the pick;
    the last lines give the mean and hit of the picks in the test year against every window's."""
    r = _yearly(rows).filter(pl.col("sessions") >= 100)
    picks = (
        r.filter(pl.col("y").is_in(list(pick_years)))
        .group_by("symbol", "t_in", "t_out", "across")
        .agg(pl.col("total").min().alias("worst"), pl.len().alias("k"))
        .filter((pl.col("k") == len(pick_years)) & (pl.col("worst") > 0))
        .sort("worst", descending=True)
        .group_by("symbol", maintain_order=True)
        .first()
    )
    test = r.filter(pl.col("y") == test_year).select(
        "symbol",
        "t_in",
        "t_out",
        "across",
        pl.col("total").alias("test_total"),
        pl.col("hit").alias("test_hit"),
        pl.col("sessions").alias("test_sessions"),
    )
    joined = picks.join(test, on=["symbol", "t_in", "t_out", "across"], how="left").sort(
        "worst", descending=True
    )
    table = joined.head(top).select(
        "symbol",
        "t_in",
        "t_out",
        "across",
        pl.col("worst").alias(f"worst of {'/'.join(map(str, pick_years))}"),
        pl.col("test_total").alias(f"{test_year}"),
        (pl.col("test_hit") * 100).round(0).alias(f"{test_year} hit %"),
        pl.col("test_sessions").alias(f"{test_year} sessions"),
    )
    scored = joined.filter(pl.col("test_total").is_not_null())
    universe = r.filter(pl.col("y") == test_year)

    def f(v: Any) -> float:
        return float(v) if v is not None else float("nan")

    summary = (
        f"\n\nPicked windows in {test_year}: {scored.height} stocks, mean total "
        f"{f(scored['test_total'].mean()) * 100:+.1f}%, median {f(scored['test_total'].median()) * 100:+.1f}%, "
        f"{f((scored['test_total'] > 0).mean()) * 100:.0f}% positive; every window of every stock that year: "
        f"median {f(universe['total'].median()) * 100:+.1f}%, {f((universe['total'] > 0).mean()) * 100:.0f}% positive."
    )
    pct = (f"worst of {'/'.join(map(str, pick_years))}", f"{test_year}")
    return _md(table, pct=pct) + summary
