"""The forecast rows: one per (symbol, decision minute) over equity_1m, features from the bars up to t,
the daily layer frozen at the previous session, the target as the volatility-scaled log return from
the next bar's open to the close :horizon_minutes later. docs/finance-forecast.md is the design.

ROWS_SQL is the query a Fit registers, with its parameters; ``screen`` is the per-feature IC screen
that decides which features stay. Nothing here touches the runtime.
"""

# ruff: noqa: E501  (the SQL reads better unwrapped)

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from finance.ml import data

INDEX = "NIFTY 50"
"""The Kite symbol of the index whose minute bars are the market's."""
INDEX_DAILY = "Nifty 50"
"""Its name in index_daily."""
SESSION_MINUTES = 375
EXPIRY_WEEKDAY_CHANGE = date(2025, 9, 1)
"""NSE index derivatives expire on Thursday before this, Tuesday from it."""

PARAMS: dict[str, Any] = {
    "horizon_minutes": 15,
    "step_minutes": 5,
    "vol_minutes": 60,
    "liq_minutes": 60,
    "min_turnover": 2_000_000.0,
    "start": "2023-10-03",
    "end": "2030-12-31",
}

KEY = ["symbol", "ts"]

INTRADAY = [
    "r1",
    "r5",
    "r15",
    "r30",
    "r60",
    "r_open",
    "range_pos",
    "vwap_dev",
    "rv30",
    "rv_open",
    "relvol5",
    "relvol15",
    "turn_liq",
    "i_r5",
    "i_r15",
    "i_r30",
    "i_r60",
    "i_r_open",
    "x_r15",
    "x_r60",
]
PRESSURE = [
    "imb5",
    "imb15",
    "imb30",
    "clv5",
    "clv15",
    "clv30",
    "vwret5",
    "vwret15",
    "vwret30",
    "up5",
    "up15",
    "up30",
    "impact15",
    "impact30",
    "turn_accel",
]
DAILY = [
    "ret_1d",
    "gap",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "dist_hi",
    "dist_lo",
    "turn_rel",
    "deliv_pct",
    "rs_20",
]
CLOCK = ["m", "m_left", "dow", "expiry", "after_holiday"]
FEATURES: list[str] = INTRADAY + PRESSURE + DAILY + CLOCK
CATEGORICAL = {"dow"}
CONTEXT = {"i_r5", "i_r15", "i_r30", "i_r60", "i_r_open", *CLOCK}
"""Constant across stocks at a decision point: the market and the clock. They condition the others
rather than rank stocks, so they have no cross-sectional IC and are not rank-normalised in the Fit."""

ROWS_SQL = f"""
with
bars as (
    select distinct symbol, ts, ts::date as day, open, high, low, close, volume, close * volume as turnover
    from equity_1m
    where symbol not like 'NIFTY %' and ts::date between :start and :end
),
idx0 as (
    select distinct ts, ts::date as day, open, close from equity_1m where symbol = '{INDEX}' and ts::date between :start and :end
),
idx as (
    select ts,
           ln(close / lag(close, 5) over w) as i_r5, ln(close / lag(close, 15) over w) as i_r15,
           ln(close / lag(close, 30) over w) as i_r30, ln(close / lag(close, 60) over w) as i_r60,
           ln(close / first_value(open) over w) as i_r_open
    from idx0 window w as (partition by day order by ts)
),
b1 as (
    select *,
           datediff('minute', day::timestamp + interval 9 hour + interval 15 minute, ts) as m,
           ln(close / lag(close) over w) as ret1,
           ln(close / lag(close, 5) over w) as r5, ln(close / lag(close, 15) over w) as r15,
           ln(close / lag(close, 30) over w) as r30, ln(close / lag(close, 60) over w) as r60,
           first_value(open) over w as day_open,
           ln(close / first_value(open) over w) as r_open,
           (close - min(low) over wd) / nullif(max(high) over wd - min(low) over wd, 0) as range_pos,
           close / nullif(sum(close * volume) over wd / nullif(sum(volume) over wd, 0), 0) - 1 as vwap_dev,
           sum(volume) over wd as cumvol,
           volume * ((close - low) - (high - close)) / nullif(high - low, 0) as clv_bar,
           lead(open, 1) over w as entry,
           lead(close, :horizon_minutes) over w as exit_h,
           last_value(close) over (partition by symbol, day order by ts rows between unbounded preceding and unbounded following) as day_close
    from bars
    window w as (partition by symbol, day order by ts),
           wd as (partition by symbol, day order by ts rows between unbounded preceding and current row)
),
b2 as (
    select *,
           ret1 as r1,
           stddev_samp(ret1) over w30 as rv30,
           stddev_samp(ret1) over wd as rv_open,
           stddev_samp(ret1) over (partition by symbol, day order by ts rows between :vol_minutes - 1 preceding and current row) as sig1,
           sum(volume) over w5 as vol5, sum(volume) over w15 as vol15,
           sum(turnover) over w5 as turn5, sum(turnover) over w30 as turn30,
           sum(turnover) over (partition by symbol, day order by ts rows between :liq_minutes - 1 preceding and current row) as turn_liq,
           sum(case when ret1 > 0 then volume when ret1 < 0 then -volume end) over w5 / nullif(sum(volume) over w5, 0) as imb5,
           sum(case when ret1 > 0 then volume when ret1 < 0 then -volume end) over w15 / nullif(sum(volume) over w15, 0) as imb15,
           sum(case when ret1 > 0 then volume when ret1 < 0 then -volume end) over w30 / nullif(sum(volume) over w30, 0) as imb30,
           sum(clv_bar) over w5 / nullif(sum(volume) over w5, 0) as clv5,
           sum(clv_bar) over w15 / nullif(sum(volume) over w15, 0) as clv15,
           sum(clv_bar) over w30 / nullif(sum(volume) over w30, 0) as clv30,
           sum(ret1 * volume) over w5 / nullif(sum(volume) over w5, 0) as vwret5,
           sum(ret1 * volume) over w15 / nullif(sum(volume) over w15, 0) as vwret15,
           sum(ret1 * volume) over w30 / nullif(sum(volume) over w30, 0) as vwret30,
           avg(case when ret1 > 0 then 1.0 when ret1 < 0 then 0.0 end) over w5 as up5,
           avg(case when ret1 > 0 then 1.0 when ret1 < 0 then 0.0 end) over w15 as up15,
           avg(case when ret1 > 0 then 1.0 when ret1 < 0 then 0.0 end) over w30 as up30,
           abs(r15) / nullif(sum(turnover) over w15, 0) * 1e7 as impact15,
           abs(r30) / nullif(sum(turnover) over w30, 0) * 1e7 as impact30
    from b1
    window w5 as (partition by symbol, day order by ts rows between 4 preceding and current row),
           w15 as (partition by symbol, day order by ts rows between 14 preceding and current row),
           w30 as (partition by symbol, day order by ts rows between 29 preceding and current row),
           wd as (partition by symbol, day order by ts rows between unbounded preceding and current row)
),
b3 as (
    select *,
           vol5 / nullif(avg(vol5) over wc, 0) as relvol5,
           vol15 / nullif(avg(vol15) over wc, 0) as relvol15,
           turn5 / nullif(turn30 / 6, 0) as turn_accel
    from b2
    window wc as (partition by symbol, m order by day rows between 20 preceding and 1 preceding)
),
d0 as (
    select symbol, date, close, high, low, turnover, deliv_pct,
           ln(close / lag(close) over (partition by symbol order by date)) as dret
    from equity
    qualify row_number() over (partition by symbol, date order by series) = 1
),
d1 as (
    select symbol, date, close as d_close, deliv_pct,
           dret as ret_1d,
           ln(close / lag(close, 5) over w) as ret_5d,
           ln(close / lag(close, 20) over w) as ret_20d,
           stddev_samp(dret) over (partition by symbol order by date rows between 19 preceding and current row) as vol_20d,
           close / nullif(max(high) over (partition by symbol order by date rows between 251 preceding and current row), 0) as dist_hi,
           close / nullif(min(low) over (partition by symbol order by date rows between 251 preceding and current row), 0) as dist_lo,
           turnover / nullif(quantile_cont(turnover, 0.5) over (partition by symbol order by date rows between 20 preceding and 1 preceding), 0) as turn_rel
    from d0 window w as (partition by symbol order by date)
),
i1 as (
    select date, ln(close / lag(close, 20) over (order by date)) as i_ret_20d
    from index_daily where index_name = '{INDEX_DAILY}'
),
days as (
    select day, lag(day) over (order by day) as prev_day from (select distinct day from bars)
),
rows_ as (
    select b.symbol, b.ts, b.day, b.close, b.entry, b.sig1,
           case when :horizon_minutes >= {SESSION_MINUTES} then b.day_close else b.exit_h end as exit_,
           b.r1, b.r5, b.r15, b.r30, b.r60, b.r_open, b.range_pos, b.vwap_dev, b.rv30, b.rv_open,
           b.relvol5, b.relvol15, b.turn_liq,
           i.i_r5, i.i_r15, i.i_r30, i.i_r60, i.i_r_open,
           b.r15 - i.i_r15 as x_r15, b.r60 - i.i_r60 as x_r60,
           b.imb5, b.imb15, b.imb30, b.clv5, b.clv15, b.clv30, b.vwret5, b.vwret15, b.vwret30,
           b.up5, b.up15, b.up30, b.impact15, b.impact30, b.turn_accel,
           d.ret_1d, b.day_open / nullif(d.d_close, 0) - 1 as gap, d.ret_5d, d.ret_20d, d.vol_20d,
           d.dist_hi, d.dist_lo, d.turn_rel, d.deliv_pct, d.ret_20d - ii.i_ret_20d as rs_20,
           b.m, {SESSION_MINUTES} - b.m as m_left, isodow(b.day) as dow,
           (b.day = last_day(b.day) - (((isodow(last_day(b.day)) - (case when b.day < date '{EXPIRY_WEEKDAY_CHANGE.isoformat()}' then 4 else 2 end) + 7) % 7))::int) as expiry,
           (b.day - dy.prev_day > case when isodow(b.day) = 1 then 3 else 1 end)::int as after_holiday
    from b3 b
    left join idx i using (ts)
    asof left join d1 d on b.symbol = d.symbol and b.day > d.date
    asof left join i1 ii on b.day > ii.date
    left join days dy using (day)
    where b.m % :step_minutes = 0 and b.turn_liq >= :min_turnover
)
select *,
       ln(exit_ / entry) as r,
       ln(exit_ / entry) / (sig1 * sqrt(:horizon_minutes)) as y
from rows_
where entry is not null and exit_ is not null and sig1 > 0
order by ts, symbol
"""


def fit_definition(horizon_minutes: int, **overrides: Any) -> dict[str, Any]:
    """The Fit body for one horizon: the query above, its params, features with kinds, the target."""
    params = {**PARAMS, "horizon_minutes": horizon_minutes, **overrides}
    return {
        "name": f"forecast {horizon_minutes}m",
        "description": f"Expected volatility-scaled log return {horizon_minutes} minutes ahead; docs/finance-forecast.md",
        "query": ROWS_SQL,
        "params": params,
        "key": KEY,
        "features": [
            {"name": f, "kind": "categorical" if f in CATEGORICAL else "numeric"} for f in FEATURES
        ],
        "target": {"columns": ["y"], "task": "regression"},
    }


# ---- the IC screen ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class IC:
    feature: str
    year: int
    ic: float
    se: float
    points: int
    """Decision points (distinct ts) with at least MIN_CROSS_SECTION rows."""


MIN_CROSS_SECTION = 20


def materialise(con: duckdb.DuckDBPyConnection, params: dict[str, Any], out: Path) -> int:
    """Run ROWS_SQL once and write the frame to Parquet; the screen reads three columns at a time from it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    data.relation(con, ROWS_SQL, params).write_parquet(out.as_posix(), compression="zstd")
    return int(con.sql(f"select count(*) from read_parquet('{out.as_posix()}')").fetchall()[0][0])


def screen(
    con: duckdb.DuckDBPyConnection, rows: Path, features: list[str] | None = None
) -> list[IC]:
    """Per feature, per year: the Spearman correlation with y within each decision point, averaged over
    decision points, with the standard error of that mean."""
    out: list[IC] = []
    for f in features or [f for f in FEATURES if f not in CONTEXT]:
        got = con.sql(
            f"""
            with ranked as (
                select ts, year(ts) as yr,
                       rank() over (partition by ts order by {f}) as rf,
                       rank() over (partition by ts order by y) as ry
                from read_parquet('{rows.as_posix()}') where {f} is not null and y is not null),
            per_point as (
                select ts, yr, corr(rf, ry) as c from ranked group by ts, yr having count(*) >= {MIN_CROSS_SECTION})
            select yr, avg(c), stddev_samp(c) / sqrt(count(c)), count(c) from per_point where c is not null group by yr order by yr"""
        ).fetchall()
        out.extend(IC(f, int(yr), float(ic), float(se or 0.0), int(n)) for yr, ic, se, n in got)
    return out


def screen_table(ics: list[IC]) -> str:
    """Markdown: one row per feature, one column per year with IC and its t-statistic, for the design doc."""
    years = sorted({i.year for i in ics})
    by: dict[str, dict[int, IC]] = {}
    for i in ics:
        by.setdefault(i.feature, {})[i.year] = i
    lines = [
        "| feature | " + " | ".join(str(y) for y in years) + " | verdict |",
        "|---|" + "---|" * (len(years) + 1),
    ]
    for f in FEATURES:
        if f in CONTEXT:
            lines.append(f"| {f} | " + " | ".join("" for _ in years) + " | context |")
            continue
        cells = by.get(f, {})
        ts = [
            cells[y].ic / cells[y].se if y in cells and cells[y].se > 0 else math.nan for y in years
        ]
        parts = [
            f"{cells[y].ic:+.4f} (t={t:.1f})" if y in cells else ""
            for y, t in zip(years, ts, strict=True)
        ]
        signs = {
            math.copysign(1, cells[y].ic)
            for y in years
            if y in cells and abs(cells[y].ic / cells[y].se) >= 2
        }
        if all(abs(t) < 2 for t in ts if not math.isnan(t)):
            verdict = "drop"
        elif len(signs) > 1:
            verdict = "flips"
        else:
            verdict = "keep"
        lines.append(f"| {f} | " + " | ".join(parts) + f" | {verdict} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """python -m finance.ml.forecast --horizon 15 [--start ... --end ... --min-turnover ...]"""
    import argparse

    ap = argparse.ArgumentParser(description="materialise the forecast rows and run the IC screen")
    ap.add_argument("--horizon", type=int, default=PARAMS["horizon_minutes"])
    ap.add_argument("--step", type=int, default=PARAMS["step_minutes"])
    ap.add_argument("--vol", type=int, default=PARAMS["vol_minutes"])
    ap.add_argument("--liq", type=int, default=PARAMS["liq_minutes"])
    ap.add_argument("--min-turnover", type=float, default=PARAMS["min_turnover"])
    ap.add_argument("--start", default=PARAMS["start"])
    ap.add_argument("--end", default=PARAMS["end"])
    ap.add_argument(
        "--root", type=Path, default=None, help="the mirror; default the store's mirror dir"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="rows parquet; default work_dir()/forecast/<params>.parquet",
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory", default="16GB")
    a = ap.parse_args(argv)
    if a.root is None:
        from finance import mirror, settings

        a.root = mirror.mirror_dir(settings.HF_REPO)
    params = {
        "horizon_minutes": a.horizon,
        "step_minutes": a.step,
        "vol_minutes": a.vol,
        "liq_minutes": a.liq,
        "min_turnover": a.min_turnover,
        "start": a.start,
        "end": a.end,
    }
    tag = f"h{a.horizon}_s{a.step}_v{a.vol}_l{a.liq}_t{int(a.min_turnover)}_{a.start}_{a.end}"
    out = a.out or data.work_dir() / "forecast" / f"{tag}.parquet"
    con = data.connect(a.root)
    con.execute(f"set threads={a.threads}; set memory_limit='{a.memory}'")
    import time

    t0 = time.time()
    n = materialise(con, params, out)
    print(f"rows: {n:,} in {time.time() - t0:.0f}s -> {out}")
    t0 = time.time()
    ics = screen(con, out)
    print(f"screen in {time.time() - t0:.0f}s\n")
    print(screen_table(ics))


if __name__ == "__main__":
    main()
