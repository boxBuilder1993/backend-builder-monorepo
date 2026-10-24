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
"""09:15 to 15:29. The Muhurat session (one evening a year, 18:00 to 19:15) is not part of the rows:
its minutes fall outside this range and are filtered out."""
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
SHAPE = ["or_pos", "or_break", "clv_prev"]
"""Opening range (first 15 minutes): position within it and whether price is outside it; yesterday's
close location in its own range."""
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
PARTICIPATION = ["log_trades", "avg_trade", "trades_rel"]
"""Yesterday's number of trades, average trade size (turnover over trades) and trades against the
20-day median: large average trades are institutions, many small ones are retail."""
ATTRIBUTES = [
    "industry",
    "sector_idx",
    "in_n50",
    "in_n500",
    "in_mid100",
    "in_small250",
    "log_px",
    "tick_bps",
    "log_turn20",
    "vol_60d",
]
"""Slow properties of the stock: NSE industry and sector index from the newest constituent snapshot
on or before the day, index membership, price level and tick size over price (market mechanics),
typical turnover and volatility."""
PEERS = [
    "p_r5",
    "p_r15",
    "p_r30",
    "p_r60",
    "p_r_open",
    "xp_r15",
    "xp_r60",
    "pw_r15",
    "sp_5d",
    "sp_20d",
    "s_r15",
    "s_r60",
    "s_r_open",
    "xs_r15",
]
"""The stock's industry peers at the same minute: their average return (p_*), the stock minus them
(xp_*), the turnover-weighted peer return (pw_r15, where the leaders dominate), the slow spread of the
stock against its peers over 5 and 20 sessions (sp_*); and the stock's sector index (s_*, xs_*)."""
BREADTH = ["breadth_open", "breadth_15", "disp_15"]
"""Share of stocks up since the open and over the last 15 minutes, and the cross-sectional spread of
15-minute returns: whether the whole market moved or a few names did."""
REGIME = ["mkt_vol20", "mkt_vol60", "mkt_ret60", "vix"]
"""The market's regime as of yesterday: Nifty 50 realised volatility over 20 and 60 sessions, its
60-session return, India VIX. What a year stands for, measurable in a year the model never saw."""
OVERSEAS = [
    "spy_oc",
    "spy_cc",
    "qqq_oc",
    "qqq_cc",
    "spy_5d",
    "spy_range",
    "gold_cc",
    "silver_cc",
    "crude_cc",
]
"""The previous US session (open to close, close to close, five sessions, range) and the previous MCX
session, all closed hours before the NSE open."""
CLOCK = ["m", "m_left", "dow", "dom", "moy", "expiry", "dte", "after_holiday"]
FEATURES: list[str] = (
    INTRADAY
    + PRESSURE
    + SHAPE
    + DAILY
    + PARTICIPATION
    + ATTRIBUTES
    + PEERS
    + BREADTH
    + REGIME
    + OVERSEAS
    + CLOCK
)
CATEGORICAL = {"dow", "industry", "sector_idx", "moy"}
CONTEXT = {"i_r5", "i_r15", "i_r30", "i_r60", "i_r_open", *BREADTH, *REGIME, *OVERSEAS, *CLOCK}
"""Constant across stocks at a decision point: the market, breadth, regime, overseas and the clock.
They condition the others rather than rank stocks, so they have no cross-sectional IC and are not
rank-normalised in the Fit."""

SECTOR_INDEX: dict[str, str] = {
    "Financial Services": "NIFTY FIN SERVICE",
    "Information Technology": "NIFTY IT",
    "Automobile and Auto Components": "NIFTY AUTO",
    "Healthcare": "NIFTY HEALTHCARE",
    "Fast Moving Consumer Goods": "NIFTY FMCG",
    "Metals & Mining": "NIFTY METAL",
    "Realty": "NIFTY REALTY",
    "Oil Gas & Consumable Fuels": "NIFTY OIL AND GAS",
    "Power": "NIFTY ENERGY",
    "Consumer Durables": "NIFTY CONSUMPTION",
    "Consumer Services": "NIFTY CONSUMPTION",
    "Capital Goods": "NIFTY INFRA",
    "Construction": "NIFTY INFRA",
    "Construction Materials": "NIFTY INFRA",
    "Media Entertainment & Publication": "NIFTY MEDIA",
    "Chemicals": "NIFTY COMMODITIES",
}
"""NSE industry (the constituent files' label) to the Kite index whose minute bars stand for the
sector; industries without one, and stocks without an industry, fall back to NIFTY 500."""
SECTOR_FALLBACK = "NIFTY 500"
_SECTOR_CASE = (
    "case a.industry "
    + " ".join(f"when '{k}' then '{v}'" for k, v in SECTOR_INDEX.items())
    + f" else '{SECTOR_FALLBACK}' end"
)
_EXPIRY_DAY = (
    "(last_day(day) - (((isodow(last_day(day)) - (case when day < date "
    f"'{EXPIRY_WEEKDAY_CHANGE.isoformat()}' then 4 else 2 end) + 7) % 7)::int))"
)

ROWS_SQL = f"""
with
bars as (
    select distinct symbol, ts, ts::date as day, open, high, low, close, volume, close * volume as turnover
    from equity_1m
    where symbol not like 'NIFTY%' and symbol <> 'INDIA VIX' and ts::date between :start and :end
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
sidx as (  -- every sector and size index; the stock's own sector joins by name
    select symbol as sector_idx, ts,
           ln(close / lag(close, 15) over w) as s_r15, ln(close / lag(close, 60) over w) as s_r60,
           ln(close / first_value(open) over w) as s_r_open
    from (select distinct symbol, ts, ts::date as day, open, close from equity_1m
          where symbol like 'NIFTY%' and ts::date between :start and :end)
    window w as (partition by symbol, day order by ts)
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
           max(case when datediff('minute', day::timestamp + interval 9 hour + interval 15 minute, ts) < 15 then high end) over wd as or_high,
           min(case when datediff('minute', day::timestamp + interval 9 hour + interval 15 minute, ts) < 15 then low end) over wd as or_low,
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
           abs(r30) / nullif(sum(turnover) over w30, 0) * 1e7 as impact30,
           case when m >= 15 then (close - or_low) / nullif(or_high - or_low, 0) end as or_pos,
           case when m < 15 then null when close > or_high then 1 when close < or_low then -1 else 0 end as or_break
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
    select symbol, date, close, high, low, turnover, trades, deliv_pct,
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
           stddev_samp(dret) over (partition by symbol order by date rows between 59 preceding and current row) as vol_60d,
           close / nullif(max(high) over (partition by symbol order by date rows between 251 preceding and current row), 0) as dist_hi,
           close / nullif(min(low) over (partition by symbol order by date rows between 251 preceding and current row), 0) as dist_lo,
           turnover / nullif(quantile_cont(turnover, 0.5) over (partition by symbol order by date rows between 20 preceding and 1 preceding), 0) as turn_rel,
           ln(nullif(quantile_cont(turnover, 0.5) over (partition by symbol order by date rows between 19 preceding and current row), 0)) as log_turn20,
           ln(nullif(trades, 0)) as log_trades,
           ln(nullif(turnover / nullif(trades, 0), 0)) as avg_trade,
           trades / nullif(quantile_cont(trades, 0.5) over (partition by symbol order by date rows between 20 preceding and 1 preceding), 0) as trades_rel,
           ((close - low) - (high - close)) / nullif(high - low, 0) as clv_prev,
           ln(close) as log_px
    from d0 window w as (partition by symbol order by date)
),
i0 as (
    select date, close, ln(close / lag(close) over (order by date)) as iret
    from index_daily where index_name = '{INDEX_DAILY}'
),
i1 as (
    select date,
           ln(close / lag(close, 20) over (order by date)) as i_ret_20d,
           ln(close / lag(close, 60) over (order by date)) as mkt_ret60,
           stddev_samp(iret) over (order by date rows between 19 preceding and current row) as mkt_vol20,
           stddev_samp(iret) over (order by date rows between 59 preceding and current row) as mkt_vol60
    from i0
),
vix as (select date, close as vix from index_daily where index_name = 'India VIX'),
us as (
    select date,
           max(case when symbol = 'SPY' then ln(close / open) end) as spy_oc,
           max(case when symbol = 'SPY' then ln(close / lag_close) end) as spy_cc,
           max(case when symbol = 'QQQ' then ln(close / open) end) as qqq_oc,
           max(case when symbol = 'QQQ' then ln(close / lag_close) end) as qqq_cc,
           max(case when symbol = 'SPY' then ln(close / lag_close5) end) as spy_5d,
           max(case when symbol = 'SPY' then (high - low) / nullif(close, 0) end) as spy_range
    from (select symbol, date, open, high, low, close,
                 lag(close) over (partition by symbol order by date) as lag_close,
                 lag(close, 5) over (partition by symbol order by date) as lag_close5
          from us_equity where symbol in ('SPY', 'QQQ'))
    group by date
),
mcx as (
    select date,
           max(case when symbol = 'GOLD' then ln(close / lag_close) end) as gold_cc,
           max(case when symbol = 'SILVER' then ln(close / lag_close) end) as silver_cc,
           max(case when symbol = 'CRUDEOIL' then ln(close / lag_close) end) as crude_cc
    from (select symbol, date, close, lag(close) over (partition by symbol order by date) as lag_close
          from commodity_continuous where symbol in ('GOLD', 'SILVER', 'CRUDEOIL'))
    group by date
),
cons as (  -- one row per (symbol, snapshot day): membership and the industry label
    select symbol, as_of,
           any_value(industry) as industry,
           max(case when index = 'NIFTY 50' then 1 else 0 end) as in_n50,
           max(case when index = 'NIFTY 500' then 1 else 0 end) as in_n500,
           max(case when index = 'NIFTY MIDCAP 100' then 1 else 0 end) as in_mid100,
           max(case when index = 'NIFTY SMALLCAP 250' then 1 else 0 end) as in_small250
    from index_constituents group by symbol, as_of
),
cons_first as (select * from cons qualify row_number() over (partition by symbol order by as_of) = 1),
ticks as (select tradingsymbol as symbol, any_value(tick_size) as tick from kite_instruments where segment = 'NSE' group by 1),
days as (
    select day, lag(day) over (order by day) as prev_day from (select distinct day from bars)
),
attr as (  -- the stock's slow attributes on the day: the newest snapshot on or before it, else the first
    select b.symbol, b.day,
           coalesce(c.industry, f.industry) as industry,
           coalesce(c.in_n50, f.in_n50, 0) as in_n50, coalesce(c.in_n500, f.in_n500, 0) as in_n500,
           coalesce(c.in_mid100, f.in_mid100, 0) as in_mid100, coalesce(c.in_small250, f.in_small250, 0) as in_small250,
           t.tick
    from (select distinct symbol, day from bars) b
    asof left join cons c on b.symbol = c.symbol and b.day >= c.as_of
    left join cons_first f on b.symbol = f.symbol
    left join ticks t on t.symbol = b.symbol
),
r1 as (
    select b.*, i.i_r5, i.i_r15, i.i_r30, i.i_r60, i.i_r_open,
           d.ret_1d, d.d_close, d.ret_5d, d.ret_20d, d.vol_20d, d.vol_60d, d.dist_hi, d.dist_lo, d.turn_rel,
           d.deliv_pct, d.log_turn20, d.log_trades, d.avg_trade, d.trades_rel, d.clv_prev, d.log_px,
           ii.i_ret_20d, ii.mkt_ret60, ii.mkt_vol20, ii.mkt_vol60, v.vix,
           u.spy_oc, u.spy_cc, u.qqq_oc, u.qqq_cc, u.spy_5d, u.spy_range, x.gold_cc, x.silver_cc, x.crude_cc,
           a.industry, a.in_n50, a.in_n500, a.in_mid100, a.in_small250,
           a.tick / nullif(d.d_close, 0) * 1e4 as tick_bps,
           {_SECTOR_CASE} as sector_idx,
           dy.prev_day
    from b3 b
    left join idx i using (ts)
    asof left join d1 d on b.symbol = d.symbol and b.day > d.date
    asof left join i1 ii on b.day > ii.date
    asof left join vix v on b.day > v.date
    asof left join us u on b.day > u.date
    asof left join mcx x on b.day > x.date
    left join attr a on a.symbol = b.symbol and a.day = b.day
    left join days dy on dy.day = b.day
    where b.m between 0 and {SESSION_MINUTES - 1}  -- the regular session only: Muhurat evening bars run past it
),
r2 as (  -- peers at the same minute, breadth at the same minute
    select r.*,
           avg(r5) over wp as p_r5, avg(r15) over wp as p_r15, avg(r30) over wp as p_r30,
           avg(r60) over wp as p_r60, avg(r_open) over wp as p_r_open,
           sum(r15 * exp(log_turn20)) over wp / nullif(sum(exp(log_turn20)) over wp, 0) as pw_r15,
           ret_5d - avg(ret_5d) over wp as sp_5d, ret_20d - avg(ret_20d) over wp as sp_20d,
           avg(case when r_open > 0 then 1.0 else 0.0 end) over wt as breadth_open,
           avg(case when r15 > 0 then 1.0 else 0.0 end) over wt as breadth_15,
           stddev_samp(r15) over wt as disp_15,
           s.s_r15, s.s_r60, s.s_r_open
    from r1 r
    left join sidx s on s.sector_idx = r.sector_idx and s.ts = r.ts
    window wp as (partition by coalesce(r.industry, 'none'), r.ts), wt as (partition by r.ts)
),
rows_ as (
    select symbol, ts, day, close, entry, sig1,
           case when :horizon_minutes >= {SESSION_MINUTES} then day_close else exit_h end as exit_,  -- a session horizon is the close
           r1, r5, r15, r30, r60, r_open, range_pos, vwap_dev, rv30, rv_open, relvol5, relvol15, turn_liq,
           i_r5, i_r15, i_r30, i_r60, i_r_open, r15 - i_r15 as x_r15, r60 - i_r60 as x_r60,
           imb5, imb15, imb30, clv5, clv15, clv30, vwret5, vwret15, vwret30, up5, up15, up30, impact15, impact30, turn_accel,
           or_pos, or_break, clv_prev,
           ret_1d, day_open / nullif(d_close, 0) - 1 as gap, ret_5d, ret_20d, vol_20d, dist_hi, dist_lo, turn_rel, deliv_pct,
           ret_20d - i_ret_20d as rs_20,
           log_trades, avg_trade, trades_rel,
           industry, sector_idx, in_n50, in_n500, in_mid100, in_small250, log_px, tick_bps, log_turn20, vol_60d,
           p_r5, p_r15, p_r30, p_r60, p_r_open, r15 - p_r15 as xp_r15, r60 - p_r60 as xp_r60, pw_r15, sp_5d, sp_20d,
           s_r15, s_r60, s_r_open, r15 - s_r15 as xs_r15,
           breadth_open, breadth_15, disp_15,
           mkt_vol20, mkt_vol60, mkt_ret60, vix,
           spy_oc, spy_cc, qqq_oc, qqq_cc, spy_5d, spy_range, gold_cc, silver_cc, crude_cc,
           m, {SESSION_MINUTES} - m as m_left, isodow(day) as dow, dayofmonth(day) as dom, month(day) as moy,
           (day = {_EXPIRY_DAY})::int as expiry,
           greatest(0, {_EXPIRY_DAY} - day) as dte,
           (day - prev_day > case when isodow(day) = 1 then 3 else 1 end)::int as after_holiday
    from r2
    where m % :step_minutes = 0 and turn_liq >= :min_turnover
)
select *,
       sig1 * sqrt(least(:horizon_minutes, m_left - 1)) as scale,
       ln(exit_ / entry) as r,
       ln(exit_ / entry) / (sig1 * sqrt(least(:horizon_minutes, m_left - 1))) as y
from rows_
where entry is not null and exit_ is not null and sig1 > 0
order by ts, symbol
"""


# ---- the daily grain -------------------------------------------------------------------------------

DAILY_PARAMS: dict[str, Any] = {
    "horizon_sessions": 1,
    "vol_sessions": 20,
    "min_turnover_day": 1e7,
    "start": "2021-09-01",
    "end": "2030-12-31",
}
"""One row per (symbol, session t), decided at t's close, entered at the next session's open, exited at
the close of the ``horizon_sessions``-th session after t (1 = the entry session's own close). The 1-session and 21-session Fits run on
these rows: five years of every EQ stock, where the minute rows have three years of the liquid ones.
A row needs the stock's 20-session median turnover at or above ``min_turnover_day`` rupees."""

D_PRICE = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "vol_60d",
    "dist_hi",
    "dist_lo",
    "turn_rel",
    "deliv_pct",
    "rs_20",
    "rs_60",
    "gap_d",
    "range_pos_d",
    "clv_d",
]
D_SHAPE = ["rv_day", "vwap_dev_close", "r_first60", "r_last60", "relvol_day"]
"""The session's own shape from its minute bars where held (NSE, 2023-10 on, liquid names; null
before and for the US): realised volatility, the close against the day's VWAP, the first and last
hour's returns, the day's volume against the stock's 20-session norm."""
D_ATTRIBUTES = [
    "industry",
    "sector_idx",
    "in_n50",
    "in_n500",
    "in_mid100",
    "in_small250",
    "log_px",
    "tick_bps",
    "log_turn20",
]
"""For the US the four membership flags are size buckets from the screener's market cap: mega
(100 B+), mega or large (10 B+), mid (2 to 10 B), small (300 M to 2 B)."""
D_PEERS = ["p_ret_1d", "p_ret_5d", "p_ret_20d", "pw_ret_20d", "sp_5d", "sp_20d", "sp_60d"]
D_SECTOR = ["s_ret_1d", "s_ret_5d", "s_ret_20d", "xs_ret_20d"]
"""The stock's sector index (NSE: the Nifty sector index in index_daily; US: the SPDR sector ETF)."""
D_BREADTH = ["breadth_d", "disp_d"]
D_REGIME = ["mkt_ret_1d", "mkt_ret60", "mkt_vol20", "mkt_vol60", "vix"]
D_OVERSEAS = {
    "nse": [
        "spy_oc",
        "spy_cc",
        "qqq_oc",
        "qqq_cc",
        "spy_5d",
        "spy_range",
        "gold_cc",
        "silver_cc",
        "crude_cc",
    ],
    "us": ["nifty_cc", "nifty_5d", "qqq_cc", "iwm_cc", "gld_cc", "slv_cc", "uso_cc"],
}
"""What closed before the entry, elsewhere: for NSE rows the US session dated t and the MCX session
dated t; for US rows the Nifty session dated the entry day (it closes at 06:00 ET, before the 09:30
open), and the previous session of the size and commodity ETFs."""
D_CLOCK = ["dow", "dom", "moy", "dte", "after_holiday"]


def daily_features(market: str = "nse") -> list[str]:
    """The daily rows' features for a market; the US has no trades count, delivery or minute bars."""
    price = D_PRICE if market == "nse" else [f for f in D_PRICE if f != "deliv_pct"]
    shape = D_SHAPE if market == "nse" else []
    participation = PARTICIPATION if market == "nse" else []
    return (
        price
        + shape
        + participation
        + D_ATTRIBUTES
        + D_PEERS
        + D_SECTOR
        + D_BREADTH
        + D_REGIME
        + D_OVERSEAS[market]
        + D_CLOCK
    )


def daily_context(market: str = "nse") -> set[str]:
    return {*D_BREADTH, *D_REGIME, *D_OVERSEAS[market], *D_CLOCK}


DAILY_FEATURES: list[str] = daily_features("nse")
DAILY_CONTEXT = daily_context("nse")

NSE_SECTOR_DAILY: dict[str, str] = {
    "Financial Services": "Nifty Financial Services",
    "Information Technology": "Nifty IT",
    "Automobile and Auto Components": "Nifty Auto",
    "Healthcare": "Nifty Healthcare Index",
    "Fast Moving Consumer Goods": "Nifty FMCG",
    "Metals & Mining": "Nifty Metal",
    "Realty": "Nifty Realty",
    "Oil Gas & Consumable Fuels": "Nifty Oil & Gas",
    "Power": "Nifty Energy",
    "Consumer Durables": "Nifty Consumer Durables",
    "Consumer Services": "Nifty India Consumption",
    "Capital Goods": "Nifty Infrastructure",
    "Construction": "Nifty Infrastructure",
    "Construction Materials": "Nifty Infrastructure",
    "Media Entertainment & Publication": "Nifty Media",
    "Chemicals": "Nifty Chemicals",
}
"""NSE industry to the sector index in index_daily (the names NSE's index files use)."""
US_SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Finance": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Telecommunications": "XLC",
}
"""Nasdaq screener sector to the SPDR sector ETF whose daily bars stand for the sector."""


def _case(col: str, mapping: dict[str, str], fallback: str) -> str:
    return (
        f"case {col} "
        + " ".join(f"when '{k}' then '{v}'" for k, v in mapping.items())
        + f" else '{fallback}' end"
    )


def _third_friday(col: str) -> str:
    """US monthly options expire on the third Friday."""
    first = f"date_trunc('month', {col})::date"
    return f"({first} + ((5 - isodow({first}) + 7) % 7 + 14)::int)"


def _nse_expiry(col: str) -> str:
    return (
        f"(last_day({col}) - (((isodow(last_day({col})) - (case when {col} < date "
        f"'{EXPIRY_WEEKDAY_CHANGE.isoformat()}' then 4 else 2 end) + 7) % 7)::int))"
    )


_SHAPE_NONE = """shape2 as (
    select null::varchar as symbol, null::date as day, null::double as rv_day, null::double as vwap_dev_close,
           null::double as r_first60, null::double as r_last60, null::double as relvol_day where false
)"""

_SHAPE_NSE = f"""shape as (  -- the session's shape from its minute bars, where held
    select symbol, day,
           stddev_samp(ret1) as rv_day,
           max(close) filter (where rn_desc = 1) / nullif(sum(close * volume) / nullif(sum(volume), 0), 0) - 1 as vwap_dev_close,
           ln(max(close) filter (where m = 59) / nullif(max(open) filter (where m = 0), 0)) as r_first60,
           ln(max(close) filter (where rn_desc = 1) / nullif(max(close) filter (where m = 314), 0)) as r_last60,
           sum(volume) as vol_day
    from (select symbol, ts::date as day, close, open, volume,
                 datediff('minute', ts::date::timestamp + interval 9 hour + interval 15 minute, ts) as m,
                 ln(close / lag(close) over (partition by symbol, ts::date order by ts)) as ret1,
                 row_number() over (partition by symbol, ts::date order by ts desc) as rn_desc
          from (select distinct symbol, ts, open, close, volume from equity_1m
                where symbol not like 'NIFTY%' and symbol <> 'INDIA VIX' and ts::date between :start and :end))
    where m between 0 and {SESSION_MINUTES - 1}
    group by symbol, day
),
shape2 as (
    select *, vol_day / nullif(avg(vol_day) over (partition by symbol order by day rows between 20 preceding and 1 preceding), 0) as relvol_day
    from shape
)"""

DAILY_MARKETS: dict[str, dict[str, str]] = {
    "nse": {
        "equity": """select symbol, date, open, high, low, close, turnover, trades, deliv_pct
    from equity where date between :start::date - interval 400 day and :end::date + interval 60 day
    qualify row_number() over (partition by symbol, date order by series) = 1""",
        "shape": _SHAPE_NSE,
        "index": f"select date, close from index_daily where index_name = '{INDEX_DAILY}'",
        "vix": "select date, close as vix from index_daily where index_name = 'India VIX'",
        "cons": """select symbol, as_of, any_value(industry) as industry,
           max(case when index = 'NIFTY 50' then 1 else 0 end) as in_n50,
           max(case when index = 'NIFTY 500' then 1 else 0 end) as in_n500,
           max(case when index = 'NIFTY MIDCAP 100' then 1 else 0 end) as in_mid100,
           max(case when index = 'NIFTY SMALLCAP 250' then 1 else 0 end) as in_small250
    from index_constituents group by symbol, as_of""",
        "ticks": "select tradingsymbol as symbol, any_value(tick_size) as tick from kite_instruments where segment = 'NSE' group by 1",
        "sector_src": "select date, index_name as name, close from index_daily",
        "sector_case": _case("r.industry", NSE_SECTOR_DAILY, INDEX_DAILY),
        "overseas_ctes": """us as (
    select date,
           max(case when symbol = 'SPY' then ln(close / open) end) as spy_oc,
           max(case when symbol = 'SPY' then ln(close / lag_close) end) as spy_cc,
           max(case when symbol = 'QQQ' then ln(close / open) end) as qqq_oc,
           max(case when symbol = 'QQQ' then ln(close / lag_close) end) as qqq_cc,
           max(case when symbol = 'SPY' then ln(close / lag_close5) end) as spy_5d,
           max(case when symbol = 'SPY' then (high - low) / nullif(close, 0) end) as spy_range
    from (select symbol, date, open, high, low, close,
                 lag(close) over (partition by symbol order by date) as lag_close,
                 lag(close, 5) over (partition by symbol order by date) as lag_close5
          from us_equity where symbol in ('SPY', 'QQQ'))
    group by date
),
mcx as (
    select date,
           max(case when symbol = 'GOLD' then ln(close / lag_close) end) as gold_cc,
           max(case when symbol = 'SILVER' then ln(close / lag_close) end) as silver_cc,
           max(case when symbol = 'CRUDEOIL' then ln(close / lag_close) end) as crude_cc
    from (select symbol, date, close, lag(close) over (partition by symbol order by date) as lag_close
          from commodity_continuous where symbol in ('GOLD', 'SILVER', 'CRUDEOIL'))
    group by date
)""",
        "overseas_join": """    asof left join us u on d.date >= u.date       -- the US session dated t closes before the entry at t+1's open
    asof left join mcx x on d.date >= x.date""",
        "overseas_cols": "u.spy_oc, u.spy_cc, u.qqq_oc, u.qqq_cc, u.spy_5d, u.spy_range, x.gold_cc, x.silver_cc, x.crude_cc",
        "overseas_out": "spy_oc, spy_cc, qqq_oc, qqq_cc, spy_5d, spy_range, gold_cc, silver_cc, crude_cc",
        "expiry": _nse_expiry("entry_date"),
    },
    "us": {
        "equity": """select symbol, date, open, high, low, close, close * volume as turnover, null::bigint as trades, null::double as deliv_pct
    from us_equity
    where symbol not like '^%' and symbol not in ('SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'USO', 'XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC')
      and date between :start::date - interval 400 day and :end::date + interval 60 day
      and open > 0 and close > 0""",
        "shape": _SHAPE_NONE,
        "index": "select date, close from us_equity where symbol = 'SPY'",
        "vix": "select date, close as vix from us_equity where symbol = '^VIX'",
        "cons": """select symbol, as_of, any_value(sector) as industry,
           max(case when cap_bucket = 'mega' then 1 else 0 end) as in_n50,
           max(case when cap_bucket in ('mega', 'large') then 1 else 0 end) as in_n500,
           max(case when cap_bucket = 'mid' then 1 else 0 end) as in_mid100,
           max(case when cap_bucket = 'small' then 1 else 0 end) as in_small250
    from us_sectors group by symbol, as_of""",
        "ticks": "select distinct symbol, 0.01 as tick from us_equity",
        "sector_src": "select date, symbol as name, close from us_equity where symbol like 'XL%' or symbol = 'SPY'",
        "sector_case": _case("r.industry", US_SECTOR_ETF, "SPY"),
        "overseas_ctes": """nifty as (
    select date, ln(close / lag(close) over (order by date)) as nifty_cc,
           ln(close / lag(close, 5) over (order by date)) as nifty_5d
    from index_daily where index_name = 'Nifty 50'
),
etfs as (
    select date,
           max(case when symbol = 'QQQ' then ln(close / lag_close) end) as qqq_cc,
           max(case when symbol = 'IWM' then ln(close / lag_close) end) as iwm_cc,
           max(case when symbol = 'GLD' then ln(close / lag_close) end) as gld_cc,
           max(case when symbol = 'SLV' then ln(close / lag_close) end) as slv_cc,
           max(case when symbol = 'USO' then ln(close / lag_close) end) as uso_cc
    from (select symbol, date, close, lag(close) over (partition by symbol order by date) as lag_close
          from us_equity where symbol in ('QQQ', 'IWM', 'GLD', 'SLV', 'USO'))
    group by date
)""",
        "overseas_join": """    asof left join nifty n on d.entry_date >= n.date  -- the Nifty session of the entry day closes at 06:00 ET
    asof left join etfs x on d.date >= x.date""",
        "overseas_cols": "n.nifty_cc, n.nifty_5d, x.qqq_cc, x.iwm_cc, x.gld_cc, x.slv_cc, x.uso_cc",
        "overseas_out": "nifty_cc, nifty_5d, qqq_cc, iwm_cc, gld_cc, slv_cc, uso_cc",
        "expiry": _third_friday("entry_date"),
    },
}

_DAILY_TEMPLATE = """
with
e as (
    __EQUITY__
),
d1 as (
    select *,
           ln(close / lag(close) over w) as ret_1d,
           ln(close / lag(close, 5) over w) as ret_5d,
           ln(close / lag(close, 20) over w) as ret_20d,
           ln(close / lag(close, 60) over w) as ret_60d,
           open / nullif(lag(close) over w, 0) - 1 as gap_d,
           (close - low) / nullif(high - low, 0) as range_pos_d,
           ((close - low) - (high - close)) / nullif(high - low, 0) as clv_d,
           ln(nullif(trades, 0)) as log_trades,
           ln(nullif(turnover / nullif(trades, 0), 0)) as avg_trade,
           ln(close) as log_px,
           lead(open) over w as entry,
           lead(close, :horizon_sessions) over w as exit_,
           lead(date) over w as entry_date
    from e window w as (partition by symbol order by date)
),
d2 as (
    select *,
           stddev_samp(ret_1d) over (partition by symbol order by date rows between 19 preceding and current row) as vol_20d,
           stddev_samp(ret_1d) over (partition by symbol order by date rows between 59 preceding and current row) as vol_60d,
           stddev_samp(ret_1d) over (partition by symbol order by date rows between :vol_sessions - 1 preceding and current row) as sig_d,
           close / nullif(max(high) over (partition by symbol order by date rows between 251 preceding and current row), 0) as dist_hi,
           close / nullif(min(low) over (partition by symbol order by date rows between 251 preceding and current row), 0) as dist_lo,
           turnover / nullif(quantile_cont(turnover, 0.5) over (partition by symbol order by date rows between 20 preceding and 1 preceding), 0) as turn_rel,
           quantile_cont(turnover, 0.5) over (partition by symbol order by date rows between 19 preceding and current row) as turn20,
           trades / nullif(quantile_cont(trades, 0.5) over (partition by symbol order by date rows between 20 preceding and 1 preceding), 0) as trades_rel
    from d1
),
__SHAPE__,
i0 as (select date, close, ln(close / lag(close) over (order by date)) as iret from (__INDEX__)),
i1 as (
    select date, iret as mkt_ret_1d,
           ln(close / lag(close, 20) over (order by date)) as i_ret_20d,
           ln(close / lag(close, 60) over (order by date)) as mkt_ret60,
           stddev_samp(iret) over (order by date rows between 19 preceding and current row) as mkt_vol20,
           stddev_samp(iret) over (order by date rows between 59 preceding and current row) as mkt_vol60
    from i0
),
vix as (__VIX__),
sec as (
    select date, name,
           ln(close / lag(close) over w) as s_ret_1d,
           ln(close / lag(close, 5) over w) as s_ret_5d,
           ln(close / lag(close, 20) over w) as s_ret_20d
    from (__SECTOR_SRC__) window w as (partition by name order by date)
),
__OVERSEAS_CTES__,
cons as (
    __CONS__
),
cons_first as (select * from cons qualify row_number() over (partition by symbol order by as_of) = 1),
ticks as (__TICKS__),
sessions as (select date, lag(date) over (order by date) as prev_date from (select distinct date from e)),
r1 as (
    select d.*,
           coalesce(c.industry, f.industry) as industry,
           coalesce(c.in_n50, f.in_n50, 0) as in_n50, coalesce(c.in_n500, f.in_n500, 0) as in_n500,
           coalesce(c.in_mid100, f.in_mid100, 0) as in_mid100, coalesce(c.in_small250, f.in_small250, 0) as in_small250,
           t.tick / nullif(d.close, 0) * 1e4 as tick_bps,
           ln(nullif(d.turn20, 0)) as log_turn20,
           s.rv_day, s.vwap_dev_close, s.r_first60, s.r_last60, s.relvol_day,
           ii.mkt_ret_1d, ii.i_ret_20d, ii.mkt_ret60, ii.mkt_vol20, ii.mkt_vol60, v.vix,
           __OVERSEAS_COLS__,
           ss.prev_date
    from d2 d
    asof left join cons c on d.symbol = c.symbol and d.date >= c.as_of
    left join cons_first f on d.symbol = f.symbol
    left join ticks t on t.symbol = d.symbol
    left join shape2 s on s.symbol = d.symbol and s.day = d.date
    left join i1 ii on ii.date = d.date
    left join vix v on v.date = d.date
__OVERSEAS_JOIN__
    left join sessions ss on ss.date = d.date
    where d.date between :start and :end and d.turn20 >= :min_turnover_day
),
r2 as (
    select r.*,
           __SECTOR_CASE__ as sector_idx,
           avg(ret_1d) over wp as p_ret_1d, avg(ret_5d) over wp as p_ret_5d, avg(ret_20d) over wp as p_ret_20d,
           sum(ret_20d * turn20) over wp / nullif(sum(turn20) over wp, 0) as pw_ret_20d,
           ret_5d - avg(ret_5d) over wp as sp_5d, ret_20d - avg(ret_20d) over wp as sp_20d, ret_60d - avg(ret_60d) over wp as sp_60d,
           avg(case when ret_1d > 0 then 1.0 else 0.0 end) over wt as breadth_d,
           stddev_samp(ret_1d) over wt as disp_d
    from r1 r
    window wp as (partition by coalesce(r.industry, 'none'), r.date), wt as (partition by r.date)
),
r3 as (
    select r.*, sx.s_ret_1d, sx.s_ret_5d, sx.s_ret_20d, r.ret_20d - sx.s_ret_20d as xs_ret_20d
    from r2 r left join sec sx on sx.name = r.sector_idx and sx.date = r.date
)
select symbol, date::timestamp + interval 15 hour + interval 30 minute as ts, date as day, close, entry, entry_date, exit_, sig_d,
       ret_1d, ret_5d, ret_20d, ret_60d, vol_20d, vol_60d, dist_hi, dist_lo, turn_rel, deliv_pct,
       ret_20d - i_ret_20d as rs_20, ret_60d - mkt_ret60 as rs_60, gap_d, range_pos_d, clv_d,
       rv_day, vwap_dev_close, r_first60, r_last60, relvol_day,
       log_trades, avg_trade, trades_rel,
       industry, sector_idx, in_n50, in_n500, in_mid100, in_small250, log_px, tick_bps, log_turn20,
       p_ret_1d, p_ret_5d, p_ret_20d, pw_ret_20d, sp_5d, sp_20d, sp_60d,
       s_ret_1d, s_ret_5d, s_ret_20d, xs_ret_20d,
       breadth_d, disp_d,
       mkt_ret_1d, mkt_ret60, mkt_vol20, mkt_vol60, vix,
       __OVERSEAS_OUT__,
       isodow(entry_date) as dow, dayofmonth(entry_date) as dom, month(entry_date) as moy,
       greatest(0, __EXPIRY__ - entry_date) as dte,
       (entry_date - date > case when isodow(entry_date) = 1 then 3 else 1 end)::int as after_holiday,
       sig_d * sqrt(:horizon_sessions) as scale,
       ln(exit_ / entry) as r,
       ln(exit_ / entry) / (sig_d * sqrt(:horizon_sessions)) as y
from r3
where entry is not null and exit_ is not null and sig_d > 0 and entry > 0 and exit_ > 0
order by date, symbol
"""


def daily_sql(market: str = "nse") -> str:
    """The daily rows query for a market, from one template and the market's sources."""
    cfg = DAILY_MARKETS[market]
    out = _DAILY_TEMPLATE
    for key, text in cfg.items():
        out = out.replace(f"__{key.upper()}__", text)
    return out


DAILY_SQL = daily_sql("nse")
US_DAILY_SQL = daily_sql("us")


def daily_fit_definition(
    horizon_sessions: int, market: str = "nse", **overrides: Any
) -> dict[str, Any]:
    """The Fit body for a daily-grain horizon of a market: its query, params, features, the target."""
    params = {**DAILY_PARAMS, "horizon_sessions": horizon_sessions, **overrides}
    if market == "us" and "start" not in overrides:
        params["start"] = "2015-01-01"
    return {
        "name": f"forecast {horizon_sessions}d {market}",
        "description": (
            f"{market.upper()}: expected volatility-scaled log return over {horizon_sessions} "
            "session(s), decided at the close, entered at the next open; docs/finance-forecast.md"
        ),
        "query": daily_sql(market),
        "params": params,
        "key": KEY,
        "features": [
            {"name": f, "kind": "categorical" if f in CATEGORICAL else "numeric"}
            for f in daily_features(market)
        ],
        "target": {"columns": ["y"], "task": "regression"},
    }


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


EVALUATION_METRICS = [
    "ic",
    "ic_se",
    "ic_points",
    "calibration_slope",
    "calibration_monotone",
    "r2",
    "net_return_raw",
    "trades_raw",
]


def evaluation_params(
    threshold: float, cost_bps: float = 10.0, by: str | None = "month"
) -> dict[str, Any]:
    """Params for an Evaluation of a forecast InferRun: IC within ts, raw net return where the
    predicted raw move (pred_y × scale) clears the threshold, grouped by month."""
    out: dict[str, Any] = {
        "group": "ts",
        "threshold": threshold,
        "cost_bps": cost_bps,
        "return_column": "r",
        "scale_column": "scale",
    }
    if by:
        out["by"] = by
    return out


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


def materialise(
    con: duckdb.DuckDBPyConnection, params: dict[str, Any], out: Path, sql: str = ROWS_SQL
) -> int:
    """Run the rows query once and write the frame to Parquet; the screen reads three columns at a
    time from it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    data.relation(con, sql, params).write_parquet(out.as_posix(), compression="zstd")
    return int(con.sql(f"select count(*) from read_parquet('{out.as_posix()}')").fetchall()[0][0])


def screen(
    con: duckdb.DuckDBPyConnection,
    rows: Path,
    features: list[str] | None = None,
    all_features: list[str] = FEATURES,
    context: set[str] = CONTEXT,
) -> list[IC]:
    """Per feature, per year: the Spearman correlation with y within each decision point, averaged over
    decision points, with the standard error of that mean."""
    out: list[IC] = []
    for f in features or [f for f in all_features if f not in context and f not in CATEGORICAL]:
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


def screen_table(
    ics: list[IC], all_features: list[str] = FEATURES, context: set[str] = CONTEXT
) -> str:
    """Markdown: one row per feature, one column per year with IC and its t-statistic, for the design doc."""
    years = sorted({i.year for i in ics})
    by: dict[str, dict[int, IC]] = {}
    for i in ics:
        by.setdefault(i.feature, {})[i.year] = i
    lines = [
        "| feature | " + " | ".join(str(y) for y in years) + " | verdict |",
        "|---|" + "---|" * (len(years) + 1),
    ]
    for f in all_features:
        if f in context or f in CATEGORICAL:
            kind = "context" if f in context else "categorical"
            lines.append(f"| {f} | " + " | ".join("" for _ in years) + f" | {kind} |")
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
    ap.add_argument("--grain", choices=["minute", "daily"], default="minute")
    ap.add_argument("--market", choices=["nse", "us"], default="nse", help="daily grain only")
    ap.add_argument(
        "--horizon",
        type=int,
        default=PARAMS["horizon_minutes"],
        help="minutes; sessions for --grain daily",
    )
    ap.add_argument("--min-turnover-day", type=float, default=DAILY_PARAMS["min_turnover_day"])
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
    if a.grain == "daily":
        sql, all_features, context = (
            daily_sql(a.market),
            daily_features(a.market),
            daily_context(a.market),
        )
        start = a.start if a.start != PARAMS["start"] else DAILY_PARAMS["start"]
        params: dict[str, Any] = {
            "horizon_sessions": a.horizon,
            "vol_sessions": DAILY_PARAMS["vol_sessions"],
            "min_turnover_day": a.min_turnover_day,
            "start": start,
            "end": a.end,
        }
        tag = f"daily_{a.market}_h{a.horizon}_t{int(a.min_turnover_day)}_{start}_{a.end}"
    else:
        sql, all_features, context = ROWS_SQL, FEATURES, CONTEXT
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
    tmp = data.work_dir() / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"set threads={a.threads}; set memory_limit='{a.memory}'; set temp_directory='{tmp.as_posix()}'"
    )
    import time

    t0 = time.time()
    n = materialise(con, params, out, sql)
    print(f"rows: {n:,} in {time.time() - t0:.0f}s -> {out}")
    t0 = time.time()
    ics = screen(con, out, all_features=all_features, context=context)
    print(f"screen in {time.time() - t0:.0f}s\n")
    print(screen_table(ics, all_features, context))


if __name__ == "__main__":
    main()
