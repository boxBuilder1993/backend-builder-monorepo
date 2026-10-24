"""The session study: for every instrument of a market, the return of holding overnight (close to the
next open), of being short in the daytime (open to close, sign flipped), and of both, per year, with
hit rates and consistency. Pure SQL over the daily tables, so it reaches back to 2021 for NSE and to
the 1990s for the US; the minute-bar version (15:14 to 09:29) is the same shape over three years.

Costs are not in the table: they depend on the instrument and the size, and the write-up applies
them (docs/finance-forecast.md, the session study). Variants an instrument cannot trade (a short
on an index without futures, a delivery-taxed overnight on a small account) are a matter of the
write-up too; the table reports what the prices did.
"""

# ruff: noqa: E501  (the SQL reads better unwrapped)

from __future__ import annotations

from typing import Any

US_CONTEXT = (
    "'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'USO', 'XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', "
    "'XLB', 'XLU', 'XLRE', 'XLC'"
)
NSE_INDICES = (
    "'Nifty 50', 'Nifty Bank', 'Nifty Midcap 100', 'Nifty Smallcap 250', 'Nifty 500', "
    "'Nifty Next 50', 'Nifty IT', 'Nifty Financial Services'"
)

MARKETS: dict[str, dict[str, str]] = {
    "nse": {
        "stocks": """select symbol as instrument, 'stock' as kind, date as d, open, close, turnover
    from equity where series = 'EQ' and open > 0 and close > 0
    qualify row_number() over (partition by symbol, date order by series) = 1""",
        "indices": f"""select index_name as instrument, 'index' as kind, date as d, open, close, null::double as turnover
    from index_daily where open > 0 and close > 0 and index_name in ({NSE_INDICES})""",
    },
    "us": {
        "stocks": f"""select symbol as instrument, 'stock' as kind, date as d, open, close, close * volume as turnover
    from us_equity where open > 0 and close > 0 and symbol not like '^%' and symbol not in ({US_CONTEXT})""",
        "indices": """select symbol as instrument, 'index' as kind, date as d, open, close, null::double as turnover
    from us_equity where open > 0 and close > 0 and symbol in ('SPY', 'QQQ', 'IWM')""",
    },
}

STUDY_SQL = """
with px as (
    __STOCKS__
    union all
    __INDICES__
),
liq as (select instrument from px where kind = 'stock' and d >= :liq_from group by 1
        having median(turnover) >= :min_turnover),
u as (select p.* from px p where p.kind = 'index' or p.instrument in (select instrument from liq)),
s as (
    select instrument, kind, d, year(d) as y,
           ln(lead(open) over w / close) as overnight,
           -ln(close / open) as short_day,
           ln(lead(close) over w / close) as buy_hold,
           lead(d) over w as nd
    from u where d between :start and :end
    window w as (partition by instrument order by d)
),
r as (select *, overnight + short_day as combined from s where overnight is not null and nd - d <= 5),
per_year as (
    select instrument, kind, y, count(*) as sessions,
           sum(overnight) as overnight_total, avg(overnight) as overnight_mean,
           avg(case when overnight > 0 then 1.0 else 0.0 end) as overnight_hit,
           sum(short_day) as short_day_total, avg(short_day) as short_day_mean,
           avg(case when short_day > 0 then 1.0 else 0.0 end) as short_day_hit,
           sum(combined) as combined_total, avg(combined) as combined_mean,
           avg(case when combined > 0 then 1.0 else 0.0 end) as combined_hit,
           stddev_samp(combined) as combined_sd,
           sum(buy_hold) as buy_hold_total
    from r group by 1, 2, 3 having count(*) >= :min_sessions
)
select * from per_year order by kind desc, instrument, y
"""


def study_sql(market: str) -> str:
    cfg = MARKETS[market]
    return STUDY_SQL.replace("__STOCKS__", cfg["stocks"]).replace("__INDICES__", cfg["indices"])


DEFAULTS: dict[str, dict[str, Any]] = {
    "nse": {
        "start": "2021-09-01",
        "end": "2030-12-31",
        "liq_from": "2024-01-01",
        "min_turnover": 5e7,
        "min_sessions": 60,
    },
    "us": {
        "start": "2015-01-01",
        "end": "2030-12-31",
        "liq_from": "2024-01-01",
        "min_turnover": 5e6,
        "min_sessions": 60,
    },
}
"""The NSE liquidity floor is 5 crore a day (median since 2024), the US 5 million dollars."""

VARIANTS = ("overnight", "short_day", "combined")


def summary_table(
    rows: list[dict[str, Any]], variant: str, top: int = 15, min_years: int = 3
) -> str:
    """Markdown: the instruments with the most consistent variant, ranked by the worst full year's
    total (a full year has at least 150 sessions), one line each with the per-year totals."""
    by: dict[str, dict[int, dict[str, Any]]] = {}
    kinds: dict[str, str] = {}
    for r in rows:
        by.setdefault(str(r["instrument"]), {})[int(r["y"])] = r
        kinds[str(r["instrument"])] = str(r["kind"])
    years = sorted({int(r["y"]) for r in rows})
    ranked: list[tuple[float, float, str]] = []
    for inst, per in by.items():
        full = [per[y] for y in years if y in per and int(per[y]["sessions"]) >= 150]
        if len(full) < min_years:
            continue
        totals = [float(f[f"{variant}_total"]) for f in full]
        ranked.append((min(totals), sum(totals) / len(totals), inst))
    ranked.sort(reverse=True)
    head = f"| {variant}: instrument | kind | " + " | ".join(str(y) for y in years)
    lines = [head + " | worst year | mean year |", "|---|---|" + "---|" * (len(years) + 2)]
    for worst, mean, inst in ranked[:top]:
        cells = [
            f"{float(by[inst][y][f'{variant}_total']) * 100:+.0f}%" if y in by[inst] else ""
            for y in years
        ]
        lines.append(
            f"| {inst} | {kinds[inst]} | "
            + " | ".join(cells)
            + f" | {worst * 100:+.0f}% | {mean * 100:+.0f}% |"
        )
    return "\n".join(lines)
