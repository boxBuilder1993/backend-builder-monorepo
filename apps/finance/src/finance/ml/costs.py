"""What a trade costs at the broker: one schedule per segment, charges per executed order in rupees.

The figures are Zerodha's published charges as pasted by the owner on 2026-10-25 (equity delivery,
equity intraday, F&O futures; options are not traded here). The depository (DP) charge is not on
that page: it is Zerodha's published CDSL charge, ₹15.34 plus GST, levied once per stock per day on
which that stock is sold from the demat account, whatever the quantity. It is an assumption to
confirm against a contract note. See docs/finance-portfolio.md §8.
"""

from __future__ import annotations

from dataclasses import dataclass

CRORE = 1e7


@dataclass(frozen=True)
class Schedule:
    name: str
    brokerage_pct: float
    """Brokerage as a fraction of the order value, capped at ``brokerage_cap`` rupees per order."""
    brokerage_cap: float
    stt_buy: float
    stt_sell: float
    exchange: float
    """NSE transaction charge, a fraction of the order value, both sides."""
    sebi_per_crore: float
    stamp_buy: float
    gst: float
    """On brokerage + exchange + SEBI charges."""
    dp_per_sale: float
    """Rupees per stock per sale day, GST included; zero where shares never leave the demat."""


DELIVERY = Schedule(
    name="zerodha_2026_delivery",
    brokerage_pct=0.0,
    brokerage_cap=0.0,
    stt_buy=0.001,
    stt_sell=0.001,
    exchange=0.0000307,
    sebi_per_crore=10.0,
    stamp_buy=0.00015,
    gst=0.18,
    dp_per_sale=15.34 * 1.18,
)
INTRADAY = Schedule(
    name="zerodha_2026_intraday",
    brokerage_pct=0.0003,
    brokerage_cap=20.0,
    stt_buy=0.0,
    stt_sell=0.00025,
    exchange=0.0000307,
    sebi_per_crore=10.0,
    stamp_buy=0.00003,
    gst=0.18,
    dp_per_sale=0.0,
)
FUTURES = Schedule(
    name="zerodha_2026_futures",
    brokerage_pct=0.0003,
    brokerage_cap=20.0,
    stt_buy=0.0,
    stt_sell=0.0005,
    exchange=0.0000183,
    sebi_per_crore=10.0,
    stamp_buy=0.00002,
    gst=0.18,
    dp_per_sale=0.0,
)
SCHEDULES: dict[str, Schedule] = {s.name: s for s in (DELIVERY, INTRADAY, FUTURES)}


def order_charges(schedule: Schedule, side: str, value: float) -> float:
    """Rupees charged on one executed order of ``value`` rupees, ``side`` "buy" or "sell", without
    the DP charge (which is per stock per sale day, not per order; see ``dp_charge``)."""
    if value <= 0:
        return 0.0
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be buy or sell, not {side!r}")
    brokerage = min(schedule.brokerage_cap, schedule.brokerage_pct * value)
    exchange = schedule.exchange * value
    sebi = schedule.sebi_per_crore * value / CRORE
    stt = (schedule.stt_buy if side == "buy" else schedule.stt_sell) * value
    stamp = schedule.stamp_buy * value if side == "buy" else 0.0
    return brokerage + exchange + sebi + stt + stamp + schedule.gst * (brokerage + exchange + sebi)


def dp_charge(schedule: Schedule) -> float:
    """Rupees for one stock sold on one day from the demat account."""
    return schedule.dp_per_sale


def round_trip_bps(schedule: Schedule, value: float) -> float:
    """Total charges of buying and selling ``value`` rupees once, in basis points of ``value``."""
    total = order_charges(schedule, "buy", value) + order_charges(schedule, "sell", value)
    return (total + dp_charge(schedule)) / value * 1e4
