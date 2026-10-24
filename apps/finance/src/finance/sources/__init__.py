"""The legs: runtime-free functions that fetch what is missing, write files and report back.

A leg takes the data directory, today's date and a view of the ledger, and returns a ``LegResult``.
Nothing here knows about reactions, rows or the store. Each leg lives in its own module; this file
is the registry and the shared types. Legs not yet written return ``LegResult.not_implemented()``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Written:
    """A file the leg wrote or found complete."""

    path: str  # relative to the data dir, posix
    kind: str  # raw | parquet | reference
    source: str
    day: date | None
    bytes: int
    sha256: str | None = None


@dataclass(frozen=True)
class NoData:
    """A (source, day) the source says has no file: holiday, weekend, not yet published."""

    source: str
    day: date


@dataclass(frozen=True)
class Ledger:
    """What the app already knows, read from DataFile rows before the leg runs."""

    ok: frozenset[tuple[str, date]] = frozenset()
    no_data: frozenset[tuple[str, date]] = frozenset()
    """No-data days to trust; recent ones are left out so the leg asks again."""


@dataclass
class LegResult:
    written: list[Written] = field(default_factory=list[Written])
    no_data: list[NoData] = field(default_factory=list[NoData])
    notes: dict[str, Any] = field(default_factory=dict[str, Any])
    implemented: bool = True

    @classmethod
    def not_implemented(cls) -> LegResult:
        return cls(implemented=False)


Leg = Callable[[Path, date, Ledger], LegResult]


def _stub(data_dir: Path, today: date, ledger: Ledger) -> LegResult:
    return LegResult.not_implemented()


LEGS: dict[str, Leg] = {
    "reference": _stub,
    "nse": _stub,
    "us": _stub,
    "kite": _stub,
}
"""Leg name -> function. Filled in one source at a time (docs/finance-app.md, Order of the move).
The leg modules import the types above, so they are registered at the bottom of this file."""

SOURCES: dict[str, tuple[str, ...]] = {
    "reference": (
        "nse_meta",
        "corp_actions",
        "holidays",
        "symbol_changes",
        "equity_master",
        "securities",
        "nse_indices",
        "index_constituents",
    ),
    "nse": ("nse_cm", "nse_full", "nse_index", "mcx", "equity", "index", "commodity"),
    "us": ("nasdaq", "yahoo", "us_listings", "us_equity", "us_symbols"),
    "kite": ("kite_1m", "equity_1m", "kite_instruments"),
}
"""The DataFile sources each leg owns; the ledger handed to a leg covers these."""

BUDGETS: dict[str, float] = {"reference": 3600, "nse": 3600, "us": 3600, "kite": 7200}
"""Seconds per leg, inside the night's budget. Kite's backfill is 9,000 requests at 3 a second."""


def _register() -> None:
    from finance.sources import kite, nse, reference, us

    LEGS["reference"] = reference.leg
    LEGS["nse"] = nse.leg
    LEGS["us"] = us.leg
    LEGS["kite"] = kite.leg


_register()
