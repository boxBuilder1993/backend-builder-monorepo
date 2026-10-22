"""The shape every daily source has: fetch one date, get bytes or a reason there are none.

A fetcher validates the date inside what it downloaded before anything is saved, because NSE serves
the previous session's file with HTTP 200 on a non-trading day and niftyindices.com serves an HTML
page. ``Transient`` is for what a retry may fix; a 404 or a stale file is ``NoFile``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class Got:
    content: bytes


@dataclass(frozen=True)
class NoFile:
    reason: str


class Fetcher(Protocol):
    name: str
    ext: str
    host: str

    def fetch(self, day: date) -> Got | NoFile: ...
