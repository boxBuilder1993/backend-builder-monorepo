"""MCX date-wise bhavcopy, futures only, saved as the JSON row list."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from finance.build.mcx import mcx_date
from finance.http import Mcx
from finance.settings import MCX_INSTRUMENT
from finance.sources.daily import Got, NoFile


class McxBhav:
    name, ext, host = "mcx", "json.gz", "mcx"

    def __init__(self, client: Mcx | None = None) -> None:
        self.c = client or Mcx()

    def fetch(self, day: date) -> Got | NoFile:
        j: dict[str, Any] = self.c.get_json(
            {"InstrumentName": MCX_INSTRUMENT, "fromDate": f"{day:%d/%m/%Y}"}
        )
        rows: list[dict[str, Any]] = j.get("Data") or []
        if not j.get("IsSuccess") or not rows:
            return NoFile(str(j.get("Message") or "empty"))
        found = mcx_date(rows)
        if found != day:
            return NoFile(f"stale:{found}")
        return Got(json.dumps(rows, separators=(",", ":")).encode())
