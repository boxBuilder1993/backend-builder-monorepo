"""The app's own settings. Runtime settings are backend-builder's; these are the data's."""

import os
from datetime import date

HF_REPO = os.environ.get("FINANCE_HF_REPO", "shr22avan/finance-data")
PUBLISH = os.environ.get("FINANCE_PUBLISH", "on").lower() not in ("off", "0", "false", "no")
"""FINANCE_PUBLISH=off runs the legs but never uploads: the shadow phase next to the old poller."""
RECENT_DAYS = 7
"""A no_data day inside this window is asked again on every run: a file can be published late."""

BACKFILL_START = date(2021, 9, 1)
UDIFF_START = date(2024, 1, 2)
"""The cm bhavcopy's UDiFF layout is served from here; verified by probing."""
OLD_CM_END = date(2024, 7, 5)
"""The old cm bhavcopy layout returns 404 from 8 July 2024."""

REQUEST_PAUSE = {"nsearchives": 0.25, "niftyindices": 0.25, "mcx": 0.5}
"""Seconds between requests to a host. Polite, and enough to stay unblocked."""

MCX_INSTRUMENT = "FUTCOM"
