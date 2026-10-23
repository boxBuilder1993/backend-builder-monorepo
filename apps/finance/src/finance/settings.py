"""The app's own settings. Runtime settings are backend-builder's; these are the data's."""

import os
from datetime import date
from pathlib import Path

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

# Reference data (NSE metadata)
CORP_URL = (
    "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
    "&from_date=01-01-{y}&to_date=31-12-{y}"
)
HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
SYMBOLCHANGE_URL = "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"
EQUITY_L_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
LINK_MAX_GAP_DAYS = 15
"""Max gap when linking an ISIN change on the same symbol (a face-value split)."""

# US equities (Nasdaq Trader listings, Yahoo prices)
NASDAQ_TRADER = "https://www.nasdaqtrader.com/dynamic/SymDir/{f}"
LISTING_FILES = ("nasdaqlisted.txt", "otherlisted.txt")
US_BENCHMARKS = ("SPY", "QQQ", "IWM")
US_DORMANT_AFTER = 3
"""Consecutive empty Yahoo answers before a symbol is parked."""
US_BATCH = 25
"""Symbols per Yahoo call, each in a short-lived child: yfinance leaks open files."""
US_PAUSE = 2.0
US_MIN_HIT_RATE = 0.2
"""A batch answering for fewer symbols than this is a failed batch, not 80% delistings."""

# Kite Connect (Zerodha) minute candles. Credentials and the daily token live outside the repo.
KITE_CONFIG = Path("~/.config/ifin/kite.toml").expanduser()
KITE_TOKEN = Path("~/.config/ifin/kite_token.json").expanduser()
KITE_START = date(2023, 10, 1)
KITE_INTERVAL = "minute"
KITE_MIN_TURNOVER = 25e7
KITE_EXTRA = ("NIFTYBEES", "BANKBEES", "JUNIORBEES")
KITE_INDICES = ("NIFTY 50", "NIFTY BANK", "NIFTY MIDCAP 100")
KITE_RANGE_DAYS = 60
KITE_PAUSE = 0.35
