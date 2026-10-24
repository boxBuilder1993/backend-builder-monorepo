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
INDEX_LIST_URL = "https://nsearchives.nseindia.com/content/indices/{name}.csv"
INDEX_FILES: dict[str, str] = {
    "NIFTY 50": "ind_nifty50list",
    "NIFTY 500": "ind_nifty500list",
    "NIFTY TOTAL MARKET": "ind_niftytotalmarket_list",
    "NIFTY MIDCAP 100": "ind_niftymidcap100list",
    "NIFTY SMALLCAP 250": "ind_niftysmallcap250list",
    "NIFTY BANK": "ind_niftybanklist",
    "NIFTY IT": "ind_niftyitlist",
    "NIFTY AUTO": "ind_niftyautolist",
    "NIFTY PHARMA": "ind_niftypharmalist",
    "NIFTY FMCG": "ind_niftyfmcglist",
    "NIFTY METAL": "ind_niftymetallist",
    "NIFTY REALTY": "ind_niftyrealtylist",
    "NIFTY ENERGY": "ind_niftyenergylist",
}
"""NSE index constituent files (company, industry, symbol, series, ISIN): the sector labels and the
Kite universe's index half. A snapshot is stored only when the file's bytes change."""
LINK_MAX_GAP_DAYS = 15
"""Max gap when linking an ISIN change on the same symbol (a face-value split)."""

# US equities (Nasdaq Trader listings, Yahoo prices)
NASDAQ_TRADER = "https://www.nasdaqtrader.com/dynamic/SymDir/{f}"
LISTING_FILES = ("nasdaqlisted.txt", "otherlisted.txt")
US_BENCHMARKS = (
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
    "XLF",
    "XLV",
    "XLY",
    "XLP",
    "XLE",
    "XLI",
    "XLB",
    "XLU",
    "XLRE",
    "XLC",  # SPDR sectors
    "GLD",
    "SLV",
    "USO",  # gold, silver, crude, as the US side's commodity read
)
"""Listed ETFs fetched beside the stocks: the market, size and sector context of the US rows."""
US_INDICES = {"^VIX": "CBOE Volatility Index"}
"""Yahoo index symbols that are not in the listings, fetched like the benchmarks."""
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
"""Every US-listed stock with sector, industry, country, IPO year and market cap, in one answer: the
sector labels and size buckets of the US rows; only the stable columns are kept, on change."""
US_CAP_BUCKETS = ((100e9, "mega"), (10e9, "large"), (2e9, "mid"), (300e6, "small"), (0, "micro"))
"""Market-cap floors, in dollars, for the size bucket of a US stock."""
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
KITE_UNIVERSE_INDEX = "NIFTY 500"
"""Every member of this index is fetched, liquid or not, besides the turnover rule."""
KITE_STAY_SESSIONS = 250
"""A stock fetched on any of the last this many sessions stays in the universe: entry is a gate, not
a nightly vote, so minute histories have no holes from hovering around the turnover bar."""
KITE_EXTRA = ("NIFTYBEES", "BANKBEES", "JUNIORBEES")
KITE_INDICES = (
    "NIFTY 50",
    "NIFTY BANK",
    "NIFTY MIDCAP 100",
    "NIFTY 500",
    "NIFTY NEXT 50",
    "NIFTY MIDCAP 150",
    "NIFTY AUTO",
    "NIFTY IT",
    "NIFTY PHARMA",
    "NIFTY FMCG",
    "NIFTY METAL",
    "NIFTY REALTY",
    "NIFTY ENERGY",
    "NIFTY FIN SERVICE",
    "NIFTY PSU BANK",
    "NIFTY PVT BANK",
    "NIFTY INFRA",
    "NIFTY OIL AND GAS",
    "NIFTY HEALTHCARE",
    "NIFTY CONSUMPTION",
    "NIFTY MEDIA",
    "NIFTY COMMODITIES",
    "INDIA VIX",
)
"""Minute bars of the market, size and sector indices, as context features: a stock's move against
its sector is a different signal from its move against the market. Kite's INDICES segment names."""
KITE_RANGE_DAYS = 60
KITE_PAUSE = 0.35
