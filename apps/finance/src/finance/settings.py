"""The app's own settings, from the environment. Runtime settings are backend-builder's."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("FINANCE_DATA_DIR", "~/finance-data")).expanduser()
HF_REPO = os.environ.get("FINANCE_HF_REPO", "shr22avan/finance-data")
RECENT_DAYS = 7  # a no_data day inside this window is asked again on every run
BACKFILL_START_YEAR = 2021
