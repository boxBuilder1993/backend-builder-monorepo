from datetime import date, datetime

from backend_builder import Entity


class DataFile(Entity):
    """One file in the store, or one (source, day) known to have no file.

    The ledger. Files stay the data; this row says what was done to them. ``path`` is unique
    (migration); a no_data row has the virtual path ``<source>/.nodata/<day>`` and no file.
    """

    path: str
    kind: str
    """``raw`` | ``parquet`` | ``reference``."""
    source: str
    """raw: nse_cm, nse_full, nse_index, mcx, yahoo, kite_1m; parquet: table; reference: name."""
    day: date | None = None
    status: str = "ok"
    """``ok`` | ``no_data``."""
    bytes: int = 0
    sha256: str | None = None
    written_at: datetime | None = None
    checked_at: datetime | None = None
    published_at: datetime | None = None
