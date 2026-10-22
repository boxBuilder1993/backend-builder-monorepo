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


class StoreSync(Entity):
    """One check or pull of the store into the local mirror. The store is the source of truth; the
    mirror is never read before it has been brought to the store's current revision."""

    repo: str
    revision: str
    """The store's commit the mirror was brought to, or found to be at."""
    patterns: list[str]
    files: int = 0
    """Fetched this time; zero when the mirror was already current."""
    bytes: int = 0
    seconds: float = 0
    status: str = "current"
    """``current`` | ``pulled`` | ``failed``."""
    error: str | None = None
