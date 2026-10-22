"""Where files live under the data directory, and the two writes the invariants allow.

Raw: ``raw/<source>/<YYYY>/<source>_<YYYYMMDD>.<ext>``. Parquet: ``parquet/<table>/year=<YYYY>/
<table>_<YYYYMMDD>.parquet``. Reference: ``parquet/reference/<name>.parquet``. The same layout the
old poller used, so the store continues where it left off.
"""

from __future__ import annotations

import gzip
import hashlib
from datetime import date
from pathlib import Path


def raw_rel(source: str, day: date, ext: str) -> str:
    return f"raw/{source}/{day:%Y}/{source}_{day:%Y%m%d}.{ext}"


def marker_rel(source: str, day: date) -> str:
    """The old poller's no-data marker. Read, never written: the ledger holds no-data days now."""
    return f"raw/{source}/{day:%Y}/{source}_{day:%Y%m%d}.nodata"


def parquet_rel(table: str, day: date) -> str:
    return f"parquet/{table}/year={day:%Y}/{table}_{day:%Y%m%d}.parquet"


def reference_rel(name: str) -> str:
    return f"parquet/reference/{name}.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_if_changed(path: Path, content: bytes) -> bool:
    """Atomic write, skipped when the bytes are identical, so a rebuilt file never churns."""
    if (
        path.exists()
        and hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(content).digest()
    ):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)
    return True


def write_raw(path: Path, content: bytes) -> None:
    """A raw file is written once and never rewritten. Gzip with mtime=0 keeps the bytes stable."""
    if path.exists():
        return
    if path.name.endswith(".gz"):
        content = gzip.compress(content, mtime=0)
    write_if_changed(path, content)


def read_raw(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.name.endswith(".gz") else data
