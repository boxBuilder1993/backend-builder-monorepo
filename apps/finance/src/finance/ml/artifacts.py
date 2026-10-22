"""The model directory: the estimator's own files plus schema.json and fit.json, and a stable hash
of the whole directory. Self-describing: load needs nothing but the directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from finance.ml import estimators as E
from finance.ml.schema import Schema


def save(directory: Path, fitted: E.Fitted, schema: Schema, info: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fitted.save(directory)
    (directory / "schema.json").write_text(json.dumps(schema.to_dict(), indent=1, sort_keys=True))
    (directory / "fit.json").write_text(json.dumps(info, indent=1, sort_keys=True, default=str))


def load(directory: Path) -> tuple[E.Fitted, Schema, dict[str, Any]]:
    schema = Schema.from_dict(json.loads((directory / "schema.json").read_text()))
    info = json.loads((directory / "fit.json").read_text())
    return E.load(info["algorithm"], directory, schema), schema, info


def archive_hash(directory: Path) -> tuple[str, int]:
    """sha256 over every file's relative path and bytes in sorted order, and the total size."""
    h = hashlib.sha256()
    total = 0
    for p in sorted(x for x in directory.rglob("*") if x.is_file()):
        rel = p.relative_to(directory).as_posix().encode()
        data = p.read_bytes()
        h.update(len(rel).to_bytes(4, "big") + rel + len(data).to_bytes(8, "big") + data)
        total += len(data)
    return h.hexdigest(), total
