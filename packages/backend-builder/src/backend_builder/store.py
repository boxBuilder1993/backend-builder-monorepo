"""Storage interface and the in-memory implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from backend_builder.entity import Entity

Row = dict[str, Any]
Predicate = Callable[[Row], bool]


class Store(ABC):
    """Where rows live. Each backend implements this."""

    @abstractmethod
    def get(self, entity: type[Entity], where: Predicate | None = None) -> list[Row]:
        """Rows of ``entity``, optionally filtered. Rows are returned by reference."""

    @abstractmethod
    def create(self, entity: type[Entity], **fields: Any) -> Row:
        """Store a new row with a fresh ``id`` and return it by reference."""


class InMemoryStore(Store):
    """Rows held in dicts: entity -> id -> row."""

    def __init__(self) -> None:
        self._rows: dict[type[Entity], dict[str, Row]] = {}

    def get(self, entity: type[Entity], where: Predicate | None = None) -> list[Row]:
        rows = self._rows.get(entity, {}).values()
        if where is None:
            return list(rows)
        return [row for row in rows if where(row)]

    def create(self, entity: type[Entity], **fields: Any) -> Row:
        row_id = str(uuid4())
        row: Row = {"id": row_id, **fields}
        self._rows.setdefault(entity, {})[row_id] = row
        return row
