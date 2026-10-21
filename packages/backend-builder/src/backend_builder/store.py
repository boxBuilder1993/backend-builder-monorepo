"""Storage interface and the in-memory implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from backend_builder.entity import Entity

Row = dict[str, Any]
Predicate = Callable[[Row], bool]


class DuplicateKeyError(Exception):
    """Raised when a row is created with an ``id`` that already exists."""


class Store(ABC):
    """Where rows live. Each backend implements this."""

    @abstractmethod
    def get(self, entity: type[Entity], where: Predicate | None = None) -> list[Row]:
        """Rows of ``entity``, optionally filtered. Rows are returned by reference."""

    @abstractmethod
    def get_by_id(self, entity: type[Entity], row_id: str) -> Row | None:
        """The row of ``entity`` with this ``id``, by reference, or ``None``."""

    @abstractmethod
    def create(self, entity: type[Entity], **fields: Any) -> Row:
        """Store a new row and return it by reference.

        ``id`` is the primary key. If ``fields`` supplies one it must not already
        exist for this entity; otherwise a fresh UUID string is generated.
        """


class InMemoryStore(Store):
    """Rows held in dicts: entity -> id -> row."""

    def __init__(self) -> None:
        self._rows: dict[type[Entity], dict[str, Row]] = {}

    def get(self, entity: type[Entity], where: Predicate | None = None) -> list[Row]:
        rows = self._rows.get(entity, {}).values()
        if where is None:
            return list(rows)
        return [row for row in rows if where(row)]

    def get_by_id(self, entity: type[Entity], row_id: str) -> Row | None:
        return self._rows.get(entity, {}).get(row_id)

    def create(self, entity: type[Entity], **fields: Any) -> Row:
        rows = self._rows.setdefault(entity, {})
        if "id" in fields:
            row_id = fields.pop("id")
            if row_id in rows:
                raise DuplicateKeyError(f"{entity.__name__} id {row_id!r} already exists")
        else:
            row_id = str(uuid4())
            while row_id in rows:
                row_id = str(uuid4())
        row: Row = {"id": row_id, **fields}
        rows[row_id] = row
        return row
