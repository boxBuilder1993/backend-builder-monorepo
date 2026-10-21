"""Storage interface and the in-memory implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from backend_builder.entity import Entity


class DuplicateKeyError(Exception):
    """Raised when a row is created with an ``id`` that already exists."""


class Store(ABC):
    """Where rows live. Each backend implements this. Rows are entity instances
    and are always returned by reference."""

    @abstractmethod
    def get[E: Entity](self, entity: type[E], where: Callable[[E], bool] | None = None) -> list[E]:
        """Live rows of ``entity``, optionally filtered."""

    @abstractmethod
    def get_by_id[E: Entity](self, entity: type[E], row_id: str) -> E | None:
        """The live row of ``entity`` with this ``id``, or ``None``."""

    @abstractmethod
    def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        """Construct and store a new row.

        ``id`` is the primary key. If ``fields`` supplies one it must not already
        exist for this entity; otherwise the entity generates one.
        """


class InMemoryStore(Store):
    """Rows held in dicts: entity -> id -> instance."""

    def __init__(self) -> None:
        self._rows: dict[type[Entity], dict[str, Entity]] = {}

    def _table[E: Entity](self, entity: type[E]) -> dict[str, E]:
        table: dict[str, E] = self._rows.setdefault(entity, {})  # type: ignore[assignment]
        return table

    def get[E: Entity](self, entity: type[E], where: Callable[[E], bool] | None = None) -> list[E]:
        rows = (row for row in self._table(entity).values() if row.deleted_at is None)
        if where is None:
            return list(rows)
        return [row for row in rows if where(row)]

    def get_by_id[E: Entity](self, entity: type[E], row_id: str) -> E | None:
        row = self._table(entity).get(row_id)
        return row if row is not None and row.deleted_at is None else None

    def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        table = self._table(entity)
        row = entity(**fields)
        if row.id in table:
            raise DuplicateKeyError(f"{entity.__name__} id {row.id!r} already exists")
        table[row.id] = row
        return row
