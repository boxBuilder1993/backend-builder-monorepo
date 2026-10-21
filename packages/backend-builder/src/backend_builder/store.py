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
    def get[E: Entity](
        self,
        entity: type[E],
        where: Callable[[E], bool] | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[E]:
        """Rows of ``entity``, optionally filtered. Soft-deleted rows are
        excluded unless ``include_deleted`` is set."""

    @abstractmethod
    def get_by_id[E: Entity](
        self, entity: type[E], row_id: str, *, include_deleted: bool = False
    ) -> E | None:
        """The row of ``entity`` with this ``id``, or ``None``. A soft-deleted
        row is ``None`` unless ``include_deleted`` is set."""

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

    def get[E: Entity](
        self,
        entity: type[E],
        where: Callable[[E], bool] | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[E]:
        return [
            row
            for row in self._table(entity).values()
            if (include_deleted or row.deleted_at is None) and (where is None or where(row))
        ]

    def get_by_id[E: Entity](
        self, entity: type[E], row_id: str, *, include_deleted: bool = False
    ) -> E | None:
        row = self._table(entity).get(row_id)
        if row is None or (row.deleted_at is not None and not include_deleted):
            return None
        return row

    def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        table = self._table(entity)
        row = entity(**fields)
        if row.id in table:
            raise DuplicateKeyError(f"{entity.__name__} id {row.id!r} already exists")
        table[row.id] = row
        return row
