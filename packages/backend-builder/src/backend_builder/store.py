"""Storage interface, the in-memory store, and the session."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import fields
from typing import Any

from backend_builder.entity import Entity


class DuplicateKeyError(Exception):
    """Raised when a row is created with an ``id`` that already exists."""


class MissingRowError(Exception):
    """Raised when a row is updated that does not exist."""


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

    @abstractmethod
    def update[E: Entity](self, entity: type[E], row: E) -> None:
        """Replace the stored row with the same ``id`` as ``row``."""


def _clone[E: Entity](row: E) -> E:
    return copy.deepcopy(row)


def _field_values(row: Entity) -> dict[str, Any]:
    return {f.name: getattr(row, f.name) for f in fields(row)}


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

    def update[E: Entity](self, entity: type[E], row: E) -> None:
        table = self._table(entity)
        if row.id not in table:
            raise MissingRowError(f"{entity.__name__} id {row.id!r} does not exist")
        table[row.id] = row


class Session(InMemoryStore):
    """The working set of one run: an in-memory store that loads from a
    backing store on a miss and writes back on ``flush``.

    Rows loaded from the backing store are copied, so nothing a run does is
    visible outside it until ``flush``. Discarding the session discards the
    run. With no backing store the session is simply an in-memory store.
    """

    def __init__(self, backing: Store | None = None) -> None:
        super().__init__()
        self._backing = backing
        self._created: set[tuple[type[Entity], str]] = set()
        self._fully_loaded: set[type[Entity]] = set()

    def get[E: Entity](
        self,
        entity: type[E],
        where: Callable[[E], bool] | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[E]:
        if self._backing is not None and entity not in self._fully_loaded:
            table = self._table(entity)
            for row in self._backing.get(entity, include_deleted=True):
                table.setdefault(row.id, _clone(row))
            self._fully_loaded.add(entity)
        return super().get(entity, where, include_deleted=include_deleted)

    def get_by_id[E: Entity](
        self, entity: type[E], row_id: str, *, include_deleted: bool = False
    ) -> E | None:
        table = self._table(entity)
        if row_id not in table and self._backing is not None:
            row = self._backing.get_by_id(entity, row_id, include_deleted=True)
            if row is not None:
                table[row_id] = _clone(row)
        return super().get_by_id(entity, row_id, include_deleted=include_deleted)

    def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        row = super().create(entity, **fields)
        if self._backing is not None and self._backing.get_by_id(
            entity, row.id, include_deleted=True
        ):
            del self._table(entity)[row.id]
            raise DuplicateKeyError(f"{entity.__name__} id {row.id!r} already exists")
        self._created.add((entity, row.id))
        return row

    def flush(self) -> None:
        """Write every row in the working set to the backing store: created
        rows as inserts, loaded rows as updates."""
        if self._backing is None:
            return
        for entity, table in self._rows.items():
            for row in table.values():
                if (entity, row.id) in self._created:
                    self._backing.create(entity, **_field_values(_clone(row)))
                else:
                    self._backing.update(entity, _clone(row))
        self._created.clear()
