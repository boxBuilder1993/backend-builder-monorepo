"""Storage interface, the in-memory store, and the session."""

from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import fields
from datetime import UTC, datetime
from typing import Any

from backend_builder.entity import Entity
from backend_builder.log import get_logger

logger = get_logger(__name__)


class DuplicateKeyError(Exception):
    """Raised when a row is created with an ``id`` that already exists."""


class MissingRowError(Exception):
    """Raised when a row is updated that does not exist."""


class NotSupported(Exception):
    """Raised when a read is asked of a store that has no native way to do it."""


class Store(ABC):
    """Where rows live between runs. Each backend implements this.

    Reads are awaited because some backends do I/O. Each backend keeps its
    native reads native: ``get`` takes a Python predicate and exists only on
    the in-memory store; ``query`` takes a native statement and exists only
    on SQL stores. ``get_by_id``, ``create`` and ``update`` exist everywhere.
    """

    async def get[E: Entity](
        self,
        entity: type[E],
        where: Callable[[E], bool] | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[E]:
        """Rows of ``entity`` matching a Python predicate. In-memory only."""
        raise NotSupported(f"{type(self).__name__} cannot evaluate a Python predicate")

    @abstractmethod
    async def get_by_id[E: Entity](
        self, entity: type[E], row_id: str, *, include_deleted: bool = False
    ) -> E | None:
        """The row of ``entity`` with this ``id``, or ``None``. A soft-deleted
        row is ``None`` unless ``include_deleted`` is set."""

    async def query[E: Entity](self, entity: type[E], statement: str, **params: Any) -> list[E]:
        """Rows of ``entity`` returned by a native statement. SQL stores only."""
        raise NotSupported(f"{type(self).__name__} cannot run a native statement")

    async def query_raw(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        """Untracked rows returned by a native statement. SQL stores only."""
        raise NotSupported(f"{type(self).__name__} cannot run a native statement")

    @abstractmethod
    async def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        """Insert a new row. ``id`` is the primary key; a duplicate raises
        ``DuplicateKeyError``."""

    @abstractmethod
    async def update[E: Entity](self, entity: type[E], row: E) -> None:
        """Replace the stored row with the same ``id`` as ``row``."""

    @asynccontextmanager
    async def run_scope(self) -> AsyncGenerator[None]:
        """Held for the whole of a run. A backend with real transactions
        yields immediately; one without serialises runs here so two runs can
        never load, mutate and flush the same row over each other."""
        yield

    def ping(self) -> bool:
        """Whether the backend is reachable. Used by the health endpoint."""
        return True


def _clone[E: Entity](row: E) -> E:
    return copy.deepcopy(row)


def _field_values(row: Entity) -> dict[str, Any]:
    return {f.name: getattr(row, f.name) for f in fields(row)}


def _visible(row: Entity, include_deleted: bool) -> bool:
    return include_deleted or row.deleted_at is None


class InMemoryStore(Store):
    """Rows held in dicts: entity -> id -> instance, returned by reference.

    Native to Python: ``get`` evaluates a predicate over instances. Has no
    transactions, so runs are serialised with a lock, one at a time per
    store. Enforces the primary key and nothing else.
    """

    def __init__(self) -> None:
        self._rows: dict[type[Entity], dict[str, Entity]] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def run_scope(self) -> AsyncGenerator[None]:
        async with self._lock:
            yield

    def _table[E: Entity](self, entity: type[E]) -> dict[str, E]:
        table: dict[str, E] = self._rows.setdefault(entity, {})  # type: ignore[assignment]
        return table

    async def get[E: Entity](
        self,
        entity: type[E],
        where: Callable[[E], bool] | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[E]:
        return [
            row
            for row in self._table(entity).values()
            if _visible(row, include_deleted) and (where is None or where(row))
        ]

    async def get_by_id[E: Entity](
        self, entity: type[E], row_id: str, *, include_deleted: bool = False
    ) -> E | None:
        row = self._table(entity).get(row_id)
        if row is None or not _visible(row, include_deleted):
            return None
        return row

    async def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        table = self._table(entity)
        row = entity(**fields)
        if row.id in table:
            raise DuplicateKeyError(f"{entity.__name__} id {row.id!r} already exists")
        table[row.id] = row
        return row

    async def update[E: Entity](self, entity: type[E], row: E) -> None:
        table = self._table(entity)
        if row.id not in table:
            raise MissingRowError(f"{entity.__name__} id {row.id!r} does not exist")
        table[row.id] = row


class Session:
    """The working set of one run: an in-memory table of instances that
    loads from a backing store on a miss and writes back on ``flush``.

    Rows loaded from the backing store are copied, so nothing a run does is
    visible outside it until ``flush``. Discarding the session discards the
    run. With no backing store the session is simply an in-memory store.
    """

    def __init__(self, backing: Store | None = None) -> None:
        self._backing = backing
        self._rows: dict[type[Entity], dict[str, Entity]] = {}
        self._created: set[tuple[type[Entity], str]] = set()
        self._fully_loaded: set[type[Entity]] = set()
        self.vars: dict[str, Any] = {}
        """Free-form values kept for the run. Shared by every reaction in the
        run and readable afterwards. Never flushed to the backing store."""

    def _table[E: Entity](self, entity: type[E]) -> dict[str, E]:
        table: dict[str, E] = self._rows.setdefault(entity, {})  # type: ignore[assignment]
        return table

    def _merge[E: Entity](self, entity: type[E], row: E) -> E:
        """Register a row loaded from the backing store, unless an instance
        with its id is already in the working set, which wins."""
        table = self._table(entity)
        existing = table.get(row.id)
        if existing is not None:
            return existing
        copied = _clone(row)
        table[row.id] = copied
        return copied

    async def get[E: Entity](
        self,
        entity: type[E],
        where: Callable[[E], bool] | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[E]:
        """Rows matching a Python predicate. Loads the whole entity from the
        backing store once per run, so it is only available when the backing
        store is in-memory."""
        if self._backing is not None and entity not in self._fully_loaded:
            for row in await self._backing.get(entity, include_deleted=True):
                self._merge(entity, row)
            self._fully_loaded.add(entity)
        return [
            row
            for row in self._table(entity).values()
            if _visible(row, include_deleted) and (where is None or where(row))
        ]

    async def get_by_id[E: Entity](
        self, entity: type[E], row_id: str, *, include_deleted: bool = False
    ) -> E | None:
        row = self._table(entity).get(row_id)
        if row is None and self._backing is not None:
            loaded = await self._backing.get_by_id(entity, row_id, include_deleted=True)
            if loaded is not None:
                row = self._merge(entity, loaded)
        if row is None or not _visible(row, include_deleted):
            return None
        return row

    async def query[E: Entity](self, entity: type[E], statement: str, **params: Any) -> list[E]:
        """Rows returned by a native statement on the backing store, merged
        into the working set. Returned in statement order; an instance already
        in the working set is returned in place of the freshly loaded row.
        No deleted-row filtering: the statement decides."""
        if self._backing is None:
            raise NotSupported("no backing store")
        rows = await self._backing.query(entity, statement, **params)
        return [self._merge(entity, row) for row in rows]

    async def query_raw(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        if self._backing is None:
            raise NotSupported("no backing store")
        return await self._backing.query_raw(statement, **params)

    def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        """Construct a new instance and register it as created. Never touches
        the backing store; a duplicate id there is rejected at flush."""
        table = self._table(entity)
        row = entity(**fields)
        if row.id in table:
            raise DuplicateKeyError(f"{entity.__name__} id {row.id!r} already exists")
        table[row.id] = row
        self._created.add((entity, row.id))
        return row

    def delete(self, row: Entity) -> None:
        """Soft-delete ``row``: stamp ``deleted_at`` with the current UTC time.
        The row stays in the working set and is written back on flush."""
        row.deleted_at = datetime.now(UTC)

    async def flush(self, *, reaction_id: str | None = None) -> None:
        """Write every row in the working set to the backing store: created
        rows as inserts, loaded rows as updates."""
        if self._backing is None:
            return
        inserted = updated = 0
        logger.info("flush started reaction_id=%s", reaction_id)
        for entity, table in self._rows.items():
            for row in table.values():
                if (entity, row.id) in self._created:
                    logger.debug("flush insert reaction_id=%s row=%r", reaction_id, row)
                    await self._backing.create(entity, **_field_values(_clone(row)))
                    inserted += 1
                else:
                    logger.debug("flush update reaction_id=%s row=%r", reaction_id, row)
                    await self._backing.update(entity, _clone(row))
                    updated += 1
        self._created.clear()
        logger.info(
            "flush finished reaction_id=%s inserted=%d updated=%d", reaction_id, inserted, updated
        )
