"""The Postgres store: SQLAlchemy Core over asyncpg.

A run is one ``SERIALIZABLE`` transaction opened by ``run_scope``; every
read and write inside the run uses its connection. A serialization failure,
or a primary key violation on a client-generated id, raises ``TransientError``
and the engine retries the run from the seed. Outside a run, each operation
uses its own short transaction.
"""

from __future__ import annotations

import asyncio
import random
import typing
import uuid
from collections.abc import AsyncGenerator, Iterable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from backend_builder.codec import EncodeError, from_jsonable, to_jsonable
from backend_builder.entity import Entity
from backend_builder.log import get_logger
from backend_builder.migrations import async_url
from backend_builder.outbox import (
    OUTBOX_TABLE,
    SCHEDULE_TABLE,
    OutboxEntry,
    OutboxRow,
    OutboxStats,
    now,
)
from backend_builder.schema import build_metadata, persisted_fields, table_name
from backend_builder.store import DuplicateKeyError, MissingRowError, Store, TransientError

if typing.TYPE_CHECKING:
    from backend_builder.triggers import OnSchedule

logger = get_logger(__name__)

SERIALIZATION_FAILURE = "40001"
DEADLOCK_DETECTED = "40P01"
UNIQUE_VIOLATION = "23505"


class PostgresDuplicateKeyError(DuplicateKeyError, TransientError):
    """A client-generated id collided. Retrying the run regenerates ids."""


def _plain(value: Any) -> Any:
    """Driver types to the entity's types: raw ``text()`` statements bypass
    column type processing, so uuid columns arrive as ``uuid.UUID``."""
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def _sqlstate(exc: BaseException) -> str | None:
    orig = getattr(exc, "orig", exc)
    code = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    return str(code) if code else None


class PostgresStore(Store):
    def __init__(self, url: str, entities: Iterable[type[Entity]]) -> None:
        self.url = async_url(url)
        self.metadata = build_metadata(entities)
        self._tables = {table_name(e): e for e in entities}
        self._hints: dict[type[Entity], dict[str, Any]] = {}
        self._engines: dict[asyncio.AbstractEventLoop, AsyncEngine] = {}
        self._current: ContextVar[AsyncConnection | None] = ContextVar("connection", default=None)

    # --- engine and connections -----------------------------------------------

    @property
    def engine(self) -> AsyncEngine:
        """One engine, and so one connection pool, per running event loop.
        asyncpg connections are bound to the loop that opened them, and a
        store may be used from more than one loop, for example by a test
        client that runs the app on a thread of its own."""
        loop = asyncio.get_running_loop()
        engine = self._engines.get(loop)
        if engine is None:
            engine = self._engines[loop] = create_async_engine(self.url)
        return engine

    async def close(self) -> None:
        """Dispose the current loop's pool. Pools on other loops are disposed
        when those loops call ``close`` themselves."""
        loop = asyncio.get_running_loop()
        engine = self._engines.pop(loop, None)
        if engine is not None:
            await engine.dispose()

    @asynccontextmanager
    async def run_scope(self) -> AsyncGenerator[None]:
        """One serializable transaction for the whole run."""
        async with self.engine.connect() as connection:
            await connection.execution_options(isolation_level="SERIALIZABLE")
            token = self._current.set(connection)
            try:
                async with connection.begin():
                    yield
            except DBAPIError as exc:
                raise self._translate(exc) from exc
            finally:
                self._current.reset(token)

    @asynccontextmanager
    async def _connection(self) -> AsyncGenerator[AsyncConnection]:
        current = self._current.get()
        if current is not None:
            yield current
            return
        async with self.engine.begin() as connection:
            yield connection

    def _translate(self, exc: DBAPIError) -> Exception:
        code = _sqlstate(exc)
        if code in (SERIALIZATION_FAILURE, DEADLOCK_DETECTED):
            return TransientError(f"postgres {code}: {exc.orig}")
        if code == UNIQUE_VIOLATION:
            return PostgresDuplicateKeyError(str(exc.orig))
        return exc

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                await connection.execute(sa.text("select 1"))
            return True
        except Exception:
            return False

    # --- rows to instances and back ----------------------------------------------

    def _table(self, entity: type[Entity]) -> sa.Table:
        return self.metadata.tables[table_name(entity)]

    def _annotations(self, entity: type[Entity]) -> dict[str, Any]:
        if entity not in self._hints:
            self._hints[entity] = typing.get_type_hints(entity)
        return self._hints[entity]

    def _is_json(self, entity: type[Entity], name: str) -> bool:
        return isinstance(self._table(entity).c[name].type, pg.JSONB)

    def _encode(self, entity: type[Entity], values: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in values.items():
            if self._is_json(entity, name):
                try:
                    out[name] = to_jsonable(value, path=f"{entity.__name__}.{name}")
                except EncodeError as exc:
                    raise EncodeError(str(exc)) from None
            else:
                out[name] = value
        return out

    def _decode[E: Entity](self, entity: type[E], row: Mapping[Any, Any]) -> E:
        hints = self._annotations(entity)
        values: dict[str, Any] = {}
        for field in persisted_fields(entity):
            if field.name not in row:
                if field.default is not field.default_factory:  # has some default
                    continue
                raise LookupError(f"row has no column {field.name!r} for {entity.__name__}")
            value = _plain(row[field.name])
            if self._is_json(entity, field.name):
                value = from_jsonable(hints[field.name], value)
            values[field.name] = value
        return entity(**values)

    # --- Store ------------------------------------------------------------------------

    async def get_by_id[E: Entity](
        self, entity: type[E], row_id: str, *, include_deleted: bool = False
    ) -> E | None:
        table = self._table(entity)
        statement = sa.select(table).where(table.c.id == row_id)
        if not include_deleted:
            statement = statement.where(table.c.deleted_at.is_(None))
        async with self._connection() as connection:
            row = (await connection.execute(statement)).mappings().first()
        return self._decode(entity, row) if row is not None else None

    async def query[E: Entity](self, entity: type[E], statement: str, **params: Any) -> list[E]:
        async with self._connection() as connection:
            rows = (await connection.execute(sa.text(statement), params)).mappings().all()
        return [self._decode(entity, row) for row in rows]

    async def query_raw(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        async with self._connection() as connection:
            rows = (await connection.execute(sa.text(statement), params)).mappings().all()
        return [{str(k): _plain(v) for k, v in row.items()} for row in rows]

    async def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        row = entity(**fields)
        values = self._encode(
            entity, {f.name: getattr(row, f.name) for f in persisted_fields(entity)}
        )
        try:
            async with self._connection() as connection:
                await connection.execute(sa.insert(self._table(entity)).values(**values))
        except IntegrityError as exc:
            raise self._translate(exc) from exc
        return row

    # --- outbox ---------------------------------------------------------------------

    @property
    def _outbox(self) -> sa.Table:
        return self.metadata.tables[OUTBOX_TABLE]

    async def enqueue(self, entries: list[OutboxEntry]) -> None:
        if not entries:
            return
        stamp = now()
        rows = [
            {
                "id": e.id,
                "reaction": e.reaction,
                "input": e.input,
                "created_at": stamp,
                "available_at": e.available_at or stamp,
                "scheduled_for": e.scheduled_for,
                "timeout_seconds": e.timeout_seconds if e.timeout_seconds is not None else 60.0,
            }
            for e in entries
        ]
        statement = pg.insert(self._outbox).on_conflict_do_nothing(
            index_elements=["reaction", "scheduled_for"],
            index_where=sa.text("scheduled_for IS NOT NULL"),
        )
        async with self._connection() as connection:
            await connection.execute(statement, rows)

    async def claim(self, limit: int, worker_id: str, grace_seconds: float) -> list[OutboxRow]:
        statement = sa.text(
            f"""
            update {OUTBOX_TABLE} set claimed_by = :worker, claimed_at = :stamp
            where id in (
                select id from {OUTBOX_TABLE}
                where dead_at is null
                  and available_at <= :stamp
                  and (
                    claimed_at is null
                    or claimed_at + make_interval(secs => timeout_seconds + :grace) < :stamp
                  )
                order by available_at
                limit :limit
                for update skip locked
            )
            returning id, reaction, input, attempts, available_at, created_at,
                      scheduled_for, timeout_seconds
            """
        )
        stamp = now()
        params = {
            "worker": worker_id,
            "stamp": stamp,
            "grace": grace_seconds,
            "limit": limit,
        }
        async with self.engine.begin() as connection:
            rows = (await connection.execute(statement, params)).mappings().all()
        return [
            OutboxRow(
                id=str(r["id"]),
                reaction=r["reaction"],
                input=r["input"],
                attempts=r["attempts"],
                available_at=r["available_at"],
                created_at=r["created_at"],
                scheduled_for=r["scheduled_for"],
                timeout_seconds=float(r["timeout_seconds"]),
            )
            for r in rows
        ]

    async def ack(self, row_id: str) -> None:
        async with self._connection() as connection:
            await connection.execute(sa.delete(self._outbox).where(self._outbox.c.id == row_id))

    async def release(self, row_id: str, error: str, max_attempts: int) -> None:
        table = self._outbox
        async with self.engine.begin() as connection:
            current = (
                await connection.execute(sa.select(table.c.attempts).where(table.c.id == row_id))
            ).scalar_one_or_none()
            if current is None:
                return
            attempts = int(current) + 1
            stamp = now()
            values: dict[str, Any] = {
                "attempts": attempts,
                "last_error": error[:2000],
                "claimed_by": None,
                "claimed_at": None,
            }
            if attempts >= max_attempts:
                values["dead_at"] = stamp
            else:
                backoff = min(2**attempts, 300) * random.uniform(0.5, 1.5)
                values["available_at"] = stamp + timedelta(seconds=backoff)
            await connection.execute(sa.update(table).where(table.c.id == row_id).values(**values))

    async def outbox_stats(self) -> OutboxStats:
        table = self._outbox
        pending = sa.select(sa.func.count(), sa.func.min(table.c.available_at)).where(
            table.c.dead_at.is_(None), table.c.claimed_at.is_(None)
        )
        claimed = sa.select(sa.func.count()).where(
            table.c.dead_at.is_(None), table.c.claimed_at.is_not(None)
        )
        dead = sa.select(sa.func.count()).where(table.c.dead_at.is_not(None))
        async with self.engine.connect() as connection:
            count, oldest = (await connection.execute(pending)).one()
            claimed_count = (await connection.execute(claimed)).scalar_one()
            dead_count = (await connection.execute(dead)).scalar_one()
        return OutboxStats(
            pending=int(count),
            claimed=int(claimed_count),
            dead=int(dead_count),
            oldest_pending=oldest,
        )

    async def seed_schedule(self, schedule: OnSchedule, moment: datetime) -> bool:
        from backend_builder.reaction import qualified_name

        name = qualified_name(schedule.reaction)
        table = self.metadata.tables[SCHEDULE_TABLE]
        async with self.engine.begin() as connection:
            upsert = pg.insert(table).values(
                reaction=name, cron=schedule.cron, last_scheduled_for=None, updated_at=moment
            )
            await connection.execute(upsert.on_conflict_do_nothing(index_elements=["reaction"]))
            row = (
                await connection.execute(
                    sa.select(table.c.cron, table.c.last_scheduled_for)
                    .where(table.c.reaction == name)
                    .with_for_update()
                )
            ).one()
            last: datetime | None = row.last_scheduled_for
            cron_changed = row.cron != schedule.cron
            if last is not None and last >= moment and not cron_changed:
                return False
            occurrence = schedule.next_after(moment)
            entry = schedule.entry(occurrence)
            await connection.execute(
                pg.insert(self._outbox)
                .values(
                    id=entry.id,
                    reaction=entry.reaction,
                    input=entry.input,
                    created_at=moment,
                    available_at=occurrence,
                    scheduled_for=occurrence,
                    timeout_seconds=entry.timeout_seconds
                    if entry.timeout_seconds is not None
                    else 60.0,
                )
                .on_conflict_do_nothing(
                    index_elements=["reaction", "scheduled_for"],
                    index_where=sa.text("scheduled_for IS NOT NULL"),
                )
            )
            await connection.execute(
                sa.update(table)
                .where(table.c.reaction == name)
                .values(cron=schedule.cron, last_scheduled_for=occurrence, updated_at=moment)
            )
        return True

    async def update(self, entity: type[Entity], row_id: str, changes: dict[str, Any]) -> None:
        if not changes:
            return
        table = self._table(entity)
        statement = (
            sa.update(table).where(table.c.id == row_id).values(**self._encode(entity, changes))
        )
        async with self._connection() as connection:
            result = await connection.execute(statement)
        if result.rowcount == 0:
            raise MissingRowError(f"{entity.__name__} id {row_id!r} does not exist")
