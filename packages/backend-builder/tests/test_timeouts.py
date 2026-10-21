import asyncio
import logging
from collections.abc import AsyncIterator

import pytest
from backend_builder import (
    App,
    Context,
    Entity,
    InMemoryStore,
    OnCreate,
    OutboxEntry,
    PostgresStore,
    Reaction,
    Settings,
    Worker,
    run,
)
from backend_builder.reaction import qualified_name
from sqlalchemy.ext.asyncio import create_async_engine


class Thing(Entity):
    label: str = ""


class Sleep(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Thing, label="before sleep")
        await asyncio.sleep(ctx.input.get("seconds", 1.0))
        ctx.create(Thing, label="after sleep")


class Quick(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Thing, label="quick")


class TightTrigger(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(Sleep, {"seconds": 1.0}, timeout=0.1)
        ctx.create(Thing, label="never")


class LooseTrigger(Reaction):
    """An inner timeout longer than the run's: the run's wins."""

    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(Sleep, {"seconds": 1.0}, timeout=30)


class DeferWithBudget(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(Sleep, {"seconds": 0}, deferred=True, timeout=120)
        await ctx.trigger(Quick, {}, deferred=True)


class FlakyThenSlow(Reaction):
    calls: int = 0

    async def calculate(self, ctx: Context) -> None:
        FlakyThenSlow.calls += 1
        if FlakyThenSlow.calls == 1:
            raise RuntimeError("first attempt fails")
        await asyncio.sleep(1.0)


async def _create_tables(store: PostgresStore) -> None:
    engine = create_async_engine(store.url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(store.metadata.create_all)
    finally:
        await engine.dispose()


@pytest.fixture
async def store(database_url: str) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(database_url, [Thing])
    await _create_tables(store)
    yield store
    await store.close()


# --- the run's deadline ----------------------------------------------------------------


async def test_run_times_out_writes_nothing_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="backend_builder")
    store = InMemoryStore()
    with pytest.raises(TimeoutError):
        await run(store, Sleep, {"seconds": 1.0}, timeout=0.1)
    assert await store.get(Thing) == []
    assert any(
        m.startswith("run timed out reaction=Sleep")
        for m in (r.getMessage() for r in caplog.records)
    )


async def test_run_within_budget_succeeds() -> None:
    store = InMemoryStore()
    await run(store, Sleep, {"seconds": 0.01}, timeout=5)
    assert len(await store.get(Thing)) == 2


async def test_deadline_covers_all_attempts() -> None:
    FlakyThenSlow.calls = 0
    with pytest.raises(TimeoutError):
        await run(InMemoryStore(), FlakyThenSlow, {}, max_attempts=3, timeout=0.5)
    assert FlakyThenSlow.calls == 2  # retried once, then the deadline ended attempt two


async def test_timeout_is_not_retried() -> None:
    store = InMemoryStore()
    with pytest.raises(TimeoutError):
        await run(store, Sleep, {"seconds": 1.0}, max_attempts=3, timeout=0.2)


# --- trigger timeouts -------------------------------------------------------------------


async def test_immediate_trigger_timeout_aborts_the_run() -> None:
    store = InMemoryStore()
    with pytest.raises(TimeoutError):
        await run(store, TightTrigger, {}, timeout=10)
    assert await store.get(Thing) == []


async def test_immediate_trigger_cannot_loosen_the_run() -> None:
    with pytest.raises(TimeoutError):
        await run(InMemoryStore(), LooseTrigger, {}, timeout=0.2)


async def test_postgres_run_timeout_rolls_back(store: PostgresStore) -> None:
    with pytest.raises(TimeoutError):
        await run(store, Sleep, {"seconds": 1.0}, timeout=0.2)
    assert await store.query_raw("select count(*) as n from thing") == [{"n": 0}]
    await run(store, Quick, {})  # the store is still usable afterwards
    assert await store.query_raw("select count(*) as n from thing") == [{"n": 1}]


# --- the outbox carries the budget ----------------------------------------------------------


async def test_deferred_rows_carry_their_budget(store: PostgresStore) -> None:
    await run(store, DeferWithBudget, {}, timeout=45)
    rows = await store.query_raw(
        "select reaction, timeout_seconds from _outbox order by timeout_seconds"
    )
    assert rows == [
        {"reaction": qualified_name(Quick), "timeout_seconds": 45.0},
        {"reaction": qualified_name(Sleep), "timeout_seconds": 120.0},
    ]


async def test_trigger_declarations_carry_their_budget(store: PostgresStore) -> None:
    await run(store, Quick, {}, timeout=9, triggers=[OnCreate(Thing, Sleep, timeout=300)])
    rows = await store.query_raw("select reaction, timeout_seconds from _outbox")
    assert rows == [{"reaction": qualified_name(Sleep), "timeout_seconds": 300.0}]


async def test_worker_delivers_under_the_rows_budget(store: PostgresStore) -> None:
    app = App(package="fixture_app", include=[Thing, Sleep], max_attempts=1)
    worker = Worker(app, store, Settings(outbox_batch=5, outbox_max_attempts=3))
    await store.enqueue(
        [OutboxEntry(reaction=qualified_name(Sleep), input={"seconds": 1.0}, timeout_seconds=0.2)]
    )
    assert await worker.run_once() == 1
    rows = await store.query_raw("select attempts, last_error, dead_at from _outbox")
    assert rows[0]["attempts"] == 1 and rows[0]["last_error"].startswith("TimeoutError")
    assert rows[0]["dead_at"] is None
    assert await store.query_raw("select count(*) as n from thing") == [{"n": 0}]


async def test_lease_is_the_rows_budget_plus_grace(store: PostgresStore) -> None:
    await store.enqueue(
        [
            OutboxEntry(reaction=qualified_name(Quick), input={}, timeout_seconds=0.2),
            OutboxEntry(reaction=qualified_name(Quick), input={}, timeout_seconds=30),
        ]
    )
    first = await store.claim(5, "a", grace_seconds=0.2)
    assert len(first) == 2
    await asyncio.sleep(0.5)
    reclaimable = await store.claim(5, "b", grace_seconds=0.2)
    assert [r.timeout_seconds for r in reclaimable] == [0.2]  # the long one is still leased
