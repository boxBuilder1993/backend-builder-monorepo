import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from backend_builder import (
    App,
    Context,
    Entity,
    InMemoryStore,
    NotSupported,
    OutboxEntry,
    PostgresStore,
    Reaction,
    Settings,
    Worker,
    run,
)
from backend_builder.cli import cli
from backend_builder.codec import EncodeError
from backend_builder.reaction import qualified_name
from sqlalchemy.ext.asyncio import create_async_engine
from typer.testing import CliRunner


class Account(Entity):
    owner: str
    balance: int = 0
    notified: int = 0


class Notify(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["account_id"])
        assert account is not None
        account.notified += 1


class Open(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = ctx.create(Account, owner=ctx.input["owner"])
        await ctx.trigger(Notify, {"account_id": account.id}, deferred=True)
        ctx.save("account", account)


class OpenThenFail(Open):
    async def calculate(self, ctx: Context) -> None:
        await super().calculate(ctx)
        raise RuntimeError("abort")


class Unstorable(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(Notify, {"when": object()}, deferred=True)


class Flaky(Reaction):
    attempts_seen: int = 0

    async def calculate(self, ctx: Context) -> None:
        Flaky.attempts_seen += 1
        raise ValueError("still failing")


class Chain(Reaction):
    """A deferred reaction that defers another."""

    async def calculate(self, ctx: Context) -> None:
        account = ctx.create(Account, owner="chained")
        await ctx.trigger(Notify, {"account_id": account.id}, deferred=True)


async def _create_tables(store: PostgresStore) -> None:
    engine = create_async_engine(store.url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(store.metadata.create_all)
    finally:
        await engine.dispose()


@pytest.fixture
async def store(database_url: str) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(database_url, [Account])
    await _create_tables(store)
    yield store
    await store.close()


@pytest.fixture
def app() -> App:
    return App(package="fixture_app", include=[Account, Notify, Open, Flaky, Chain], max_attempts=1)


@pytest.fixture
def settings() -> Settings:
    return Settings(outbox_batch=5, outbox_lease_seconds=1.0, outbox_max_attempts=2)


@pytest.fixture
def worker(app: App, store: PostgresStore, settings: Settings) -> Worker:
    return Worker(app, store, settings)


async def test_deferred_trigger_is_written_with_the_run_and_delivered(
    store: PostgresStore, worker: Worker
) -> None:
    ctx = await run(store, Open, {"owner": "ada"})
    account_id = ctx.load("account").id

    stats = await store.outbox_stats()
    assert (stats.pending, stats.claimed, stats.dead) == (1, 0, 0)
    before = await store.get_by_id(Account, account_id)
    assert before is not None and before.notified == 0

    assert await worker.run_once() == 1
    after = await store.get_by_id(Account, account_id)
    assert after is not None and after.notified == 1
    assert (await store.outbox_stats()).pending == 0
    assert await worker.run_once() == 0


async def test_failed_run_enqueues_nothing(store: PostgresStore) -> None:
    with pytest.raises(RuntimeError):
        await run(store, OpenThenFail, {"owner": "x"})
    assert (await store.outbox_stats()).pending == 0


async def test_unstorable_input_fails_at_the_trigger(store: PostgresStore) -> None:
    with pytest.raises(EncodeError, match="deferred .*Notify input.when"):
        await run(store, Unstorable, {})


async def test_failed_delivery_is_released_with_backoff_then_dead(
    store: PostgresStore, worker: Worker
) -> None:
    Flaky.attempts_seen = 0
    await store.enqueue([OutboxEntry(reaction=qualified_name(Flaky), input={})])

    assert await worker.run_once() == 1
    assert Flaky.attempts_seen == 1
    stats = await store.outbox_stats()
    assert (stats.pending, stats.dead) == (1, 0)
    rows = await store.query_raw("select attempts, last_error, available_at from _outbox")
    assert rows[0]["attempts"] == 1 and "still failing" in rows[0]["last_error"]
    assert rows[0]["available_at"] > datetime.now(UTC)
    assert await worker.run_once() == 0  # not due yet

    await store.query_raw("update _outbox set available_at = now() returning id")
    assert await worker.run_once() == 1
    assert Flaky.attempts_seen == 2
    stats = await store.outbox_stats()
    assert (stats.pending, stats.dead) == (0, 1)
    assert await worker.run_once() == 0


async def test_lapsed_lease_is_reclaimed(store: PostgresStore, worker: Worker) -> None:
    await run(store, Open, {"owner": "lease"})
    claimed = await store.claim(5, "ghost-worker", lease_seconds=1.0)
    assert len(claimed) == 1
    assert await store.claim(5, "other", lease_seconds=1.0) == []
    await asyncio.sleep(1.1)
    assert await worker.run_once() == 1


async def test_two_workers_claim_disjoint_rows(store: PostgresStore, app: App) -> None:
    for i in range(6):
        await run(store, Open, {"owner": f"u{i}"})
    a = await store.claim(4, "a", lease_seconds=30)
    b = await store.claim(4, "b", lease_seconds=30)
    assert len(a) == 4 and len(b) == 2
    assert {r.id for r in a}.isdisjoint({r.id for r in b})


async def test_deferred_reactions_can_defer_further(store: PostgresStore, worker: Worker) -> None:
    await store.enqueue([OutboxEntry(reaction=qualified_name(Chain), input={})])
    assert await worker.run_once() == 1
    assert (await store.outbox_stats()).pending == 1
    assert await worker.run_once() == 1
    rows = await store.query(Account, "select * from account where owner = 'chained'")
    assert len(rows) == 1 and rows[0].notified == 1


async def test_unknown_reaction_is_released(store: PostgresStore, worker: Worker) -> None:
    await store.enqueue([OutboxEntry(reaction="nowhere.Missing", input={})])
    assert await worker.run_once() == 1
    rows = await store.query_raw("select attempts, last_error from _outbox")
    assert rows[0]["attempts"] == 1 and "Missing" in rows[0]["last_error"]


async def test_in_memory_store_has_no_outbox() -> None:
    with pytest.raises(NotSupported):
        await run(InMemoryStore(), Open, {"owner": "mem"})


def test_cli_worker_once_and_outbox(database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv("BACKEND_BUILDER_DATABASE_URL", database_url)
    store = PostgresStore(database_url, [Account])
    asyncio.run(_create_tables(store))
    asyncio.run(
        store.enqueue([OutboxEntry(reaction=qualified_name(Notify), input={"account_id": "x"})])
    )
    asyncio.run(store.close())

    result = runner.invoke(cli, ["outbox", "fixture_app"])
    assert result.exit_code == 0, result.output
    assert "pending=1 claimed=0 dead=0" in result.output

    result = runner.invoke(cli, ["worker", "fixture_app", "--once"])
    assert result.exit_code == 0, result.output
    assert result.output.strip().splitlines()[-1] == "claimed=1"
