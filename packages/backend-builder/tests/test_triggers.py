from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from backend_builder import (
    App,
    Change,
    Context,
    Entity,
    InMemoryStore,
    OnChange,
    OnCreate,
    OnDelete,
    OnSchedule,
    PostgresStore,
    Reaction,
    Scheduler,
    Settings,
    TriggerError,
    Worker,
    run,
)
from backend_builder.reaction import qualified_name
from backend_builder.triggers import match
from sqlalchemy.ext.asyncio import create_async_engine


class Account(Entity):
    owner: str
    balance: int = 0
    note: str = ""


class Audit(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.save("input", ctx.input)


class Tick(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Account, owner=f"tick {ctx.input['scheduled_for']}")


class OpenTwo(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Account, owner="a")
        ctx.create(Account, owner="b")


class Edit(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["id"])
        assert account is not None
        for name, value in ctx.input["set"].items():
            setattr(account, name, value)
        if ctx.input.get("delete"):
            ctx.delete(account)


class Revert(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["id"])
        assert account is not None
        original = account.balance
        account.balance = original + 100
        account.balance = original


TRIGGERS = [
    OnCreate(Account, Audit),
    OnChange(Account, ["balance"], Audit),
    OnDelete(Account, Audit),
]


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


async def _outbox(store: PostgresStore) -> list[dict[str, object]]:
    return await store.query_raw(
        "select reaction, input, scheduled_for from _outbox order by created_at, input::text"
    )


# --- declarations ----------------------------------------------------------------


def test_on_change_validates_fields() -> None:
    with pytest.raises(TriggerError, match="no persisted field"):
        OnChange(Account, ["nope"], Audit)
    with pytest.raises(TriggerError, match="at least one field"):
        OnChange(Account, [], Audit)


def test_on_schedule_validates_cron_and_computes_next() -> None:
    with pytest.raises(TriggerError, match="not a valid cron"):
        OnSchedule("every day", Tick)
    schedule = OnSchedule("0 3 * * *", Tick)
    nxt = schedule.next_after(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    assert nxt == datetime(2026, 1, 2, 3, 0, tzinfo=UTC)
    entry = schedule.entry(nxt)
    assert entry.scheduled_for == nxt and entry.available_at == nxt
    assert entry.input == {"scheduled_for": nxt.isoformat()}


def test_app_validates_trigger_targets() -> None:
    with pytest.raises(TriggerError, match="not one of this app's reactions"):
        App(package="fixture_app", include=[Account], triggers=[OnCreate(Account, Audit)])
    with pytest.raises(TriggerError, match="not one of this app's entities"):
        App(package="fixture_app", include=[Audit], triggers=[OnCreate(Account, Audit)])
    app = App(package="fixture_app", include=[Account, Audit, Tick], triggers=TRIGGERS)
    assert app.describe()["triggers"] == [
        f"OnCreate(Account) -> {qualified_name(Audit)}",
        f"OnChange(Account, ['balance']) -> {qualified_name(Audit)}",
        f"OnDelete(Account) -> {qualified_name(Audit)}",
    ]


def test_in_memory_store_is_refused_when_triggers_are_declared() -> None:
    app = App(package="fixture_app", include=[Account, Audit], triggers=TRIGGERS)
    with pytest.raises(TriggerError, match="in-memory store has none"):
        app.http(InMemoryStore())


# --- matching ------------------------------------------------------------------------


def test_match_is_one_entry_per_change_per_trigger() -> None:
    changes = [
        Change(entity=Account, id="1", kind="created"),
        Change(entity=Account, id="2", kind="updated", changed=("note",)),
        Change(entity=Account, id="3", kind="updated", changed=("note", "balance")),
        Change(entity=Account, id="4", kind="deleted"),
    ]
    entries = match(TRIGGERS, changes)
    assert [e.input for e in entries] == [
        {"entity": "account", "id": "1", "kind": "created"},
        {"entity": "account", "id": "3", "kind": "updated", "changed": ["balance"]},
        {"entity": "account", "id": "4", "kind": "deleted"},
    ]
    assert all(e.reaction == qualified_name(Audit) for e in entries)


# --- at flush, on Postgres ---------------------------------------------------------------


async def test_creates_enqueue_one_row_each(store: PostgresStore) -> None:
    await run(store, OpenTwo, {}, triggers=TRIGGERS)
    rows = await _outbox(store)
    assert len(rows) == 2
    assert all(r["reaction"] == qualified_name(Audit) for r in rows)
    kinds = {str(r["input"]["kind"]) for r in rows}  # type: ignore[index]
    assert kinds == {"created"}


async def test_updates_match_only_listed_fields(store: PostgresStore) -> None:
    account = await store.create(Account, owner="x")
    await run(store, Edit, {"id": account.id, "set": {"note": "hi"}}, triggers=TRIGGERS)
    assert await _outbox(store) == []

    await run(
        store, Edit, {"id": account.id, "set": {"note": "yo", "balance": 5}}, triggers=TRIGGERS
    )
    (row,) = await _outbox(store)
    assert row["input"] == {
        "entity": "account",
        "id": account.id,
        "kind": "updated",
        "changed": ["balance"],
    }


async def test_delete_is_deleted_not_updated(store: PostgresStore) -> None:
    account = await store.create(Account, owner="x")
    await run(
        store, Edit, {"id": account.id, "set": {"balance": 9}, "delete": True}, triggers=TRIGGERS
    )
    (row,) = await _outbox(store)
    assert row["input"] == {"entity": "account", "id": account.id, "kind": "deleted"}


async def test_reverted_change_enqueues_nothing(store: PostgresStore) -> None:
    account = await store.create(Account, owner="x", balance=1)
    await run(store, Revert, {"id": account.id}, triggers=TRIGGERS)
    assert await _outbox(store) == []


async def test_failed_run_enqueues_nothing(store: PostgresStore) -> None:
    class Boom(OpenTwo):
        async def calculate(self, ctx: Context) -> None:
            await super().calculate(ctx)
            raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        await run(store, Boom, {}, triggers=TRIGGERS, max_attempts=1)
    assert await _outbox(store) == []


# --- schedules, through the worker ------------------------------------------------------------


async def test_scheduler_seeds_once_and_worker_delivers(store: PostgresStore) -> None:
    app = App(
        package="fixture_app",
        include=[Account, Tick],
        triggers=[OnSchedule("* * * * *", Tick)],
        max_attempts=1,
    )
    scheduler = Scheduler(app, store, Settings())
    worker = Worker(app, store, Settings(outbox_batch=5))

    assert await scheduler.tick() == 1
    assert await scheduler.tick() == 0  # last seeded occurrence is still ahead
    rows = await _outbox(store)
    assert len(rows) == 1 and rows[0]["scheduled_for"] is not None
    first: datetime = rows[0]["scheduled_for"]  # type: ignore[assignment]
    assert first > datetime.now(UTC) and first.second == 0
    record = await store.query_raw("select reaction, cron, last_scheduled_for from _schedule")
    assert record == [
        {"reaction": qualified_name(Tick), "cron": "* * * * *", "last_scheduled_for": first}
    ]

    await store.query_raw("update _outbox set available_at = now() returning id")
    assert await worker.run_once() == 1
    ticks = await store.query(Account, "select * from account")
    assert [t.owner for t in ticks] == [f"tick {first.isoformat()}"]

    # The worker does not re-seed; the scheduler does, once the seeded time has passed.
    assert await _outbox(store) == []
    assert await scheduler.tick() == 0
    await store.query_raw(
        "update _schedule set last_scheduled_for = now() - interval '1 minute' returning reaction"
    )
    assert await scheduler.tick() == 1
    (nxt,) = await _outbox(store)
    later: datetime = nxt["scheduled_for"]  # type: ignore[assignment]
    assert later > datetime.now(UTC)


async def test_scheduler_reseeds_when_the_cron_changes(store: PostgresStore) -> None:
    app = App(
        package="fixture_app",
        include=[Account, Tick],
        triggers=[OnSchedule("* * * * *", Tick)],
    )
    assert await Scheduler(app, store, Settings()).tick() == 1
    changed = App(
        package="fixture_app",
        include=[Account, Tick],
        triggers=[OnSchedule("*/5 * * * *", Tick)],
    )
    assert await Scheduler(changed, store, Settings()).tick() == 1
    record = await store.query_raw("select cron from _schedule")
    assert record == [{"cron": "*/5 * * * *"}]
