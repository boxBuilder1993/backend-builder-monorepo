import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from backend_builder import (
    App,
    Context,
    Entity,
    InMemoryStore,
    OnCreate,
    PostgresStore,
    Reaction,
    RunRecord,
    Scheduler,
    Settings,
    run,
)
from backend_builder.reaction import qualified_name
from sqlalchemy.ext.asyncio import create_async_engine


class Account(Entity):
    owner: str
    balance: int = 0


class Audit(Reaction):
    async def calculate(self, ctx: Context) -> None:
        pass


class Inner(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["id"])
        assert account is not None
        account.balance += 5
        ctx.save("touched", account)


class OuterMem(Reaction):
    """Creates, triggers immediately, saves; no outbox needed."""

    async def calculate(self, ctx: Context) -> None:
        created = ctx.create(Account, owner=ctx.input["owner"])
        await ctx.trigger(Inner, {"id": created.id})
        ctx.save("note", "done")
        ctx.save("opaque", object())


class Outer(OuterMem):
    """The same, plus a deferred trigger: Postgres only."""

    async def calculate(self, ctx: Context) -> None:
        await super().calculate(ctx)
        await ctx.trigger(Audit, {"note": "later"}, deferred=True)


class FlakyThenOk(Reaction):
    calls: int = 0

    async def calculate(self, ctx: Context) -> None:
        FlakyThenOk.calls += 1
        ctx.create(Account, owner="flaky")
        if FlakyThenOk.calls == 1:
            raise RuntimeError("first attempt")


class Explode(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Account, owner="ghost")
        raise ValueError("boom")


class Slow(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await asyncio.sleep(1.0)


class RecordingBroken(InMemoryStore):
    async def record_run(self, record: RunRecord) -> None:
        raise OSError("disk full")


async def _create_tables(store: PostgresStore) -> None:
    engine = create_async_engine(store.url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(store.metadata.create_all)
    finally:
        await engine.dispose()


@pytest.fixture
async def pg(database_url: str) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(database_url, [Account])
    await _create_tables(store)
    yield store
    await store.close()


# --- in memory ------------------------------------------------------------------------


async def test_successful_run_is_recorded_in_full() -> None:
    store = InMemoryStore()
    ctx = await run(store, OuterMem, {"owner": "ada"}, source="test")

    (record,) = store.records
    assert record.id == ctx.reaction_id
    assert record.reaction == qualified_name(OuterMem) and record.source == "test"
    assert record.outcome == "ok" and record.error is None and record.attempts == 1
    assert record.started_at.tzinfo is UTC and record.duration_ms >= 0
    assert record.input == {"owner": "ada"}
    assert record.vars is not None
    assert record.vars["note"] == "done"
    assert record.vars["touched"]["owner"] == "ada" and record.vars["touched"]["balance"] == 5
    assert record.vars["opaque"].startswith("<object object")
    assert record.changes == [
        {"entity": "account", "id": record.vars["touched"]["id"], "kind": "created", "changed": []}
    ]
    assert record.enqueued == []
    names = [(s.attempt, s.name) for s in record.spans]
    assert names == [(1, "Inner"), (1, "OuterMem"), (1, "flush"), (1, "run")]
    assert all(s.ok for s in record.spans)


async def test_summary_level_omits_input_and_vars() -> None:
    store = InMemoryStore()
    await run(store, OuterMem, {"owner": "ada"}, records="summary")
    (record,) = store.records
    assert record.input is None and record.vars is None
    assert record.changes and record.spans


async def test_off_records_nothing() -> None:
    store = InMemoryStore()
    await run(store, OuterMem, {"owner": "ada"}, records="off")
    assert store.records == []


async def test_failed_run_is_recorded_with_its_error_and_no_changes() -> None:
    store = InMemoryStore()
    with pytest.raises(ValueError):
        await run(store, Explode, {}, max_attempts=2)
    (record,) = store.records
    assert record.outcome == "error" and record.error == "ValueError: boom"
    assert record.attempts == 2 and record.changes == []
    assert [(s.attempt, s.name, s.ok) for s in record.spans] == [
        (1, "Explode", False),
        (1, "run", False),
        (2, "Explode", False),
        (2, "run", False),
    ]


async def test_retried_run_keeps_every_attempts_spans() -> None:
    FlakyThenOk.calls = 0
    store = InMemoryStore()
    await run(store, FlakyThenOk, {}, max_attempts=3)
    (record,) = store.records
    assert record.outcome == "ok" and record.attempts == 2
    assert [s.attempt for s in record.spans] == [1, 1, 2, 2, 2]
    assert len(record.changes) == 1  # only the committed attempt's write


async def test_timed_out_run_is_recorded() -> None:
    store = InMemoryStore()
    with pytest.raises(TimeoutError):
        await run(store, Slow, {}, timeout=0.1)
    (record,) = store.records
    assert record.outcome == "timeout" and record.error is not None
    assert "0.1" in record.error


async def test_recording_failure_never_fails_the_run(caplog: pytest.LogCaptureFixture) -> None:
    store = RecordingBroken()
    ctx = await run(store, OuterMem, {"owner": "x"})
    assert ctx.load("note") == "done"
    assert any("run record failed" in r.getMessage() for r in caplog.records)


async def test_in_memory_purge() -> None:
    store = InMemoryStore()
    await run(store, OuterMem, {"owner": "x"})
    assert await store.purge_runs(datetime.now(UTC) - timedelta(days=1)) == 0
    assert await store.purge_runs(datetime.now(UTC) + timedelta(seconds=1)) == 1
    assert store.records == []


# --- postgres -----------------------------------------------------------------------------


async def test_postgres_records_runs_with_spans_and_enqueued_rows(pg: PostgresStore) -> None:
    ctx = await run(pg, Outer, {"owner": "ada"}, triggers=[OnCreate(Account, Audit)], source="t")

    (row,) = await pg.query_raw("select * from _run")
    assert row["id"] == ctx.reaction_id and row["outcome"] == "ok" and row["source"] == "t"
    assert row["input"] == {"owner": "ada"} and row["vars"]["note"] == "done"
    assert [c["kind"] for c in row["changes"]] == ["created"]
    assert len(row["enqueued"]) == 2  # the explicit deferred Audit and the OnCreate match
    outbox = await pg.query_raw("select id from _outbox")
    assert {r["id"] for r in outbox} == set(row["enqueued"])

    spans = await pg.query_raw(
        "select attempt, name, ok from _span where run_id = :id order by started_at",
        id=ctx.reaction_id,
    )
    assert [s["name"] for s in spans] == ["run", "Outer", "Inner", "flush"]
    assert all(s["ok"] and s["attempt"] == 1 for s in spans)


async def test_postgres_records_failures_outside_the_rolled_back_transaction(
    pg: PostgresStore,
) -> None:
    with pytest.raises(ValueError):
        await run(pg, Explode, {}, max_attempts=1)
    assert await pg.query_raw("select count(*) as n from account") == [{"n": 0}]
    (row,) = await pg.query_raw("select outcome, error, changes from _run")
    assert row["outcome"] == "error" and row["error"] == "ValueError: boom" and row["changes"] == []


async def test_scheduler_purges_old_records(pg: PostgresStore) -> None:
    await run(pg, Outer, {"owner": "old"})
    await run(pg, Outer, {"owner": "new"})
    await pg.query_raw(
        "update _run set started_at = now() - interval '10 days' "
        "where input->>'owner' = 'old' returning id"
    )
    app = App(package="fixture_app", include=[Account, Outer, Inner, Audit])
    scheduler = Scheduler(app, pg, Settings(run_retention_days=7))
    assert await scheduler.purge_if_due() == 1
    assert await scheduler.purge_if_due() == 0  # not due again for an hour
    remaining = await pg.query_raw("select input->>'owner' as owner from _run")
    assert remaining == [{"owner": "new"}]
    assert await pg.query_raw("select count(*) as n from _span") != [{"n": 0}]
