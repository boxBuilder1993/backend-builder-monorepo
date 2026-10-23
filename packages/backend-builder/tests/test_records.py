import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

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
        inner = await ctx.trigger(Inner, {"id": created.id})
        assert inner is not None
        ctx.save("touched", inner.load("touched"))
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
        created = ctx.create(Account, owner="flaky")
        await ctx.trigger(Inner, {"id": created.id})
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
    async def record_run(self, records: Sequence[RunRecord]) -> None:
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

    inner, record = store.records  # one row per invocation, children first
    assert record.id == ctx.reaction_id and record.is_root and record.root_id == record.id
    assert record.reaction == qualified_name(OuterMem) and record.source == "test"
    assert record.outcome == "ok" and record.error is None and record.attempt == 1
    assert record.committed and record.flush_ms is not None
    assert record.started_at.tzinfo is UTC and record.duration_ms >= 0
    assert record.input == {"owner": "ada"}
    assert record.vars is not None
    assert record.vars["note"] == "done"
    assert record.vars["touched"]["owner"] == "ada" and record.vars["touched"]["balance"] == 5
    assert record.vars["opaque"].startswith("<object object")
    account_id = record.vars["touched"]["id"]
    assert record.changes == [
        {"entity": "account", "id": account_id, "kind": "created", "changed": []}
    ]
    assert record.enqueued == []

    assert inner.reaction == qualified_name(Inner) and inner.source == "trigger"
    assert inner.root_id == record.id and inner.parent_reaction_id == record.id
    assert inner.attempt == 1 and inner.outcome == "ok" and inner.committed
    assert inner.input == {"id": account_id}
    assert inner.vars == {
        "touched": record.vars["touched"]
    }  # its own save, nothing of the parent's
    assert inner.changes == [] and inner.flush_ms is None and inner.deadline_at is None


async def test_summary_level_omits_input_and_vars() -> None:
    store = InMemoryStore()
    await run(store, OuterMem, {"owner": "ada"}, records="summary")
    inner, record = store.records
    assert record.input is None and record.vars is None and inner.input is None
    assert record.changes


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
    assert record.attempt == 2 and record.changes == [] and not record.committed


async def test_retried_run_keeps_every_attempts_invocations() -> None:
    FlakyThenOk.calls = 0
    store = InMemoryStore()
    ctx = await run(store, FlakyThenOk, {}, max_attempts=3)
    root = next(r for r in store.records if r.is_root)
    assert root.outcome == "ok" and root.attempt == 2 and root.committed
    assert len(root.changes) == 1  # only the committed attempt's write
    children = [r for r in store.records if not r.is_root]
    assert [(c.attempt, c.committed) for c in children] == [(1, False), (2, True)]
    assert all(c.root_id == ctx.reaction_id and c.outcome == "ok" for c in children)


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
    assert (
        await store.purge_runs(datetime.now(UTC) + timedelta(seconds=1)) == 1
    )  # one run, two rows
    assert store.records == []


async def test_a_child_that_raises_is_recorded_even_when_the_parent_recovers() -> None:
    class Fragile(Reaction):
        async def calculate(self, ctx: Context) -> None:
            raise ValueError("nope")

    class Tolerant(Reaction):
        async def calculate(self, ctx: Context) -> None:
            try:
                await ctx.trigger(Fragile, {"n": 1})
            except ValueError:
                ctx.save("recovered", True)

    store = InMemoryStore()
    await run(store, Tolerant, {})
    child, root = store.records
    assert root.outcome == "ok" and root.vars == {"recovered": True}
    assert child.outcome == "error" and child.error == "ValueError: nope" and child.committed


# --- postgres -----------------------------------------------------------------------------


async def test_postgres_records_every_invocation_and_the_enqueued_rows(pg: PostgresStore) -> None:
    ctx = await run(pg, Outer, {"owner": "ada"}, triggers=[OnCreate(Account, Audit)], source="t")

    (row,) = await pg.query_raw("select * from _run where id = root_id")
    assert row["id"] == ctx.reaction_id and row["outcome"] == "ok" and row["source"] == "t"
    assert row["committed"] is True and row["flush_ms"] is not None and row["attempt"] == 1
    assert row["input"] == {"owner": "ada"} and row["vars"]["note"] == "done"
    assert [c["kind"] for c in row["changes"]] == ["created"]
    assert len(row["enqueued"]) == 2  # the explicit deferred Audit and the OnCreate match
    outbox = await pg.query_raw("select id from _outbox")
    assert {r["id"] for r in outbox} == set(row["enqueued"])

    (child,) = await pg.query_raw(
        "select reaction, source, parent_reaction_id, attempt, outcome, committed, input, vars, "
        "changes, enqueued from _run where root_id = :id and id <> root_id",
        id=ctx.reaction_id,
    )
    assert child["reaction"] == qualified_name(Inner) and child["source"] == "trigger"
    assert child["parent_reaction_id"] == ctx.reaction_id and child["attempt"] == 1
    assert child["outcome"] == "ok" and child["committed"] is True
    assert child["input"] == {"id": row["changes"][0]["id"]}
    assert set(child["vars"]) == {"touched"} and child["changes"] == [] and child["enqueued"] == []


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
    remaining = await pg.query_raw("select input->>'owner' as owner from _run where id = root_id")
    assert remaining == [{"owner": "new"}]
    assert await pg.query_raw("select count(*) as n from _run where id <> root_id") == [{"n": 1}]


# --- running and lost -----------------------------------------------------------------


async def test_a_running_record_exists_while_the_run_executes_and_is_replaced() -> None:
    store = InMemoryStore()
    seen: list[tuple[str, int]] = []

    class Peek(Reaction):
        async def calculate(self, ctx: Context) -> None:
            seen.append((store.records[0].outcome, len(store.records)))
            assert store.records[0].id == ctx.reaction_id
            assert store.records[0].deadline_at is not None

    ctx = await run(store, Peek, {"k": "v"}, timeout=5, source="test")
    assert seen == [("running", 1)]
    (record,) = store.records
    assert record.id == ctx.reaction_id and record.outcome == "ok"
    assert record.deadline_at == record.started_at + timedelta(seconds=5)


async def test_running_records_past_their_deadline_are_marked_lost() -> None:
    store = InMemoryStore()
    stale_id, fresh_id = str(uuid4()), str(uuid4())
    stale = RunRecord(
        id=stale_id,
        root_id=stale_id,
        parent_reaction_id=None,
        reaction="x.Dead",
        source="test",
        attempt=0,
        outcome="running",
        error=None,
        committed=False,
        started_at=datetime.now(UTC) - timedelta(minutes=5),
        duration_ms=0,
        flush_ms=None,
        input=None,
        vars=None,
        changes=[],
        enqueued=[],
        deadline_at=datetime.now(UTC) - timedelta(minutes=4),
    )
    fresh = replace(
        stale, id=fresh_id, root_id=fresh_id, deadline_at=datetime.now(UTC) + timedelta(minutes=1)
    )
    await store.record_run([stale, fresh])
    app = App(package="fixture_app", include=[Account, OuterMem, Inner])
    assert await Scheduler(app, store, Settings()).sweep_lost() == 1
    by_id = {r.id: r for r in store.records}
    assert by_id[stale.id].outcome == "lost" and "stopped reporting" in (
        by_id[stale.id].error or ""
    )
    assert by_id[fresh.id].outcome == "running"


async def test_postgres_shows_the_running_row_mid_run_and_sweeps_lost_ones(
    pg: PostgresStore,
) -> None:
    class PeekPg(Reaction):
        async def calculate(self, ctx: Context) -> None:
            rows = await ctx.query_raw(
                "select outcome, deadline_at from _run where id = :id", id=ctx.reaction_id
            )
            assert rows[0]["outcome"] == "running" and rows[0]["deadline_at"] is not None
            ctx.create(Account, owner="peeked")

    ctx = await run(pg, PeekPg, {}, source="test")
    (row,) = await pg.query_raw(
        "select outcome, changes from _run where id = :id", id=ctx.reaction_id
    )
    assert row["outcome"] == "ok" and len(row["changes"]) == 1
    assert await pg.query_raw("select count(*) as n from _run") == [{"n": 1}]

    await pg.query_raw(
        "update _run set outcome = 'running', deadline_at = now() - interval '2 minutes' "
        "where id = :id returning id",
        id=ctx.reaction_id,
    )
    app = App(package="fixture_app", include=[Account, Outer, Inner, Audit])
    assert await Scheduler(app, pg, Settings()).sweep_lost() == 1
    (row,) = await pg.query_raw(
        "select outcome, error from _run where id = :id", id=ctx.reaction_id
    )
    assert row["outcome"] == "lost" and "stopped reporting" in row["error"]
