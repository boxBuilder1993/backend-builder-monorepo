from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from backend_builder import (
    App,
    Context,
    Entity,
    InMemoryStore,
    OnCreate,
    OnSchedule,
    OutboxEntry,
    PostgresStore,
    Reaction,
    Scheduler,
    Settings,
    Worker,
    run,
)
from backend_builder.admin import create_admin_app
from backend_builder.reaction import qualified_name
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine


class Account(Entity):
    owner: str
    balance: int = 0


class Audit(Reaction):
    """Runs after every account is created."""

    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["id"])
        assert account is not None
        account.balance += 1


class Tick(Reaction):
    async def calculate(self, ctx: Context) -> None:
        pass


class Inner(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.save("inner", True)


class Open(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Account, owner=ctx.input["owner"])
        await ctx.trigger(Inner, {})
        ctx.save("note", "opened")


class Boom(Reaction):
    async def calculate(self, ctx: Context) -> None:
        raise ValueError("boom")


async def _create_tables(store: PostgresStore) -> None:
    engine = create_async_engine(store.url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(store.metadata.create_all)
    finally:
        await engine.dispose()


@pytest.fixture
def app() -> App:
    return App(
        package="fixture_app",
        include=[Account, Audit, Tick, Inner, Open, Boom],
        triggers=[OnCreate(Account, Audit), OnSchedule("* * * * *", Tick)],
        max_attempts=1,
    )


@pytest.fixture
async def pg(database_url: str, app: App) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(database_url, app.entities)
    await _create_tables(store)
    yield store
    await store.close()


@pytest.fixture
def console(app: App, pg: PostgresStore) -> TestClient:
    return TestClient(create_admin_app(app, pg))


async def _seed(app: App, pg: PostgresStore) -> tuple[str, str]:
    ok = await run(pg, Open, {"owner": "ada"}, triggers=app.triggers, source="test")
    with pytest.raises(ValueError):
        await run(pg, Boom, {}, source="test", max_attempts=1)
    await Scheduler(app, pg, Settings()).tick()
    await Worker(app, pg, Settings(outbox_batch=5)).run_once()
    failed = await pg.query_raw("select id from _run where outcome = 'error'")
    return ok.reaction_id, str(failed[0]["id"])


async def test_overview_graph_and_schedules_render(
    app: App, pg: PostgresStore, console: TestClient
) -> None:
    await _seed(app, pg)
    page = console.get("/")
    assert page.status_code == 200
    assert "reactions" in page.text and "pending" in page.text and "Recent runs" in page.text

    graph = console.get("/graph").text
    assert "Account" in graph and "OnCreate(Account)" in graph and "OnSchedule" in graph
    assert "Runs after every account is created." in graph

    schedules = console.get("/schedules").text
    assert "Tick" in schedules and "* * * * *" in schedules


async def test_run_page_shows_spans_input_vars_changes_and_deferred(
    app: App, pg: PostgresStore, console: TestClient
) -> None:
    ok_id, failed_id = await _seed(app, pg)

    page = console.get(f"/runs/{ok_id}").text
    assert "Open" in page and "↳ Inner" in page  # nested span tree
    assert '"owner": "ada"' in page and '"note": "opened"' in page
    assert "account" in page and "created" in page
    assert "delivered" in page  # the row OnCreate produced for Audit, delivered by the worker

    failed = console.get(f"/runs/{failed_id}").text
    assert "ValueError: boom" in failed and "error" in failed

    assert "No such run" in console.get("/runs/nope").text


async def test_runs_filter_and_reaction_stats(
    app: App, pg: PostgresStore, console: TestClient
) -> None:
    await _seed(app, pg)
    errors = console.get("/runs", params={"outcome": "error"}).text
    assert "Boom" in errors and "Open" not in errors.split("<table>")[1]

    page = console.get(f"/reactions/{qualified_name(Open)}").text
    assert "p95" in page and "ok" in page and "explicit triggers" in page
    assert "No such reaction" in console.get("/reactions/nowhere.X").text


async def test_outbox_pages_requeue_and_discard(
    app: App, pg: PostgresStore, console: TestClient
) -> None:
    row_id = str(uuid4())
    await pg.enqueue([OutboxEntry(id=row_id, reaction="nowhere.Missing", input={"x": 1})])
    await Worker(app, pg, Settings(outbox_batch=5, outbox_max_attempts=1)).run_once()
    assert (await pg.outbox_stats()).dead == 1

    dead = console.get("/outbox", params={"state": "dead"}).text
    assert row_id[:8] in dead
    page = console.get(f"/outbox/{row_id}").text
    assert '"x": 1' in page and "Requeue now" in page

    assert console.post(f"/outbox/{row_id}/requeue", follow_redirects=False).status_code == 303
    assert (await pg.outbox_stats()).pending == 1
    assert console.post(f"/outbox/{row_id}/discard", follow_redirects=False).status_code == 303
    assert (await pg.outbox_stats()).pending == 0
    assert "No such outbox row" in console.get(f"/outbox/{row_id}").text


def test_token_gates_every_page(app: App) -> None:
    client = TestClient(
        create_admin_app(app, InMemoryStore(), token="s3cret"), raise_server_exceptions=False
    )
    assert client.get("/graph", follow_redirects=False).status_code == 303
    assert client.get("/graph", headers={"X-Admin-Token": "s3cret"}).status_code == 200
    assert client.post("/login", data={"value": "wrong"}).status_code == 200
    assert (
        client.post("/login", data={"value": "s3cret"}, follow_redirects=False).status_code == 303
    )
    assert client.get("/graph").status_code == 200  # the cookie is kept


def test_in_memory_store_still_serves_the_graph() -> None:
    client = TestClient(create_admin_app(App(package="fixture_app"), InMemoryStore()))
    assert "keeps no run records" in client.get("/").text
    assert client.get("/graph").status_code == 200
