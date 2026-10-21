import asyncio
from collections.abc import AsyncIterator

import pytest
from backend_builder import (
    App,
    Context,
    Entity,
    InMemoryStore,
    OutboxEntry,
    PostgresStore,
    Reaction,
    Settings,
    Worker,
    metrics,
    run,
)
from backend_builder.reaction import qualified_name
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine


class Thing(Entity):
    label: str = ""


class Make(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Thing, label="made")


class Boom(Reaction):
    async def calculate(self, ctx: Context) -> None:
        raise ValueError("no")


def scrape() -> str:
    return metrics.render().decode()


def sample(text: str, metric: str, **labels: str) -> float:
    """The value of one series in scraped text, or 0 if absent. Labels are
    matched individually, in any order."""
    wanted = [f'{k}="{v}"' for k, v in labels.items()]
    for line in text.splitlines():
        if labels and line.startswith(metric + "{"):
            label_part = line.split("}")[0]
            if all(w in label_part for w in wanted):
                return float(line.rsplit(" ", 1)[1])
        if not labels and line.startswith(metric + " "):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


async def test_runs_are_counted_by_reaction_source_and_outcome() -> None:
    before = sample(
        scrape(), "bb_runs_total", reaction=qualified_name(Make), source="t", outcome="ok"
    )
    await run(InMemoryStore(), Make, {}, source="t")
    with pytest.raises(ValueError):
        await run(InMemoryStore(), Boom, {}, source="t", max_attempts=2)

    text = scrape()
    assert (
        sample(text, "bb_runs_total", reaction=qualified_name(Make), source="t", outcome="ok")
        == before + 1
    )
    assert (
        sample(text, "bb_runs_total", reaction=qualified_name(Boom), source="t", outcome="error")
        >= 1
    )
    assert sample(text, "bb_run_duration_seconds_count", reaction=qualified_name(Make)) >= 1
    assert sample(text, "bb_run_attempts_sum", reaction=qualified_name(Boom)) >= 2
    assert sample(text, "bb_span_duration_seconds_count", name="Make") >= 1


def test_http_requests_are_counted_by_route_template() -> None:
    from fixture_app import app

    client = TestClient(app.http(InMemoryStore()), raise_server_exceptions=False)
    client.post("/widgets", json={"name": "w"})
    client.post("/widgets", json={})
    client.get("/health")

    text = scrape()
    assert (
        sample(text, "bb_http_requests_total", route="/widgets", method="POST", status="200") >= 1
    )
    assert (
        sample(text, "bb_http_requests_total", route="/widgets", method="POST", status="422") >= 1
    )
    assert (
        sample(text, "bb_http_request_duration_seconds_count", route="/widgets", method="POST") >= 2
    )


def test_metrics_server_app_serves_metrics_and_health() -> None:
    async def down() -> bool:
        return False

    up = TestClient(metrics.app())
    assert up.get("/health").json() == {"status": "ok"}
    body = up.get("/metrics")
    assert body.status_code == 200 and "bb_runs_total" in body.text
    assert body.headers["content-type"].startswith("text/plain")

    assert TestClient(metrics.app(down)).get("/health").status_code == 503


async def _create_tables(store: PostgresStore) -> None:
    engine = create_async_engine(store.url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(store.metadata.create_all)
    finally:
        await engine.dispose()


@pytest.fixture
async def pg(database_url: str) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(database_url, [Thing])
    await _create_tables(store)
    yield store
    await store.close()


async def test_worker_updates_delivery_counters_and_outbox_gauges(pg: PostgresStore) -> None:
    app = App(package="fixture_app", include=[Thing, Make, Boom], max_attempts=1)
    worker = Worker(app, pg, Settings(outbox_batch=5, outbox_max_attempts=1))
    await pg.enqueue(
        [
            OutboxEntry(reaction=qualified_name(Make), input={}),
            OutboxEntry(reaction=qualified_name(Boom), input={}),
            OutboxEntry(reaction="nowhere.Missing", input={}),
        ]
    )
    assert await worker.run_once() == 3
    text = scrape()
    assert (
        sample(text, "bb_outbox_deliveries_total", reaction=qualified_name(Make), outcome="ok") >= 1
    )
    assert (
        sample(text, "bb_outbox_deliveries_total", reaction=qualified_name(Boom), outcome="error")
        >= 1
    )
    assert (
        sample(text, "bb_outbox_deliveries_total", reaction="nowhere.Missing", outcome="unknown")
        >= 1
    )

    await worker.run_once()  # refreshes the gauges after the releases
    text = scrape()
    assert sample(text, "bb_outbox_dead") == 2.0
    assert sample(text, "bb_outbox_pending") == 0.0


async def test_metrics_server_serves_on_its_port() -> None:
    import httpx2

    task = metrics.start(19199)
    text = ""
    try:
        for _ in range(50):
            try:
                async with httpx2.AsyncClient() as client:
                    res = await client.get("http://127.0.0.1:19199/metrics")
                text = res.text if res.status_code == 200 else ""
                break
            except Exception:
                await asyncio.sleep(0.1)
        assert "bb_runs_total" in text
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
