"""The gold-standard suite: every component of the shop, in its container,
exercised through the API and observed through the database and the logs."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx2 import AsyncClient

from .conftest import NOTIFY_KEY, Stack, wait_for

pytestmark = pytest.mark.e2e

LINES: list[dict[str, int]] = [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}]


@pytest.fixture
async def client(stack: Stack) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(base_url=stack.base_url, timeout=10) as client:
        yield client


async def create_order(client: AsyncClient, lines: list[dict[str, int]] = LINES) -> dict[str, Any]:
    res = await client.post(
        "/orders", json={"user_id": "e2e", "lines": lines}, headers={"X-Tenant-Id": "acme"}
    )
    assert res.status_code == 200, res.text
    UUID(res.headers["X-Reaction-Id"])
    return res.json()


async def get_order(client: AsyncClient, order_id: str) -> dict[str, Any]:
    res = await client.get(f"/orders/{order_id}", params={"include_lines": "true"})
    assert res.status_code == 200, res.text
    return res.json()


async def wait_total(client: AsyncClient, order_id: str, expected: int) -> dict[str, Any]:
    return await wait_for(
        lambda: get_order(client, order_id), lambda data: data["order"]["total"] == expected
    )


# --- the stack itself ----------------------------------------------------------------


async def test_every_service_is_up_and_migrated(stack: Stack, client: AsyncClient) -> None:
    status = stack.status()
    assert "healthy" in status["api"]
    assert "healthy" in status["worker"] and "healthy" in status["scheduler"]
    assert status["migrate"].startswith("Exited (0)")

    res = await client.get("/health")
    assert res.json() == {"status": "ok", "store": "PostgresStore"}

    assert "serving app=shop env=prod store=PostgresStore" in stack.logs("api")
    assert "worker started" in stack.logs("worker")
    assert "scheduler started app=shop schedules=1" in stack.logs("scheduler")

    check = stack.cli("migrate", "shop", "--check")
    assert check.returncode == 0 and "entities match migrations" in check.stdout


async def test_openapi_documents_the_shop(client: AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    assert schema["info"]["title"] == "shop"
    assert set(schema["paths"]) >= {
        "/orders",
        "/orders/{order_id}",
        "/orders/{order_id}/lines/{line_id}",
        "/reactions/{name}",
    }
    assert {"CreateOrderBody", "OrderResponse", "LineResponse", "ErrorOut"} <= set(
        schema["components"]["schemas"]
    )
    get = schema["paths"]["/orders/{order_id}"]["get"]
    assert get["responses"]["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "ErrorOut"
    )


# --- the request path: endpoints, API reactions, flush ---------------------------------


async def test_create_order_commits_lines_and_the_worker_totals_it(
    stack: Stack, client: AsyncClient
) -> None:
    created = await create_order(client)
    order = created["order"]
    assert order["total"] == 0 and order["status"] == "new" and order["tenant"] == "acme"
    assert len(created["lines"]) == 2

    totalled = await wait_total(client, order["id"], 25)
    assert len(totalled["lines"]) == 2

    rows = await stack.sql('select total, status from "order" where id = :id', id=order["id"])
    assert rows == [{"total": 25, "status": "new"}]
    await wait_for(  # the order's own NotifyOrder row may land a poll after the total
        lambda: stack.sql(
            "select count(*) as n from _outbox where scheduled_for is null and dead_at is null"
        ),
        lambda rows: rows == [{"n": 0}],
    )
    assert "worker delivered" in stack.logs("worker")


async def test_secret_reaches_the_reaction_and_never_the_record_or_logs(
    stack: Stack, client: AsyncClient
) -> None:
    order = (await create_order(client))["order"]
    records = await wait_for(
        lambda: stack.sql(
            "select vars, outcome from _run where reaction = 'shop.reactions.NotifyOrder' "
            "and input ->> 'id' = :id",
            id=order["id"],
        ),
        lambda rows: (
            len(rows) == 1 and rows[0]["outcome"] != "running"
        ),  # the row exists while it runs
    )
    assert records[0]["outcome"] == "ok"
    assert records[0]["vars"] == {
        "request": {"authorization": "Bearer [secret:notify_api_key]", "order": order["id"]}
    }
    assert NOTIFY_KEY not in stack.logs("worker")
    assert NOTIFY_KEY not in stack.logs("api")


async def test_update_line_retotals_through_on_change(stack: Stack, client: AsyncClient) -> None:
    order = (await create_order(client))["order"]
    await wait_total(client, order["id"], 25)
    line = (await get_order(client, order["id"]))["lines"][0]

    res = await client.patch(f"/orders/{order['id']}/lines/{line['id']}", json={"qty": 10})
    assert res.status_code == 200, res.text
    assert res.json()["line"]["qty"] == 10 and res.json()["deleted"] is False

    expected = line["price"] * 10 + sum(
        item["price"] * item["qty"]
        for item in (await get_order(client, order["id"]))["lines"]
        if item["id"] != line["id"]
    )
    await wait_total(client, order["id"], expected)


async def test_delete_line_retotals_through_on_delete(stack: Stack, client: AsyncClient) -> None:
    order = (await create_order(client))["order"]
    await wait_total(client, order["id"], 25)
    lines = (await get_order(client, order["id"]))["lines"]
    victim = next(item for item in lines if item["price"] == 10)

    res = await client.delete(f"/orders/{order['id']}/lines/{victim['id']}")
    assert res.status_code == 200, res.text
    assert res.json()["deleted"] is True

    after = await wait_total(client, order["id"], 5)
    assert [item["id"] for item in after["lines"]] == [
        item["id"] for item in lines if item["id"] != victim["id"]
    ]
    gone = await stack.sql("select deleted_at from order_line where id = :id", id=victim["id"])
    assert gone[0]["deleted_at"] is not None


async def test_ten_concurrent_orders_all_get_totalled(client: AsyncClient) -> None:
    created = await asyncio.gather(*(create_order(client) for _ in range(10)))
    await asyncio.gather(*(wait_total(client, c["order"]["id"], 25) for c in created))


# --- errors -------------------------------------------------------------------------------


async def test_validation_errors_name_their_source(client: AsyncClient) -> None:
    res = await client.post("/orders", json={"user_id": "e2e"})
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["body", "lines"]

    res = await client.get("/orders/x", params={"include_lines": "maybe"})
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["query", "include_lines"]


async def test_missing_order_is_a_404_reply_from_a_committed_run(
    stack: Stack, client: AsyncClient
) -> None:
    missing = str(uuid4())
    before = await stack.sql('select count(*) as n from "order"')

    res = await client.get(f"/orders/{missing}")
    assert res.status_code == 404
    assert res.json() == {"error": "NotFound", "message": f"no order {missing}"}
    reaction_id = res.headers["X-Reaction-Id"]
    assert await stack.sql('select count(*) as n from "order"') == before
    assert f"run finished reaction_id={reaction_id}" in stack.logs("api")
    assert "run gave up" not in stack.logs("api")


async def test_run_failure_is_500_with_correlation_id_and_no_writes(
    stack: Stack, client: AsyncClient
) -> None:
    # A line id on the wrong order: UpdateLine raises, so this is a genuine failure.
    order = (await create_order(client))["order"]
    other = (await create_order(client))["order"]
    line = (await get_order(client, order["id"]))["lines"][0]

    res = await client.patch(f"/orders/{other['id']}/lines/{line['id']}", json={"qty": 3})
    assert res.status_code == 500
    body = res.json()
    assert body["error"] == "LookupError" and line["id"] in body["message"]
    assert body["reaction_id"] == res.headers["X-Reaction-Id"]
    assert f"run gave up reaction=UpdateLineApi reaction_id={body['reaction_id']}" in stack.logs(
        "api"
    )


async def test_generic_trigger_endpoint_is_scoped_to_the_app(
    stack: Stack, client: AsyncClient
) -> None:
    order = (await create_order(client))["order"]
    await wait_total(client, order["id"], 25)
    line = (await get_order(client, order["id"]))["lines"][0]

    res = await client.post(
        "/reactions/shop.reactions.OrderTotal",
        json={"entity": "order_line", "id": line["id"], "kind": "updated", "changed": ["qty"]},
    )
    assert res.status_code == 200 and res.json() == {}
    UUID(res.headers["X-Reaction-Id"])

    res = await client.post("/reactions/nowhere.Missing", json={})
    assert res.status_code == 404 and res.json()["error"] == "LookupError"


# --- the scheduler and scheduled delivery ---------------------------------------------------


async def test_scheduler_seeds_the_schedule_and_the_worker_delivers_it(
    stack: Stack, client: AsyncClient
) -> None:
    record = await wait_for(
        lambda: stack.sql("select reaction, cron, last_scheduled_for from _schedule"),
        lambda rows: len(rows) == 1,
    )
    assert record[0]["reaction"] == "shop.reactions.ExpireUnpaidOrders"
    assert record[0]["cron"] == "* * * * *"

    pending = await wait_for(
        lambda: stack.sql(
            "select id, scheduled_for, available_at from _outbox "
            "where scheduled_for is not null and dead_at is null"
        ),
        lambda rows: len(rows) >= 1,
    )
    assert pending[0]["scheduled_for"] == record[0]["last_scheduled_for"]

    # An order created two hours ago, still new: due to expire.
    stale = (await create_order(client))["order"]
    await wait_total(client, stale["id"], 25)
    await stack.sql(
        "update \"order\" set created_at = now() - interval '2 hours' where id = :id",
        id=stale["id"],
    )
    fresh = (await create_order(client))["order"]

    # Make the scheduled occurrence due now rather than waiting for the minute.
    await stack.sql("update _outbox set available_at = now() where id = :id", id=pending[0]["id"])
    expired = await wait_for(
        lambda: get_order(client, stale["id"]), lambda data: data["order"]["status"] == "expired"
    )
    assert expired["order"]["total"] == 25
    assert (await get_order(client, fresh["id"]))["order"]["status"] == "new"
    assert "worker delivered" in stack.logs("worker")
    assert "reaction=ExpireUnpaidOrders" in stack.logs("worker")

    # The scheduler, not the worker, seeds the next occurrence, once the last one has passed.
    await stack.sql(
        "update _schedule set last_scheduled_for = now() - interval '1 minute' where reaction = :r",
        r="shop.reactions.ExpireUnpaidOrders",
    )
    reseeded = await wait_for(
        lambda: stack.sql(
            "select scheduled_for from _outbox where scheduled_for is not null and dead_at is null "
            "and available_at > now()"
        ),
        lambda rows: len(rows) >= 1,
    )
    assert reseeded[0]["scheduled_for"] is not None
    assert "scheduler seeded app=shop rows=1" in stack.logs("scheduler")


# --- metrics ----------------------------------------------------------------------------------


async def test_every_process_serves_metrics_on_its_own_port(
    stack: Stack, client: AsyncClient
) -> None:
    order = (await create_order(client))["order"]
    await wait_total(client, order["id"], 25)

    async with AsyncClient(timeout=5) as scraper:
        api = (await scraper.get(stack.metrics_url("api"))).text
        worker = (await scraper.get(stack.metrics_url("worker"))).text
        scheduler = (await scraper.get(stack.metrics_url("scheduler"))).text
        for service in ("api", "worker", "scheduler"):
            health = await scraper.get(stack.metrics_url(service).replace("/metrics", "/health"))
            assert health.json() == {"status": "ok"}

    create_run = 'reaction="shop.api.CreateOrderApi",source="endpoint POST /orders"'
    assert f'bb_runs_total{{outcome="ok",{create_run}}}' in api
    assert 'bb_http_requests_total{method="POST",route="/orders",status="200"}' in api
    assert 'bb_outbox_deliveries_total{outcome="ok",reaction="shop.reactions.OrderTotal"}' in worker
    assert "bb_outbox_pending " in worker and "bb_outbox_oldest_pending_seconds " in worker
    assert (
        'bb_runs_total{outcome="ok",reaction="shop.reactions.OrderTotal",source="outbox"}' in worker
    )
    assert "bb_schedule_seeds_total " in scheduler


# --- the admin console ----------------------------------------------------------------------------


async def test_admin_console_shows_the_run_its_spans_and_the_outbox(
    stack: Stack, client: AsyncClient
) -> None:
    order = (await create_order(client))["order"]
    await wait_total(client, order["id"], 25)

    async with AsyncClient(base_url=stack.admin_url, timeout=10) as admin:
        for path in ("/", "/graph", "/runs", "/outbox", "/outbox?state=dead", "/schedules"):
            page = await admin.get(path)
            assert page.status_code == 200, path
        graph = (await admin.get("/graph")).text
        assert "OrderTotal" in graph and "POST /orders" in graph and "OnSchedule" in graph

        runs = (await admin.get("/runs", params={"reaction": "shop.api.CreateOrderApi"})).text
        assert "CreateOrderApi" in runs
        run_rows = await stack.sql(
            "select id from _run where reaction = 'shop.api.CreateOrderApi' "
            "order by started_at desc limit 1"
        )
        page = (await admin.get(f"/runs/{run_rows[0]['id']}")).text
        assert "CreateOrder" in page and "ok" in page
        assert "order_line" in page and "created" in page  # changes written
        assert '"user_id": "e2e"' in page  # input recorded in full

        reaction_page = (await admin.get("/reactions/shop.reactions.OrderTotal")).text
        assert "OnCreate(OrderLine)" in reaction_page and "p95" in reaction_page

        schedules = (await admin.get("/schedules")).text
        assert "ExpireUnpaidOrders" in schedules

        # Trigger a reaction from the console: a run with source admin, landing on its page.
        triggered = await admin.post(
            "/reactions/shop.reactions.CreateOrder/trigger",
            data={
                "input": json.dumps(
                    {"user_id": "console", "tenant": "acme", "lines": [{"price": 3, "qty": 2}]}
                ),
                "mode": "run",
            },
        )
        assert triggered.status_code == 303
        page = (await admin.get(triggered.headers["location"])).text
        assert "admin" in page and '"user_id": "console"' in page and "ok" in page
        orders = await stack.sql("select total from \"order\" where user_id = 'console'")
        assert len(orders) == 1

        missing = await admin.get("/runs/not-a-run")
        assert missing.status_code == 200 and "No such run" in missing.text


async def test_admin_console_requeues_a_dead_row(stack: Stack) -> None:
    row_id = str(uuid4())
    await stack.sql(
        "insert into _outbox (id, reaction, input, created_at, available_at, attempts, "
        "dead_at, last_error) values (:id, 'shop.reactions.OrderTotal', "
        '\'{"id": "x", "kind": "updated"}\'::jsonb, now(), now(), 2, now(), \'gave up\')',
        id=row_id,
    )
    async with AsyncClient(base_url=stack.admin_url, timeout=10) as admin:
        dead = (await admin.get("/outbox", params={"state": "dead"})).text
        assert row_id[:8] in dead
        page = (await admin.get(f"/outbox/{row_id}")).text
        assert "gave up" in page and "Requeue now" in page

        res = await admin.post(f"/outbox/{row_id}/requeue")
        assert res.status_code == 303
        rows = await stack.sql("select dead_at, attempts from _outbox where id = :id", id=row_id)
        assert rows[0]["dead_at"] is None and rows[0]["attempts"] == 0

        # The worker picks it up; the input names a missing line, so it fails and dies again.
        await wait_for(
            lambda: stack.sql("select dead_at from _outbox where id = :id", id=row_id),
            lambda r: bool(r) and r[0]["dead_at"] is not None,
            timeout=30,
        )
        res = await admin.post(f"/outbox/{row_id}/discard")
        assert res.status_code == 303
        assert await stack.sql("select id from _outbox where id = :id", id=row_id) == []


# --- failure handling in the worker -----------------------------------------------------------


async def test_undeliverable_row_is_retried_with_backoff_then_marked_dead(stack: Stack) -> None:
    row_id = str(uuid4())
    await stack.sql(
        "insert into _outbox (id, reaction, input, created_at, available_at) "
        "values (:id, 'shop.reactions.Nope', '{}'::jsonb, now(), now())",
        id=row_id,
    )
    dead = await wait_for(
        lambda: stack.sql(
            "select attempts, last_error, dead_at from _outbox where id = :id", id=row_id
        ),
        lambda rows: rows[0]["dead_at"] is not None,
        timeout=30,
    )
    assert dead[0]["attempts"] == 2 and "Nope" in dead[0]["last_error"]

    stats = stack.cli("outbox", "shop")
    assert stats.returncode == 0 and "dead=1" in stats.stdout
    assert "worker unknown reaction" in stack.logs("worker")


async def test_worker_restart_picks_up_where_it_left_off(stack: Stack, client: AsyncClient) -> None:
    from .conftest import compose

    compose("stop", "worker")
    order = (await create_order(client))["order"]
    pending = await stack.sql(
        "select count(*) as n from _outbox where scheduled_for is null and dead_at is null"
    )
    assert pending[0]["n"] >= 2
    assert (await get_order(client, order["id"]))["order"]["total"] == 0

    compose("start", "worker")
    await wait_total(client, order["id"], 25)


# --- logs ---------------------------------------------------------------------------------------


async def test_logs_are_structured_collected_by_loki_and_shown_per_run(
    stack: Stack, client: AsyncClient
) -> None:
    order = (await create_order(client))["order"]
    await wait_total(client, order["id"], 25)
    notify = await wait_for(
        lambda: stack.sql(
            "select id, root_id from _run where reaction = 'shop.reactions.NotifyOrder' "
            "and input ->> 'id' = :id and outcome = 'ok'",
            id=order["id"],
        ),
        lambda rows: len(rows) == 1,
    )
    run_id = str(notify[0]["id"])

    # Containers write JSON lines carrying the run's fields.
    worker_lines = [
        json.loads(line) for line in stack.logs("worker").splitlines() if line.startswith("{")
    ]
    mine = [line for line in worker_lines if line.get("root_id") == run_id]
    assert any(line["message"].startswith("notify requested order=") for line in mine)
    assert all(line["app"] == "shop" and line["process"] == "worker" for line in mine)
    assert all(line["source"] == "outbox" for line in mine)
    assert not any(NOTIFY_KEY in json.dumps(line) for line in worker_lines)

    # Loki has them, filtered by root_id from inside the line.
    async with AsyncClient(base_url=stack.loki_url, timeout=10) as loki:
        found = await wait_for(
            lambda: loki.get(
                "/loki/api/v1/query_range",
                params={
                    "query": f'{{app="shop"}} | json | root_id="{run_id}"',
                    "start": str(int((time.time() - 600) * 1e9)),
                    "limit": "50",
                },
            ),
            lambda r: r.status_code == 200 and r.json()["data"]["result"],
            timeout=60,
        )
        values = [v for stream in found.json()["data"]["result"] for v in stream["values"]]
        assert any("notify requested" in v[1] for v in values)
        labels = await loki.get("/loki/api/v1/labels")
        assert set(labels.json()["data"]) <= {"app", "process", "service_name"}  # never ids

    # The console shows the same lines on the run's page.
    async with AsyncClient(base_url=stack.admin_url, timeout=10) as admin:
        page = await wait_for(
            lambda: admin.get(f"/runs/{run_id}"),
            lambda r: "notify requested order=" in r.text,
            timeout=60,
        )
        assert "open in Grafana" in page.text and f'root_id="{run_id}"' in page.text.replace(
            "&#34;", '"'
        )
