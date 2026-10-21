from collections.abc import AsyncIterator
from typing import Any

import pytest
from backend_builder import PostgresStore, Worker, migrations
from httpx2 import ASGITransport, AsyncClient
from shop import app
from shop.entities import Order, OrderLine


@pytest.fixture
async def client(store: PostgresStore) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app.http(store), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://shop") as client:
        yield client


async def count(store: PostgresStore, table: str) -> int:
    rows = await store.query_raw(f'select count(*) as n from "{table}"')
    return int(rows[0]["n"])


def test_app_scope() -> None:
    assert {e.__name__ for e in app.entities} == {"Order", "OrderLine"}
    assert {r.__name__ for r in app.reactions} == {
        "OrderTotal",
        "CreateOrder",
        "UpdateLine",
        "DeleteLine",
        "ExpireUnpaidOrders",
        "CreateOrderApi",
        "GetOrderApi",
        "UpdateLineApi",
        "DeleteLineApi",
    }
    assert [f"{e.method} {e.path}" for e in app.endpoints] == [
        "POST /orders",
        "GET /orders/{order_id}",
        "PATCH /orders/{order_id}/lines/{line_id}",
        "DELETE /orders/{order_id}/lines/{line_id}",
    ]
    assert [t.split(" -> ")[0] for t in app.describe()["triggers"]] == [
        "OnCreate(OrderLine)",
        "OnChange(OrderLine, ['price', 'qty'])",
        "OnDelete(OrderLine)",
        "OnSchedule('* * * * *')",
    ]


def test_migrations_match_entities(migrated_database_url: str) -> None:
    migrations.check(app, migrated_database_url)


async def test_create_then_total_arrives_through_the_worker(
    client: AsyncClient, store: PostgresStore, worker: Worker
) -> None:
    body: dict[str, Any] = {
        "user_id": "u7",
        "lines": [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}],
    }
    created = await client.post("/orders", json=body, headers={"X-Tenant-Id": "acme"})
    assert created.status_code == 200, created.text
    order = created.json()["order"]
    assert (order["tenant"], order["total"]) == ("acme", 0)  # total is computed after commit
    assert await count(store, "order_line") == 2
    assert await count(store, "_outbox") == 2  # one row per created line

    assert await worker.run_once() == 2
    assert await count(store, "_outbox") == 0

    fetched = await client.get(f"/orders/{order['id']}", params={"include_lines": "true"})
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["order"]["total"] == 25
    assert len(fetched.json()["lines"]) == 2

    stored = await store.get_by_id(Order, order["id"])
    assert stored is not None and stored.total == 25
    lines = await store.query(OrderLine, "select * from order_line")
    assert {line.order_id for line in lines} == {order["id"]}


async def test_missing_order_is_500_and_store_untouched(
    client: AsyncClient, store: PostgresStore
) -> None:
    res = await client.get("/orders/00000000-0000-0000-0000-000000000000")
    assert res.status_code == 500 and res.json()["error"] == "LookupError"
    assert await count(store, "order") == 0


async def test_health_reports_postgres(client: AsyncClient) -> None:
    res = await client.get("/health")
    assert res.json() == {"status": "ok", "store": "PostgresStore"}
