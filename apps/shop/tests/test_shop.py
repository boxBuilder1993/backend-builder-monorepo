from collections.abc import AsyncIterator
from typing import Any

import pytest
from backend_builder import PostgresStore, migrations
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
        "CreateOrderApi",
        "GetOrderApi",
    }
    assert [e.path for e in app.endpoints] == ["/orders", "/orders/{order_id}"]


def test_migrations_match_entities(migrated_database_url: str) -> None:
    migrations.check(app, migrated_database_url)


async def test_create_then_get(client: AsyncClient, store: PostgresStore) -> None:
    body: dict[str, Any] = {
        "user_id": "u7",
        "lines": [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}],
    }
    created = await client.post("/orders", json=body, headers={"X-Tenant-Id": "acme"})
    assert created.status_code == 200, created.text
    order = created.json()["order"]
    assert (order["tenant"], order["total"]) == ("acme", 25)
    assert await count(store, "order_line") == 2

    fetched = await client.get(f"/orders/{order['id']}", params={"include_lines": "true"})
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["order"] == order
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
