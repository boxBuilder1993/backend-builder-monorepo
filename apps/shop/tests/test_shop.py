from typing import Any

import pytest
from backend_builder import InMemoryStore
from fastapi.testclient import TestClient
from shop import app
from shop.entities import Order, OrderLine


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def client(store: InMemoryStore) -> TestClient:
    return TestClient(app.http(store), raise_server_exceptions=False)


def test_app_scope() -> None:
    assert {e.__name__ for e in app.entities} == {"Order", "OrderLine"}
    assert {r.__name__ for r in app.reactions} == {
        "OrderTotal",
        "CreateOrder",
        "CreateOrderApi",
        "GetOrderApi",
    }
    assert [e.path for e in app.endpoints] == ["/orders", "/orders/{order_id}"]


def test_create_then_get(client: TestClient, store: InMemoryStore) -> None:
    body: dict[str, Any] = {
        "user_id": "u7",
        "lines": [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}],
    }
    created = client.post("/orders", json=body, headers={"X-Tenant-Id": "acme"})
    assert created.status_code == 200, created.text
    order = created.json()["order"]
    assert (order["tenant"], order["total"]) == ("acme", 25)
    assert len(store.get(OrderLine)) == 2

    fetched = client.get(f"/orders/{order['id']}", params={"include_lines": "true"})
    assert fetched.status_code == 200
    assert fetched.json()["order"] == order
    assert len(fetched.json()["lines"]) == 2


def test_missing_order_is_500_and_store_untouched(client: TestClient, store: InMemoryStore) -> None:
    res = client.get("/orders/nope")
    assert res.status_code == 500 and res.json()["error"] == "LookupError"
    assert store.get(Order) == []
