from typing import Any

import pytest
from backend_builder import Context, Endpoint, Entity, InMemoryStore, Reaction, create_app
from backend_builder.http import REACTION_ID_HEADER
from backend_builder.reaction import qualified_name
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict


class Order(Entity):
    user_id: str
    total: int = 0


class OrderLine(Entity):
    order_id: str
    price: int
    qty: int


class LineIn(BaseModel):
    price: int
    qty: int


class CreateOrderRequest(BaseModel):
    user_id: str
    lines: list[LineIn]


class GetOrderRequest(BaseModel):
    include_lines: bool = False


class LineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    price: int
    qty: int


class OrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    user_id: str
    total: int


class OrderResponse(BaseModel):
    order: OrderOut
    lines: list[LineOut]


class OrderTotal(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = ctx.get_by_id(Order, ctx.input["order_id"])
        assert order is not None
        lines = ctx.get(OrderLine, lambda line: line.order_id == order.id)
        order.total = sum(line.price * line.qty for line in lines)


class CreateOrder(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = ctx.create(Order, user_id=ctx.input["user_id"])
        for line in ctx.input["lines"]:
            ctx.create(OrderLine, order_id=order.id, price=line["price"], qty=line["qty"])
        await ctx.trigger(OrderTotal, {"order_id": order.id})
        ctx.save("order", order)
        ctx.save("include_lines", True)


class GetOrder(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = ctx.get_by_id(Order, ctx.input["order_id"])
        if order is None:
            raise LookupError(f"no order {ctx.input['order_id']}")
        ctx.save("order", order)
        ctx.save("include_lines", ctx.input["include_lines"])


class Ping(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Order, user_id=ctx.input.get("user_id", "ping"))


def order_response(ctx: Context) -> OrderResponse:
    order: Order = ctx.load("order")
    lines: list[OrderLine] = []
    if ctx.load("include_lines"):
        lines = ctx.get(OrderLine, lambda line: line.order_id == order.id)
    return OrderResponse(
        order=OrderOut.model_validate(order), lines=[LineOut.model_validate(line) for line in lines]
    )


endpoints = [
    Endpoint(
        method="POST",
        path="/orders",
        reaction=CreateOrder,
        request=CreateOrderRequest,
        response=OrderResponse,
        build=order_response,
    ),
    Endpoint(
        method="GET",
        path="/orders/{order_id}",
        reaction=GetOrder,
        request=GetOrderRequest,
        response=OrderResponse,
        build=order_response,
    ),
]


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def client(store: InMemoryStore) -> TestClient:
    return TestClient(create_app(store, endpoints), raise_server_exceptions=False)


def test_post_creates_and_builds_response(client: TestClient, store: InMemoryStore) -> None:
    body: dict[str, Any] = {
        "user_id": "u7",
        "lines": [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}],
    }
    res = client.post("/orders", json=body)

    assert res.status_code == 200, res.text
    data = res.json()
    assert data["order"]["user_id"] == "u7" and data["order"]["total"] == 25
    assert sorted(line["price"] for line in data["lines"]) == [5, 10]
    assert len(res.headers[REACTION_ID_HEADER]) == 36
    assert store.get_by_id(Order, data["order"]["id"]) is not None


def test_get_merges_path_and_query_into_input(client: TestClient, store: InMemoryStore) -> None:
    order = store.create(Order, user_id="u1", total=3)
    store.create(OrderLine, order_id=order.id, price=3, qty=1)

    without = client.get(f"/orders/{order.id}").json()
    with_lines = client.get(f"/orders/{order.id}", params={"include_lines": "true"}).json()

    assert without["order"]["id"] == order.id and without["lines"] == []
    assert len(with_lines["lines"]) == 1


def test_validation_errors_are_422(client: TestClient) -> None:
    res = client.post("/orders", json={"user_id": "u7"})
    assert res.status_code == 422


def test_run_failure_is_500_with_reaction_id_and_store_untouched(
    client: TestClient, store: InMemoryStore
) -> None:
    res = client.get("/orders/missing")

    assert res.status_code == 500
    data = res.json()
    assert data["error"] == "LookupError" and "missing" in data["message"]
    assert data["reaction_id"] == res.headers[REACTION_ID_HEADER]
    assert store.get(Order) == []


def test_generic_trigger_by_qualified_name(client: TestClient, store: InMemoryStore) -> None:
    res = client.post(f"/reactions/{qualified_name(Ping)}", json={"user_id": "via-generic"})
    assert res.status_code == 200 and res.json() == {}
    assert [o.user_id for o in store.get(Order)] == ["via-generic"]

    assert client.post("/reactions/nowhere.Missing", json={}).status_code == 500
    assert client.post(f"/reactions/{qualified_name(Ping)}", json=[1]).status_code == 400


def test_generic_can_be_disabled(store: InMemoryStore) -> None:
    client = TestClient(create_app(store, endpoints, generic=False))
    assert client.post(f"/reactions/{qualified_name(Ping)}", json={}).status_code == 404


def test_openapi_documents_request_and_response_models(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/orders" in schema["paths"] and "/orders/{order_id}" in schema["paths"]
    components = schema["components"]["schemas"]
    assert {"CreateOrderRequest", "OrderResponse", "OrderOut", "LineOut"} <= set(components)
    get_params = {p["name"] for p in schema["paths"]["/orders/{order_id}"]["get"]["parameters"]}
    assert get_params == {"order_id", "include_lines"}
