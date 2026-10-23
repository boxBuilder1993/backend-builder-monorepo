import asyncio
from typing import Any

import pytest
from backend_builder import Context, Endpoint, Entity, InMemoryStore, Reaction, Reply, create_app
from backend_builder.http import REACTION_ID_HEADER
from backend_builder.reaction import qualified_name
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict


def rows[E: Entity](store: InMemoryStore, entity: type[E], **kw: Any) -> list[E]:
    return asyncio.run(store.get(entity, **kw))


def row[E: Entity](store: InMemoryStore, entity: type[E], row_id: str) -> E | None:
    return asyncio.run(store.get_by_id(entity, row_id))


# --- entities ---------------------------------------------------------------


class Order(Entity):
    user_id: str
    tenant: str = "default"
    total: int = 0


class OrderLine(Entity):
    order_id: str
    price: int
    qty: int


# --- request and response models -------------------------------------------


class LineIn(BaseModel):
    price: int
    qty: int


class CreateOrderBody(BaseModel):
    user_id: str
    lines: list[LineIn]


class TenantHeaders(BaseModel):
    x_tenant_id: str = "default"


class OrderPath(BaseModel):
    order_id: str


class GetOrderQuery(BaseModel):
    include_lines: bool = False


class LinePath(BaseModel):
    order_id: str
    position: int


class LineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    price: int
    qty: int


class OrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    user_id: str
    tenant: str
    total: int


class OrderResponse(BaseModel):
    order: OrderOut
    lines: list[LineOut]


class LineResponse(BaseModel):
    line: LineOut
    position: int


# --- domain reactions: never see HTTP shapes ----------------------------------


class OrderTotal(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = await ctx.get_by_id(Order, ctx.input["order_id"])
        assert order is not None
        lines = await ctx.get(OrderLine, lambda line: line.order_id == order.id)
        order.total = sum(line.price * line.qty for line in lines)


class CreateOrder(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = ctx.create(Order, user_id=ctx.input["user_id"], tenant=ctx.input["tenant"])
        for line in ctx.input["lines"]:
            ctx.create(OrderLine, order_id=order.id, price=line["price"], qty=line["qty"])
        await ctx.trigger(OrderTotal, {"order_id": order.id})
        ctx.save("order", order)


# --- API reactions: translate the structured input into domain inputs --------


class CreateOrderApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        body = ctx.input["body"]
        created = await ctx.trigger(
            CreateOrder,
            {
                "user_id": body["user_id"],
                "lines": body["lines"],
                "tenant": ctx.input["headers"]["x_tenant_id"],
            },
        )
        assert created is not None
        ctx.save("order", created.load("order"))
        ctx.save("include_lines", True)


class GetOrderApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order_id = ctx.input["path"]["order_id"]
        order = await ctx.get_by_id(Order, order_id)
        if order is None:
            raise LookupError(f"no order {order_id}")
        ctx.save("order", order)
        ctx.save("include_lines", ctx.input["query"]["include_lines"])


class GetLineApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        path = ctx.input["path"]
        lines = await ctx.get(OrderLine, lambda line: line.order_id == path["order_id"])
        ctx.save("line", lines[path["position"]])
        ctx.save("position", path["position"])


class Ping(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Order, user_id=ctx.input.get("user_id", "ping"))


# --- builders -----------------------------------------------------------------


async def order_response(ctx: Context) -> OrderResponse:
    order: Order = ctx.load("order")
    lines: list[OrderLine] = []
    if ctx.load("include_lines"):
        lines = await ctx.get(OrderLine, lambda line: line.order_id == order.id)
    return OrderResponse(
        order=OrderOut.model_validate(order), lines=[LineOut.model_validate(line) for line in lines]
    )


async def line_response(ctx: Context) -> LineResponse:
    return LineResponse(
        line=LineOut.model_validate(ctx.load("line")), position=ctx.load("position")
    )


endpoints = [
    Endpoint(
        method="POST",
        path="/orders",
        reaction=CreateOrderApi,
        body=CreateOrderBody,
        headers=TenantHeaders,
        response=OrderResponse,
        build=order_response,
    ),
    Endpoint(
        method="GET",
        path="/orders/{order_id}",
        reaction=GetOrderApi,
        path_model=OrderPath,
        query=GetOrderQuery,
        response=OrderResponse,
        build=order_response,
    ),
    Endpoint(
        method="GET",
        path="/orders/{order_id}/lines/{position}",
        reaction=GetLineApi,
        path_model=LinePath,
        response=LineResponse,
        build=line_response,
    ),
]


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def client(store: InMemoryStore) -> TestClient:
    return TestClient(create_app(store, endpoints), raise_server_exceptions=False)


def test_body_and_headers_reach_the_api_reaction_separately(
    client: TestClient, store: InMemoryStore
) -> None:
    body: dict[str, Any] = {
        "user_id": "u7",
        "lines": [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}],
    }
    res = client.post("/orders", json=body, headers={"X-Tenant-Id": "acme"})

    assert res.status_code == 200, res.text
    data = res.json()
    assert data["order"]["user_id"] == "u7"
    assert data["order"]["tenant"] == "acme"
    assert data["order"]["total"] == 25
    assert sorted(line["price"] for line in data["lines"]) == [5, 10]
    assert len(res.headers[REACTION_ID_HEADER]) == 36
    assert row(store, Order, data["order"]["id"]) is not None


def test_header_default_applies_when_absent(client: TestClient) -> None:
    res = client.post("/orders", json={"user_id": "u1", "lines": []})
    assert res.json()["order"]["tenant"] == "default"


def test_path_and_query_reach_the_api_reaction_separately(
    client: TestClient, store: InMemoryStore
) -> None:
    order = asyncio.run(store.create(Order, user_id="u1", total=3))
    asyncio.run(store.create(OrderLine, order_id=order.id, price=3, qty=1))

    without = client.get(f"/orders/{order.id}").json()
    with_lines = client.get(f"/orders/{order.id}", params={"include_lines": "true"}).json()

    assert without["order"]["id"] == order.id and without["lines"] == []
    assert len(with_lines["lines"]) == 1


def test_path_model_converts_types(client: TestClient, store: InMemoryStore) -> None:
    order = asyncio.run(store.create(Order, user_id="u1"))
    asyncio.run(store.create(OrderLine, order_id=order.id, price=1, qty=1))
    asyncio.run(store.create(OrderLine, order_id=order.id, price=2, qty=1))

    res = client.get(f"/orders/{order.id}/lines/1")
    assert res.status_code == 200, res.text
    assert res.json()["position"] == 1 and res.json()["line"]["price"] == 2

    res = client.get(f"/orders/{order.id}/lines/one")
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["path", "position"]


def test_validation_errors_name_their_source(client: TestClient) -> None:
    res = client.post("/orders", json={"user_id": "u7"})
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["body", "lines"]

    res = client.get("/orders/x", params={"include_lines": "maybe"})
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["query", "include_lines"]


def test_path_model_must_match_path_parameters() -> None:
    with pytest.raises(ValueError, match="do not match"):
        Endpoint(method="GET", path="/orders/{order_id}", reaction=Ping, path_model=LinePath)


def test_run_failure_is_500_with_reaction_id_and_store_untouched(
    client: TestClient, store: InMemoryStore
) -> None:
    res = client.get("/orders/missing")

    assert res.status_code == 500
    data = res.json()
    assert data["error"] == "LookupError" and "missing" in data["message"]
    assert data["reaction_id"] == res.headers[REACTION_ID_HEADER]
    assert rows(store, Order) == []


def test_generic_trigger_by_qualified_name(client: TestClient, store: InMemoryStore) -> None:
    res = client.post(f"/reactions/{qualified_name(Ping)}", json={"user_id": "via-generic"})
    assert res.status_code == 200 and res.json() == {}
    assert [o.user_id for o in rows(store, Order)] == ["via-generic"]

    missing = client.post("/reactions/nowhere.Missing", json={})
    assert missing.status_code == 404 and missing.json()["error"] == "LookupError"
    assert client.post(f"/reactions/{qualified_name(Ping)}", json=[1]).status_code == 400


def test_generic_can_be_disabled(store: InMemoryStore) -> None:
    client = TestClient(create_app(store, endpoints, generic=False))
    assert client.post(f"/reactions/{qualified_name(Ping)}", json={}).status_code == 404


def test_openapi_documents_every_source(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    assert {"CreateOrderBody", "OrderResponse", "OrderOut", "LineOut", "LineResponse"} <= set(
        components
    )

    post = schema["paths"]["/orders"]["post"]
    assert post["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "CreateOrderBody"
    )
    assert {(p["name"], p["in"]) for p in post["parameters"]} == {("x-tenant-id", "header")}

    get = schema["paths"]["/orders/{order_id}"]["get"]
    assert {(p["name"], p["in"]) for p in get["parameters"]} == {
        ("order_id", "path"),
        ("include_lines", "query"),
    }

    line = schema["paths"]["/orders/{order_id}/lines/{position}"]["get"]
    position = next(p for p in line["parameters"] if p["name"] == "position")
    assert position["schema"]["type"] == "integer"


class Gate(Reaction):
    """Records an outcome instead of raising; the builder decides the status."""

    async def calculate(self, ctx: Context) -> None:
        token = ctx.input["headers"]["x_token"]
        if token != "open-sesame":
            ctx.save("denied", {"error": "Unauthenticated", "message": "bad token"})
            return
        ctx.create(Order, user_id="gated")
        ctx.save("order", (await ctx.get(Order))[0])


class ErrorOut(BaseModel):
    error: str
    message: str


class TokenHeaders(BaseModel):
    x_token: str = ""


async def gate_response(ctx: Context) -> OrderOut | Reply:
    if "denied" in ctx.vars:
        return Reply(status=401, body=ErrorOut(**ctx.load("denied")))
    return OrderOut.model_validate(ctx.load("order"))


def test_builder_reply_sets_the_status_and_the_run_still_commits(store: InMemoryStore) -> None:
    endpoint = Endpoint(
        method="POST",
        path="/gated",
        reaction=Gate,
        headers=TokenHeaders,
        response=OrderOut,
        responses={401: ErrorOut},
        build=gate_response,
    )
    client = TestClient(create_app(store, [endpoint]), raise_server_exceptions=False)

    denied = client.post("/gated", headers={"X-Token": "nope"})
    assert denied.status_code == 401
    assert denied.json() == {"error": "Unauthenticated", "message": "bad token"}
    assert len(denied.headers[REACTION_ID_HEADER]) == 36
    assert rows(store, Order) == []

    allowed = client.post("/gated", headers={"X-Token": "open-sesame"})
    assert allowed.status_code == 200 and allowed.json()["user_id"] == "gated"
    assert len(rows(store, Order)) == 1

    schema = client.get("/openapi.json").json()
    post = schema["paths"]["/gated"]["post"]
    assert post["responses"]["401"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "ErrorOut"
    )
