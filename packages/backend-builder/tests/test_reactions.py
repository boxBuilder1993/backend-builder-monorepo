from typing import Any

import pytest
from backend_builder import Context, Entity, InMemoryStore, Reaction
from backend_builder.reaction import qualified_name, resolve


class Order(Entity):
    pass


class OrderLine(Entity):
    pass


class OrderTotal(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order_id: str = ctx.input
        lines = ctx.get(OrderLine, lambda row: row["order_id"] == order_id)
        (order,) = ctx.get(Order, lambda row: row["id"] == order_id)
        order["total"] = sum(line["price"] * line["qty"] for line in lines)


class CreateOrder(Reaction):
    async def calculate(self, ctx: Context) -> None:
        request: dict[str, Any] = ctx.input
        order = ctx.create(Order, user_id=request["user_id"], total=0)
        for line in request["lines"]:
            ctx.create(OrderLine, order_id=order["id"], price=line["price"], qty=line["qty"])
        await ctx.trigger(OrderTotal, order["id"])


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


async def test_create_order_computes_total(store: InMemoryStore) -> None:
    request = {"user_id": 7, "lines": [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}]}

    await Context(store, request).trigger(CreateOrder, request)

    (order,) = store.get(Order)
    assert order == {"id": order["id"], "user_id": 7, "total": 25}
    assert [line["order_id"] for line in store.get(OrderLine)] == [order["id"]] * 2


async def test_rows_are_returned_by_reference(store: InMemoryStore) -> None:
    created = store.get(Order) or [store.create(Order, total=0)]
    created[0]["total"] = 99
    assert store.get(Order)[0]["total"] == 99


async def test_ids_are_unique_uuid_strings(store: InMemoryStore) -> None:
    ids = {store.create(Order)["id"] for _ in range(3)} | {store.create(OrderLine)["id"]}
    assert len(ids) == 4
    assert all(isinstance(i, str) and len(i) == 36 for i in ids)


def test_registry_resolves_qualified_name() -> None:
    assert resolve(qualified_name(OrderTotal)) is OrderTotal
    with pytest.raises(LookupError):
        resolve("nowhere.Missing")
