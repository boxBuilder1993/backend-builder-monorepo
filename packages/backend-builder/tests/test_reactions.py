from datetime import UTC, datetime
from typing import Any

import pytest
from backend_builder import Context, DuplicateKeyError, Entity, InMemoryStore, Reaction
from backend_builder.reaction import qualified_name, resolve


class Order(Entity):
    user_id: str
    total: int = 0


class OrderLine(Entity):
    order_id: str
    price: int
    qty: int


class OrderTotal(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order_id: str = ctx.input
        lines = ctx.get(OrderLine, lambda line: line.order_id == order_id)
        order = ctx.get_by_id(Order, order_id)
        assert order is not None
        order.total = sum(line.price * line.qty for line in lines)


class CreateOrder(Reaction):
    async def calculate(self, ctx: Context) -> None:
        request: dict[str, Any] = ctx.input
        order = ctx.create(Order, user_id=request["user_id"])
        for line in request["lines"]:
            ctx.create(OrderLine, order_id=order.id, price=line["price"], qty=line["qty"])
        await ctx.trigger(OrderTotal, order.id)


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


async def test_create_order_computes_total(store: InMemoryStore) -> None:
    request = {"user_id": "u7", "lines": [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}]}

    await Context(store, request).trigger(CreateOrder, request)

    (order,) = store.get(Order)
    assert (order.user_id, order.total) == ("u7", 25)
    assert [line.order_id for line in store.get(OrderLine)] == [order.id] * 2


def test_rows_are_returned_by_reference(store: InMemoryStore) -> None:
    created = store.create(Order, user_id="u1")
    created.total = 99
    assert store.get(Order)[0] is created
    assert store.get(Order)[0].total == 99


def test_ids_are_unique_uuid_strings(store: InMemoryStore) -> None:
    ids = {store.create(Order, user_id="u").id for _ in range(3)}
    ids.add(store.create(OrderLine, order_id="o", price=1, qty=1).id)
    assert len(ids) == 4
    assert all(len(i) == 36 for i in ids)


def test_explicit_id_is_honoured_and_must_be_unique(store: InMemoryStore) -> None:
    row = store.create(Order, id="order-1", user_id="u1")
    assert row.id == "order-1"
    with pytest.raises(DuplicateKeyError):
        store.create(Order, id="order-1", user_id="u2")
    assert store.create(OrderLine, id="order-1", order_id="o", price=1, qty=1).id == "order-1"


def test_get_by_id(store: InMemoryStore) -> None:
    row = store.create(Order, user_id="u1")
    assert store.get_by_id(Order, row.id) is row
    assert store.get_by_id(Order, "missing") is None
    assert store.get_by_id(OrderLine, row.id) is None


def test_system_fields_and_soft_delete(store: InMemoryStore) -> None:
    row = store.create(Order, user_id="u1")
    assert row.created_at.tzinfo is UTC
    assert row.deleted_at is None
    row.deleted_at = datetime.now(UTC)
    assert store.get(Order) == []
    assert store.get_by_id(Order, row.id) is None


def test_registry_resolves_qualified_name() -> None:
    assert resolve(qualified_name(OrderTotal)) is OrderTotal
    with pytest.raises(LookupError):
        resolve("nowhere.Missing")


def test_entities_are_keyword_only_dataclasses() -> None:
    order = Order(user_id="u1")
    assert order.total == 0
    assert Order(user_id="u1", total=5).total == 5
