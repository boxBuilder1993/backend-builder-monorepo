from datetime import UTC, datetime
from typing import Any

import pytest
from backend_builder import (
    Context,
    DuplicateKeyError,
    Entity,
    InMemoryStore,
    MissingRowError,
    NotSupported,
    Reaction,
    Session,
    run,
)


class Account(Entity):
    balance: int = 0
    tags: list[str] | None = None


class Deposit(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["account_id"])
        assert account is not None
        account.balance += ctx.input["amount"]


class CreateAccount(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = ctx.create(Account, balance=ctx.input["balance"])
        ctx.save("account", account)


class OpenAccount(Reaction):
    async def calculate(self, ctx: Context) -> None:
        child = await ctx.trigger(CreateAccount, {"balance": 7})
        assert child.load("account") is ctx.load("account")
        ctx.save("opened", True)


class DepositThenFail(Deposit):
    async def calculate(self, ctx: Context) -> None:
        await super().calculate(ctx)
        ctx.create(Account, balance=1)
        raise RuntimeError("boom")


class CreateWithId(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Account, id=ctx.input["id"])


class StubSqlStore(InMemoryStore):
    """An in-memory store that pretends to be a SQL store: ``query`` treats
    the params as equality filters, so the session's merge can be tested."""

    async def query[E: Entity](self, entity: type[E], statement: str, **params: Any) -> list[E]:
        return [
            Account(
                **{f: getattr(r, f) for f in ("id", "created_at", "deleted_at", "balance", "tags")}
            )  # type: ignore[return-value]
            for r in self._table(entity).values()
            if all(getattr(r, k) == v for k, v in params.items())
        ]

    async def query_raw(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        return [{"statement": statement, **params}]


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


async def test_session_without_backing_is_an_in_memory_store() -> None:
    session = Session()
    row = session.create(Account, balance=5)
    assert await session.get_by_id(Account, row.id) is row
    await session.flush()


async def test_loads_copy_from_backing_and_write_back_on_flush(store: InMemoryStore) -> None:
    stored = await store.create(Account, balance=10, tags=["a"])
    session = Session(store)

    loaded = await session.get_by_id(Account, stored.id)
    assert loaded is not None and loaded is not stored
    loaded.balance = 20
    assert loaded.tags is not None
    loaded.tags.append("b")
    assert (stored.balance, stored.tags) == (10, ["a"])

    await session.flush()
    after = await store.get_by_id(Account, stored.id)
    assert after is not None and (after.balance, after.tags) == (20, ["a", "b"])


async def test_get_loads_every_row_once_and_sees_session_creates(store: InMemoryStore) -> None:
    await store.create(Account, balance=1)
    session = Session(store)
    session.create(Account, balance=2)
    assert sorted(a.balance for a in await session.get(Account)) == [1, 2]
    await store.create(Account, balance=3)
    assert sorted(a.balance for a in await session.get(Account)) == [1, 2]


async def test_soft_deleted_backing_rows_are_loaded_but_hidden(store: InMemoryStore) -> None:
    gone = await store.create(Account, balance=1)
    gone.deleted_at = datetime.now(UTC)
    session = Session(store)
    assert await session.get_by_id(Account, gone.id) is None
    assert await session.get_by_id(Account, gone.id, include_deleted=True) is not None


async def test_duplicate_id_against_backing_is_rejected_at_flush(store: InMemoryStore) -> None:
    await store.create(Account, id="acc-1")
    with pytest.raises(DuplicateKeyError):
        await run(store, CreateWithId, {"id": "acc-1"})
    assert len(await store.get(Account)) == 1


async def test_update_missing_row_raises(store: InMemoryStore) -> None:
    with pytest.raises(MissingRowError):
        await store.update(Account, "missing", {"balance": 1})


async def test_run_flushes_on_success(store: InMemoryStore) -> None:
    account = await store.create(Account, balance=10)
    await run(store, Deposit, {"account_id": account.id, "amount": 5})
    after = await store.get_by_id(Account, account.id)
    assert after is account and account.balance == 15


async def test_run_discards_everything_on_error(store: InMemoryStore) -> None:
    account = await store.create(Account, balance=10)
    with pytest.raises(RuntimeError):
        await run(store, DepositThenFail, {"account_id": account.id, "amount": 5})
    assert [a.balance for a in await store.get(Account)] == [10]


async def test_saved_values_are_shared_across_the_run_and_visible_after(
    store: InMemoryStore,
) -> None:
    ctx = await run(store, OpenAccount, {})
    assert ctx.load("opened") is True
    assert ctx.load("account").balance == 7
    with pytest.raises(KeyError):
        ctx.load("missing")


async def test_saved_values_are_not_flushed(store: InMemoryStore) -> None:
    session = Session(store)
    session.vars["note"] = object()
    await session.flush()
    assert await store.get(Account) == []


async def test_query_is_not_supported_on_the_in_memory_store(store: InMemoryStore) -> None:
    session = Session(store)
    with pytest.raises(NotSupported):
        await session.query(Account, "select * from account")
    with pytest.raises(NotSupported):
        await session.query_raw("select 1")
    with pytest.raises(NotSupported):
        await Session().query(Account, "select * from account")


async def test_query_merges_into_the_working_set() -> None:
    store = StubSqlStore()
    a = await store.create(Account, balance=1)
    b = await store.create(Account, balance=1)
    await store.create(Account, balance=2)
    session = Session(store)

    first = await session.get_by_id(Account, a.id)
    assert first is not None
    first.balance = 99

    rows = await session.query(Account, "irrelevant", balance=1)
    assert {r.id for r in rows} == {a.id, b.id}
    assert next(r for r in rows if r.id == a.id) is first
    assert first.balance == 99

    again = await session.query(Account, "irrelevant", balance=1)
    assert [id(r) for r in again] == [id(r) for r in rows]

    await session.flush()
    assert (await store.get_by_id(Account, a.id)).balance == 99  # type: ignore[union-attr]
    assert await session.query_raw("select 1", x=2) == [{"statement": "select 1", "x": 2}]
