from datetime import UTC, datetime

import pytest
from backend_builder import (
    Context,
    DuplicateKeyError,
    Entity,
    InMemoryStore,
    MissingRowError,
    Reaction,
    Session,
    run,
)


class Account(Entity):
    balance: int = 0
    tags: list[str] | None = None


class Deposit(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = ctx.get_by_id(Account, ctx.input["account_id"])
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


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def test_session_without_backing_is_an_in_memory_store() -> None:
    session = Session()
    row = session.create(Account, balance=5)
    assert session.get_by_id(Account, row.id) is row
    session.flush()


def test_loads_copy_from_backing_and_write_back_on_flush(store: InMemoryStore) -> None:
    stored = store.create(Account, balance=10, tags=["a"])
    session = Session(store)

    loaded = session.get_by_id(Account, stored.id)
    assert loaded is not None and loaded is not stored
    loaded.balance = 20
    assert loaded.tags is not None
    loaded.tags.append("b")
    assert (stored.balance, stored.tags) == (10, ["a"])

    session.flush()
    after = store.get_by_id(Account, stored.id)
    assert after is not None and (after.balance, after.tags) == (20, ["a", "b"])


def test_get_loads_every_row_once_and_sees_session_creates(store: InMemoryStore) -> None:
    store.create(Account, balance=1)
    session = Session(store)
    session.create(Account, balance=2)
    assert sorted(a.balance for a in session.get(Account)) == [1, 2]
    store.create(Account, balance=3)
    assert sorted(a.balance for a in session.get(Account)) == [1, 2]


def test_soft_deleted_backing_rows_are_loaded_but_hidden(store: InMemoryStore) -> None:
    gone = store.create(Account, balance=1)
    gone.deleted_at = datetime.now(UTC)
    session = Session(store)
    assert session.get_by_id(Account, gone.id) is None
    assert session.get_by_id(Account, gone.id, include_deleted=True) is not None


def test_create_rejects_id_present_in_backing(store: InMemoryStore) -> None:
    store.create(Account, id="acc-1")
    with pytest.raises(DuplicateKeyError):
        Session(store).create(Account, id="acc-1")


def test_update_missing_row_raises(store: InMemoryStore) -> None:
    with pytest.raises(MissingRowError):
        store.update(Account, Account(balance=1))


async def test_run_flushes_on_success(store: InMemoryStore) -> None:
    account = store.create(Account, balance=10)
    await run(store, Deposit, {"account_id": account.id, "amount": 5})
    assert account.balance == 10
    after = store.get_by_id(Account, account.id)
    assert after is not None and after.balance == 15


async def test_run_discards_everything_on_error(store: InMemoryStore) -> None:
    account = store.create(Account, balance=10)
    with pytest.raises(RuntimeError):
        await run(store, DepositThenFail, {"account_id": account.id, "amount": 5})
    assert [a.balance for a in store.get(Account)] == [10]


async def test_saved_values_are_shared_across_the_run_and_visible_after(
    store: InMemoryStore,
) -> None:
    ctx = await run(store, OpenAccount, {})
    assert ctx.load("opened") is True
    assert ctx.load("account").balance == 7
    with pytest.raises(KeyError):
        ctx.load("missing")


def test_saved_values_are_not_flushed(store: InMemoryStore) -> None:
    session = Session(store)
    session.vars["note"] = object()
    session.flush()
    assert store.get(Account) == []
