import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from backend_builder import (
    Context,
    Entity,
    MissingRowError,
    NotSupported,
    PostgresStore,
    Reaction,
    TransientError,
    run,
)
from backend_builder.codec import EncodeError
from backend_builder.postgres import PostgresDuplicateKeyError
from sqlalchemy.ext.asyncio import create_async_engine


@dataclass
class Address:
    city: str
    moved_in: datetime


class Account(Entity):
    owner: str
    balance: int = 0
    rate: Decimal = Decimal("0.05")
    tags: list[str] = field(default_factory=list[str])
    meta: dict[str, Any] = field(default_factory=dict[str, Any])
    address: Address | None = None
    scratch: list[str] = field(default_factory=list[str], metadata={"persist": False})


class Deposit(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["id"])
        assert account is not None
        await asyncio.sleep(ctx.input.get("sleep", 0))
        account.balance += ctx.input["amount"]


class Open(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.save(
            "account", ctx.create(Account, owner=ctx.input["owner"], **ctx.input.get("extra", {}))
        )


class Close(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["id"])
        assert account is not None
        ctx.delete(account)


class RichOwners(Reaction):
    async def calculate(self, ctx: Context) -> None:
        rows = await ctx.query(
            Account,
            "select * from account where balance >= :floor order by balance",
            floor=ctx.input["floor"],
        )
        for row in rows:
            row.tags.append("rich")
        ctx.save("ids", [r.id for r in rows])
        ctx.save("total", await ctx.query_raw("select sum(balance) as total from account"))


class Explode(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Account, owner="ghost")
        raise RuntimeError("boom")


class UnstorableValue(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Account, owner="x", meta={"bad": object()})


async def _create_tables(store: PostgresStore) -> None:
    engine = create_async_engine(store.url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(store.metadata.create_all)
    finally:
        await engine.dispose()


@pytest.fixture
async def store(database_url: str) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(database_url, [Account])
    await _create_tables(store)
    yield store
    await store.close()


async def test_create_get_update_round_trip(store: PostgresStore) -> None:
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    ctx = await run(
        store,
        Open,
        {
            "owner": "ada",
            "extra": {
                "rate": Decimal("1.25"),
                "tags": ["a"],
                "meta": {"n": 1, "when": when},
                "address": Address(city="Paris", moved_in=when),
            },
        },
    )
    created: Account = ctx.load("account")

    loaded = await store.get_by_id(Account, created.id)
    assert loaded is not None and loaded is not created
    assert loaded.owner == "ada" and loaded.rate == Decimal("1.25")
    assert loaded.tags == ["a"] and loaded.meta == {"n": 1, "when": when.isoformat()}
    assert loaded.address == Address(city="Paris", moved_in=when)
    assert loaded.created_at.tzinfo is not None and loaded.scratch == []

    await run(store, Deposit, {"id": created.id, "amount": 7})
    after = await store.get_by_id(Account, created.id)
    assert after is not None and after.balance == 7 and after.tags == ["a"]


async def test_soft_delete_hides_the_row(store: PostgresStore) -> None:
    ctx = await run(store, Open, {"owner": "bob"})
    account_id = ctx.load("account").id
    await run(store, Close, {"id": account_id})
    assert await store.get_by_id(Account, account_id) is None
    gone = await store.get_by_id(Account, account_id, include_deleted=True)
    assert gone is not None and gone.deleted_at is not None


async def test_query_merges_and_mutations_flush(store: PostgresStore) -> None:
    ids = [
        (await run(store, Open, {"owner": o, "extra": {"balance": b}})).load("account").id
        for o, b in (("a", 1), ("b", 50), ("c", 100))
    ]
    ctx = await run(store, RichOwners, {"floor": 50})
    assert ctx.load("ids") == ids[1:]
    assert ctx.load("total") == [{"total": 151}]
    for account_id, expected in zip(ids, ([], ["rich"], ["rich"]), strict=True):
        row = await store.get_by_id(Account, account_id)
        assert row is not None and row.tags == expected


async def test_get_with_predicate_is_not_supported(store: PostgresStore) -> None:
    with pytest.raises(NotSupported):
        await store.get(Account)


async def test_failed_run_writes_nothing(store: PostgresStore) -> None:
    with pytest.raises(RuntimeError):
        await run(store, Explode, {})
    assert await store.query_raw("select count(*) as n from account") == [{"n": 0}]


async def test_unstorable_value_fails_at_flush_naming_the_field(store: PostgresStore) -> None:
    with pytest.raises(EncodeError, match="Account.meta.bad"):
        await run(store, UnstorableValue, {})


async def test_duplicate_id_is_transient_and_eventually_raises(store: PostgresStore) -> None:
    fixed = "11111111-2222-3333-4444-555555555555"
    await store.create(Account, id=fixed, owner="first")
    with pytest.raises(PostgresDuplicateKeyError):
        await store.create(Account, id=fixed, owner="second")
    assert issubclass(PostgresDuplicateKeyError, TransientError)


async def test_update_missing_row_raises(store: PostgresStore) -> None:
    with pytest.raises(MissingRowError):
        await store.update(Account, "00000000-0000-0000-0000-000000000000", {"balance": 1})


async def test_concurrent_serializable_runs_retry_and_converge(store: PostgresStore) -> None:
    ctx = await run(store, Open, {"owner": "shared"})
    account_id = ctx.load("account").id
    inputs = [{"id": account_id, "amount": 1, "sleep": 0.02} for _ in range(4)]

    results = await asyncio.gather(
        *(run(store, Deposit, i) for i in inputs), return_exceptions=True
    )

    failures = [r for r in results if isinstance(r, BaseException)]
    after = await store.get_by_id(Account, account_id)
    assert after is not None
    assert after.balance == 4 - len(failures)
    assert all(isinstance(f, TransientError) for f in failures)
    assert len(failures) <= 1  # four conflicting writers, three attempts each: at most one gives up


async def test_ping(store: PostgresStore) -> None:
    assert await store.ping() is True
    broken = PostgresStore("postgresql://nobody:nothing@localhost:1/none", [Account])
    assert await broken.ping() is False
    await broken.close()
