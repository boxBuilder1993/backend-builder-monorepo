import asyncio
import logging

import pytest
from backend_builder import Context, Entity, InMemoryStore, Reaction, create_app, run
from fastapi.testclient import TestClient


class Account(Entity):
    balance: int = 0


class SlowDeposit(Reaction):
    """Loads, waits, writes: the shape that loses updates without a run lock."""

    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["account_id"])
        assert account is not None
        await asyncio.sleep(0.01)
        account.balance += ctx.input["amount"]


class CloseAccount(Reaction):
    async def calculate(self, ctx: Context) -> None:
        account = await ctx.get_by_id(Account, ctx.input["account_id"])
        assert account is not None
        ctx.delete(account)
        assert await ctx.get_by_id(Account, account.id) is None
        assert await ctx.get_by_id(Account, account.id, include_deleted=True) is account


class Explode(Reaction):
    async def calculate(self, ctx: Context) -> None:
        raise ValueError("no")


async def test_concurrent_runs_on_one_store_do_not_lose_updates() -> None:
    store = InMemoryStore()
    account = await store.create(Account, balance=0)
    inputs = [{"account_id": account.id, "amount": 1} for _ in range(10)]

    await asyncio.gather(*(run(store, SlowDeposit, i) for i in inputs))

    after = await store.get_by_id(Account, account.id)
    assert after is not None and after.balance == 10


async def test_delete_is_soft_and_flushes() -> None:
    store = InMemoryStore()
    account = await store.create(Account, balance=5)

    await run(store, CloseAccount, {"account_id": account.id})

    assert await store.get(Account) == []
    gone = await store.get_by_id(Account, account.id, include_deleted=True)
    assert gone is not None and gone.deleted_at is not None


def test_health_endpoint() -> None:
    client = TestClient(create_app(InMemoryStore()))
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "store": "InMemoryStore"}
    assert "/health" not in client.get("/openapi.json").json()["paths"]


async def test_run_logs_a_summary_and_debug_span_tree(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="backend_builder")
    store = InMemoryStore()
    account = await store.create(Account)
    ctx = await run(store, SlowDeposit, {"account_id": account.id, "amount": 1})

    messages = [r.getMessage() for r in caplog.records]
    summary = next(m for m in messages if m.startswith("run finished"))
    assert f"reaction_id={ctx.reaction_id}" in summary
    assert "spans=3" in summary and "failed=none" in summary
    spans = [m for m in messages if m.startswith("span ")]
    assert [m.split()[1] for m in spans] == ["run", "SlowDeposit", "flush"]


async def test_failed_run_still_logs_its_trace(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="backend_builder")
    with pytest.raises(ValueError):
        await run(InMemoryStore(), Explode, {})
    summary = next(
        m for m in (r.getMessage() for r in caplog.records) if m.startswith("run finished")
    )
    assert "failed=Explode,run" in summary
