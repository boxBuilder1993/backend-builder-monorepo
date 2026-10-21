from typing import Any

import pytest
from backend_builder import Context, Entity, InMemoryStore, Reaction, Session, run


class Account(Entity):
    balance: int = 0
    tags: list[str] | None = None
    meta: dict[str, Any] | None = None


class RecordingStore(InMemoryStore):
    """Counts the writes it receives so a test can assert what flush did."""

    def __init__(self) -> None:
        super().__init__()
        self.updates: list[tuple[str, dict[str, Any]]] = []
        self.inserts: int = 0

    async def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        self.inserts += 1
        return await super().create(entity, **fields)

    async def update(self, entity: type[Entity], row_id: str, changes: dict[str, Any]) -> None:
        self.updates.append((row_id, changes))
        await super().update(entity, row_id, changes)


class TouchNothing(Reaction):
    async def calculate(self, ctx: Context) -> None:
        for account_id in ctx.input["ids"]:
            assert await ctx.get_by_id(Account, account_id) is not None


class ChangeOne(Reaction):
    async def calculate(self, ctx: Context) -> None:
        a = await ctx.get_by_id(Account, ctx.input["a"])
        b = await ctx.get_by_id(Account, ctx.input["b"])
        assert a is not None and b is not None
        a.balance += 1
        b.balance = b.balance  # written but unchanged


class MutateNested(Reaction):
    async def calculate(self, ctx: Context) -> None:
        a = await ctx.get_by_id(Account, ctx.input["id"])
        assert a is not None and a.tags is not None and a.meta is not None
        a.tags.append("new")
        a.meta["k"] = "v"


class Close(Reaction):
    async def calculate(self, ctx: Context) -> None:
        a = await ctx.get_by_id(Account, ctx.input["id"])
        assert a is not None
        ctx.delete(a)


class RevertWithinRun(Reaction):
    async def calculate(self, ctx: Context) -> None:
        a = await ctx.get_by_id(Account, ctx.input["id"])
        assert a is not None
        a.balance = 50
        a.balance = ctx.input["original"]


@pytest.fixture
def store() -> RecordingStore:
    return RecordingStore()


async def test_unchanged_loaded_rows_produce_no_write(store: RecordingStore) -> None:
    ids = [(await store.create(Account, balance=i)).id for i in range(3)]
    await run(store, TouchNothing, {"ids": ids})
    assert store.updates == []


async def test_only_changed_fields_are_written(store: RecordingStore) -> None:
    a = await store.create(Account, balance=1, tags=["x"])
    b = await store.create(Account, balance=2)
    await run(store, ChangeOne, {"a": a.id, "b": b.id})
    assert store.updates == [(a.id, {"balance": 2})]
    assert (await store.get_by_id(Account, a.id)).tags == ["x"]  # type: ignore[union-attr]


async def test_nested_mutations_are_detected(store: RecordingStore) -> None:
    a = await store.create(Account, tags=["old"], meta={})
    await run(store, MutateNested, {"id": a.id})
    assert store.updates == [(a.id, {"tags": ["old", "new"], "meta": {"k": "v"}})]


async def test_soft_delete_writes_only_deleted_at(store: RecordingStore) -> None:
    a = await store.create(Account, balance=9)
    await run(store, Close, {"id": a.id})
    ((row_id, changes),) = store.updates
    assert row_id == a.id and set(changes) == {"deleted_at"} and changes["deleted_at"] is not None


async def test_value_changed_and_reverted_in_a_run_is_not_written(store: RecordingStore) -> None:
    a = await store.create(Account, balance=7)
    await run(store, RevertWithinRun, {"id": a.id, "original": 7})
    assert store.updates == []


async def test_created_rows_are_inserted_not_compared(store: RecordingStore) -> None:
    session = Session(store)
    created = session.create(Account, balance=3)
    with pytest.raises(KeyError):
        session.changes(created)
    await session.flush()
    assert store.inserts == 1 and store.updates == []


async def test_snapshots_refresh_after_flush(store: RecordingStore) -> None:
    a = await store.create(Account, balance=1)
    session = Session(store)
    loaded = await session.get_by_id(Account, a.id)
    assert loaded is not None
    loaded.balance = 2
    await session.flush()
    await session.flush()
    assert store.updates == [(a.id, {"balance": 2})]
    assert session.changes(loaded) == {}


async def test_written_changes_do_not_alias_session_objects(store: RecordingStore) -> None:
    a = await store.create(Account, tags=[])
    session = Session(store)
    loaded = await session.get_by_id(Account, a.id)
    assert loaded is not None and loaded.tags is not None
    loaded.tags.append("x")
    await session.flush()
    stored = await store.get_by_id(Account, a.id)
    assert stored is not None and stored.tags == ["x"] and stored.tags is not loaded.tags
