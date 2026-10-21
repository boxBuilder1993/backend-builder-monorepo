import logging

import pytest
from backend_builder import App, Context, Entity, InMemoryStore, Reaction, run


class Counter(Entity):
    value: int = 0


calls: list[int] = []


class FlakyThenOk(Reaction):
    """Fails on the first two attempts, succeeds on the third."""

    async def calculate(self, ctx: Context) -> None:
        calls.append(1)
        ctx.create(Counter, value=len(calls))
        if len(calls) < 3:
            raise RuntimeError(f"attempt {len(calls)} failed")


class AlwaysFails(Reaction):
    async def calculate(self, ctx: Context) -> None:
        calls.append(1)
        ctx.create(Counter)
        raise ValueError("never")


@pytest.fixture(autouse=True)
def reset() -> None:
    calls.clear()


async def test_any_exception_retries_until_success(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="backend_builder")
    store = InMemoryStore()
    ctx = await run(store, FlakyThenOk, {}, max_attempts=3)

    assert len(calls) == 3
    rows = await store.get(Counter)
    assert [r.value for r in rows] == [3]  # only the successful attempt's session was flushed
    retries = [r for r in caplog.records if r.getMessage().startswith("run retrying")]
    assert len(retries) == 2 and all(ctx.reaction_id in r.getMessage() for r in retries)


async def test_last_attempts_error_propagates_and_nothing_is_written() -> None:
    store = InMemoryStore()
    with pytest.raises(ValueError, match="never"):
        await run(store, AlwaysFails, {}, max_attempts=2)
    assert len(calls) == 2
    assert await store.get(Counter) == []


async def test_one_attempt_means_no_retry() -> None:
    with pytest.raises(RuntimeError, match="attempt 1"):
        await run(InMemoryStore(), FlakyThenOk, {}, max_attempts=1)
    assert len(calls) == 1


def test_app_carries_the_policy() -> None:
    assert App(package="fixture_app").max_attempts == 3
    assert App(package="fixture_app", max_attempts=5).max_attempts == 5
