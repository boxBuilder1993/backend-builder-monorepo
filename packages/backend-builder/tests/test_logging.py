import logging

import pytest
from backend_builder import Context, Entity, InMemoryStore, Reaction, run


class Thing(Entity):
    value: int = 0


class MakeThing(Reaction):
    async def calculate(self, ctx: Context) -> None:
        ctx.create(Thing, value=ctx.input["value"])


class Explode(Reaction):
    async def calculate(self, ctx: Context) -> None:
        raise ValueError("secret-payload")


async def test_info_logs_identifiers_but_never_the_input(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="backend_builder")
    ctx = await run(InMemoryStore(), MakeThing, {"value": 42})

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("reaction started reaction=MakeThing") for m in messages)
    assert any(m.startswith("reaction finished reaction=MakeThing") for m in messages)
    assert any(m == f"flush started reaction_id={ctx.reaction_id}" for m in messages)
    assert any(
        m.startswith(f"flush finished reaction_id={ctx.reaction_id} inserted=1") for m in messages
    )
    assert ctx.reaction_id in " ".join(messages)
    assert not any(m.startswith("reaction input") or "value" in m for m in messages)
    assert all(r.levelno == logging.INFO for r in caplog.records)


async def test_debug_logs_include_input_and_rows(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="backend_builder")
    await run(InMemoryStore(), MakeThing, {"value": 42})

    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(m.startswith("reaction input") and "'value': 42" in m for m in debug)
    assert any(m.startswith("flush insert") and "value=42" in m for m in debug)


async def test_failure_is_logged_at_error_and_reraised(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="backend_builder")
    with pytest.raises(ValueError):
        await run(InMemoryStore(), Explode, {}, max_attempts=1)

    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert [e.split(" reaction_id=")[0] for e in errors] == [
        "reaction failed reaction=Explode",
        "run gave up reaction=Explode",
    ]
    assert not any(m.startswith("flush") for m in (r.getMessage() for r in caplog.records))
