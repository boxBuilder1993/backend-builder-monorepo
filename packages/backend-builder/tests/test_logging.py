import io
import json
import logging

import pytest
from backend_builder import Context, Entity, InMemoryStore, Reaction, Session, log, run


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


# --- run fields on every record, both formats ----------------------------------------------------


class Child(Reaction):
    async def calculate(self, ctx: Context) -> None:
        logging.getLogger("myapp.child").info("child says hi")


class Parent(Reaction):
    async def calculate(self, ctx: Context) -> None:
        logging.getLogger("myapp.parent").info("parent starts")
        await ctx.trigger(Child, {})
        logging.getLogger("myapp.parent").warning("parent done")


def _field(record: logging.LogRecord, name: str) -> object:
    return getattr(record, name)


async def test_every_record_carries_the_runs_fields(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    ctx = await run(InMemoryStore(), Parent, {}, source="test")
    app_lines = [r for r in caplog.records if r.name.startswith("myapp.")]
    assert [r.getMessage() for r in app_lines] == ["parent starts", "child says hi", "parent done"]
    starts, child, done = app_lines
    assert _field(starts, "root_id") == ctx.reaction_id
    assert _field(starts, "reaction_id") == ctx.reaction_id
    assert _field(starts, "source") == "test"
    assert str(_field(starts, "reaction")).endswith(".Parent")
    assert _field(child, "root_id") == ctx.reaction_id
    assert _field(child, "reaction_id") != ctx.reaction_id
    assert str(_field(child, "reaction")).endswith(".Child")
    assert _field(done, "reaction_id") == ctx.reaction_id  # restored after the child
    runtime = [r for r in caplog.records if r.getMessage().startswith("reaction ")]
    assert runtime and all(_field(r, "root_id") == ctx.reaction_id for r in runtime)
    assert log.current() is None  # nothing leaks past the run
    outside = logging.getLogger("x").makeRecord("x", 20, "f", 1, "m", (), None)
    assert _field(outside, "root_id") is None


async def test_json_format_is_one_object_per_line_with_identity_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SECRET_TOKEN_X", "topsecret-value-1")
    stream = io.StringIO()
    log.identify(app="myapp", process="worker")
    log.configure(logging.INFO, stream=stream, format="json")
    try:
        ctx = await run(InMemoryStore(), Parent, {}, source="test")
        Context(Session(), {}).secret("token_x")
        logging.getLogger("myapp.leak").error("key is topsecret-value-1")
    finally:
        log.configure(logging.WARNING)  # back to a quiet terminal for the rest of the suite
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    child = next(line for line in lines if line["message"] == "child says hi")
    assert child["app"] == "myapp" and child["process"] == "worker"
    assert child["root_id"] == ctx.reaction_id and child["reaction"].endswith(".Child")
    assert child["level"] == "INFO" and child["logger"] == "myapp.child" and "time" in child
    assert child["source"] == "test"
    leak = next(line for line in lines if line["logger"] == "myapp.leak")
    assert leak["message"] == "key is [secret:token_x]" and "root_id" not in leak


def test_human_format_appends_the_reaction_id_once() -> None:
    formatter = log.HumanFormatter()
    record = logging.getLogger("myapp").makeRecord("myapp", 20, "f", 1, "hello", (), None)
    assert "reaction_id" not in formatter.format(record)
    token = log.enter(log.RunFields(root_id="r", reaction_id="abc", reaction="X", source="s"))
    try:
        record = logging.getLogger("myapp").makeRecord("myapp", 20, "f", 1, "hello", (), None)
        assert formatter.format(record).endswith("hello reaction_id=abc")
        record = logging.getLogger("myapp").makeRecord(
            "myapp", 20, "f", 1, "done reaction_id=abc", (), None
        )
        assert formatter.format(record).count("reaction_id=abc") == 1
    finally:
        log.leave(token)
