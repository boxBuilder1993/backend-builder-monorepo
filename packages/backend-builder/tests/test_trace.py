import asyncio

import pytest
from backend_builder import Context, Entity, InMemoryStore, Reaction, Session, run


class Thing(Entity):
    value: int = 0


class Inner(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await asyncio.sleep(0.01)
        ctx.create(Thing, value=1)


class Outer(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(Inner, {})
        await ctx.trigger(Inner, {})


class Explode(Reaction):
    async def calculate(self, ctx: Context) -> None:
        raise ValueError("no")


async def test_trace_records_a_span_per_invocation_and_the_flush() -> None:
    ctx = await run(InMemoryStore(), Outer, {})

    names = [s.name for s in ctx.trace.spans]
    assert names == ["Inner", "Inner", "Outer", "flush", "run"]

    inner, inner2, outer, flush, whole = ctx.trace.spans
    assert inner.parent_reaction_id == outer.reaction_id == ctx.reaction_id
    assert inner2.parent_reaction_id == ctx.reaction_id
    assert outer.parent_reaction_id is None
    assert flush.reaction_id == ctx.reaction_id
    assert all(s.ok for s in ctx.trace.spans)

    assert inner.duration_ms >= 10 and inner2.duration_ms >= 10
    assert outer.duration_ms >= inner.duration_ms + inner2.duration_ms
    assert ctx.trace.total_ms() == whole.duration_ms
    assert whole.duration_ms >= outer.duration_ms + flush.duration_ms
    assert ctx.trace.by_reaction_id(inner.reaction_id) is inner
    assert ctx.trace.by_reaction_id(ctx.reaction_id) is outer


async def test_failed_invocation_is_recorded_as_not_ok() -> None:
    store = InMemoryStore()
    seed = Context(Session(store), {})
    with pytest.raises(ValueError):
        await seed.trigger(Explode, {})
    (span,) = seed.trace.spans
    assert span.name == "Explode" and span.ok is False
    with pytest.raises(LookupError):
        seed.trace.total_ms()
