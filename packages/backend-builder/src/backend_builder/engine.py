"""Run entry point: the seed of a run."""

from __future__ import annotations

from typing import Any

from backend_builder.context import Context
from backend_builder.reaction import Reaction, invoke
from backend_builder.store import Session, Store


async def run(
    store: Store,
    reaction: type[Reaction],
    input: dict[str, Any],
    *,
    reaction_id: str | None = None,
) -> Context:
    """Run ``reaction`` as the seed of a new run against ``store``.

    Opens a session over ``store``, runs the reaction in a root context with
    ``reaction_id`` (fresh if not given) and no parent, and flushes the session
    on success. Any exception propagates and the session is discarded, leaving
    ``store`` untouched. The whole run happens inside ``store.run_scope()``,
    and the trace is logged whether the run succeeded or not. Returns the
    root context.
    """
    session = Session(store)
    ctx = Context(session, input, reaction_id=reaction_id)
    try:
        async with store.run_scope():
            with ctx.trace.span("run", ctx.reaction_id, None):
                await invoke(reaction, ctx)
                with ctx.trace.span("flush", ctx.reaction_id, None):
                    await session.flush(reaction_id=ctx.reaction_id)
    finally:
        ctx.trace.log(ctx.reaction_id)
    return ctx
