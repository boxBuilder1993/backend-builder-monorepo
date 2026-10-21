"""Run entry point: the seed of a run."""

from __future__ import annotations

import asyncio
import random
from typing import Any
from uuid import uuid4

from backend_builder.context import Context
from backend_builder.log import get_logger
from backend_builder.reaction import Reaction, invoke
from backend_builder.store import Session, Store

logger = get_logger(__name__)

DEFAULT_MAX_ATTEMPTS = 3


async def run(
    store: Store,
    reaction: type[Reaction],
    input: dict[str, Any],
    *,
    reaction_id: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    outbox_row_id: str | None = None,
) -> Context:
    """Run ``reaction`` as the seed of a new run against ``store``.

    Opens a session over ``store``, runs the reaction in a root context with
    ``reaction_id`` (fresh if not given) and no parent, and flushes the session
    on success. The whole run happens inside ``store.run_scope()``.

    Any exception discards the session, leaving ``store`` untouched, and the
    run starts again from the seed with a fresh session, up to
    ``max_attempts`` times with a short jittered pause between attempts. A run
    is a function of its input and the store, so a retry is safe. The last
    attempt's exception propagates. The trace is logged for every attempt.

    ``outbox_row_id`` names the outbox row this run delivers; it is deleted
    inside the run's transaction after the flush, so the run's writes and
    the row's removal commit together.
    """
    attempts = max(1, max_attempts)
    reaction_id = reaction_id or str(uuid4())  # one id for the run, whatever the attempt count
    for attempt in range(1, attempts + 1):
        session = Session(store)
        ctx = Context(session, input, reaction_id=reaction_id)
        try:
            async with store.run_scope():
                with ctx.trace.span("run", ctx.reaction_id, None):
                    await invoke(reaction, ctx)
                    with ctx.trace.span("flush", ctx.reaction_id, None):
                        await session.flush(reaction_id=ctx.reaction_id)
                    if outbox_row_id is not None:
                        await store.ack(outbox_row_id)
            return ctx
        except Exception as exc:
            if attempt == attempts:
                logger.error(
                    "run gave up reaction=%s reaction_id=%s attempts=%d error=%s: %s",
                    reaction.__name__,
                    ctx.reaction_id,
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                raise
            delay = random.uniform(0, 0.05 * attempt)
            logger.warning(
                "run retrying reaction=%s reaction_id=%s attempt=%d delay_ms=%.0f error=%s: %s",
                reaction.__name__,
                ctx.reaction_id,
                attempt,
                delay * 1000,
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(delay)
        finally:
            ctx.trace.log(ctx.reaction_id)
    raise AssertionError("unreachable")
