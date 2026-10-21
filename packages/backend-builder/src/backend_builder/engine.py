"""Run entry point: the seed of a run."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from backend_builder.context import Context
from backend_builder.log import get_logger
from backend_builder.reaction import Reaction, invoke
from backend_builder.store import Session, Store, TransientError

logger = get_logger(__name__)


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
    ``store`` untouched. The whole run happens inside ``store.run_scope()``.

    A ``TransientError`` from the store, such as a serialization conflict,
    makes the whole run start again with a fresh session, up to
    ``store.max_attempts`` times. A run is a function of its input and the
    store, so a retry is safe. The trace is logged for every attempt.
    """
    attempts = max(1, store.max_attempts)
    for attempt in range(1, attempts + 1):
        session = Session(store)
        ctx = Context(session, input, reaction_id=reaction_id)
        try:
            async with store.run_scope():
                with ctx.trace.span("run", ctx.reaction_id, None):
                    await invoke(reaction, ctx)
                    with ctx.trace.span("flush", ctx.reaction_id, None):
                        await session.flush(reaction_id=ctx.reaction_id)
            return ctx
        except TransientError as exc:
            if attempt == attempts:
                logger.error(
                    "run gave up reaction=%s reaction_id=%s attempts=%d error=%s",
                    reaction.__name__,
                    ctx.reaction_id,
                    attempt,
                    exc,
                )
                raise
            delay = random.uniform(0, 0.05 * attempt)
            logger.warning(
                "run retrying reaction=%s reaction_id=%s attempt=%d delay_ms=%.0f error=%s",
                reaction.__name__,
                ctx.reaction_id,
                attempt,
                delay * 1000,
                exc,
            )
            await asyncio.sleep(delay)
        finally:
            ctx.trace.log(ctx.reaction_id)
    raise AssertionError("unreachable")
