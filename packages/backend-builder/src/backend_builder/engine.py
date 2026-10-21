"""Run entry point: the seed of a run."""

from __future__ import annotations

from typing import Any

from backend_builder.context import Context
from backend_builder.reaction import Reaction
from backend_builder.store import Session, Store


async def run(store: Store, reaction: type[Reaction], input: dict[str, Any]) -> Context:
    """Run ``reaction`` as the seed of a new run against ``store``.

    Opens a session over ``store``, runs the reaction in a root context with a
    fresh ``reaction_id`` and no parent, and flushes the session on success.
    Any exception propagates and the session is discarded, leaving ``store``
    untouched. Returns the root context.
    """
    session = Session(store)
    ctx = Context(session, input)
    await reaction().calculate(ctx)
    session.flush()
    return ctx
