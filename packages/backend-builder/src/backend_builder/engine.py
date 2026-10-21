"""Run entry point: the seed of a run."""

from __future__ import annotations

from typing import Any

from backend_builder.context import Context
from backend_builder.reaction import Reaction
from backend_builder.store import Store


async def run(store: Store, reaction: type[Reaction], input: dict[str, Any]) -> Context:
    """Run ``reaction`` as the seed of a new run against ``store``.

    Builds the root context with a fresh ``reaction_id`` and no parent, runs
    the reaction, and returns the context so the caller can inspect the store.
    """
    ctx = Context(store, input)
    await reaction().calculate(ctx)
    return ctx
