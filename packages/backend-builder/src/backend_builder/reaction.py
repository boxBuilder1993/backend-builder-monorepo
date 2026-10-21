"""Reaction base class, the registry, and the one place a reaction is invoked.

Code always refers to a reaction by its class. The registry exists only for
boundaries where a class must be serialised, such as an outbox row or a CLI
argument, and is populated automatically when a subclass is defined.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from backend_builder.log import get_logger

if TYPE_CHECKING:
    from backend_builder.context import Context

logger = get_logger(__name__)

_registry: dict[str, type[Reaction]] = {}


def qualified_name(reaction: type[Reaction]) -> str:
    return f"{reaction.__module__}.{reaction.__qualname__}"


def resolve(name: str) -> type[Reaction]:
    try:
        return _registry[name]
    except KeyError:
        raise LookupError(f"no reaction registered as {name!r}") from None


class Reaction(ABC):
    """A unit of logic. Subclass and implement ``calculate``."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _registry[qualified_name(cls)] = cls

    @abstractmethod
    async def calculate(self, ctx: Context) -> None:
        """Read from and write to the context. May trigger other reactions."""


async def invoke(reaction: type[Reaction], ctx: Context) -> None:
    """Run ``reaction`` in ``ctx``, recording a span and logging start, end
    and failure. Every invocation, seed or triggered, goes through here."""
    name = reaction.__name__
    logger.info(
        "reaction started reaction=%s reaction_id=%s parent_reaction_id=%s",
        name,
        ctx.reaction_id,
        ctx.parent_reaction_id,
    )
    logger.debug("reaction input reaction_id=%s input=%r", ctx.reaction_id, ctx.input)
    try:
        with ctx.trace.span(name, ctx.reaction_id, ctx.parent_reaction_id):
            await reaction().calculate(ctx)
    except Exception:
        span = ctx.trace.spans[-1]
        logger.error(
            "reaction failed reaction=%s reaction_id=%s duration_ms=%.1f",
            name,
            ctx.reaction_id,
            span.duration_ms,
        )
        raise
    span = ctx.trace.spans[-1]
    logger.info(
        "reaction finished reaction=%s reaction_id=%s duration_ms=%.1f",
        name,
        ctx.reaction_id,
        span.duration_ms,
    )
