"""Reaction base class and the registry that maps qualified names back to classes.

Code always refers to a reaction by its class. The registry exists only for
boundaries where a class must be serialised, such as an outbox row or a CLI
argument, and is populated automatically when a subclass is defined.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend_builder.context import Context

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
