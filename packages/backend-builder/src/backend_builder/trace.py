"""Instrumentation: one span per reaction invocation and per flush.

A ``Trace`` belongs to a run and is shared by every context in it. Spans are
appended as work completes, so the list is in completion order; the parent id
on each span reconstructs the tree.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class Span:
    name: str
    """What ran: a reaction class name, or ``"flush"``."""
    reaction_id: str
    parent_reaction_id: str | None
    duration_ms: float
    ok: bool
    """False when the span ended by raising."""


@dataclass
class Trace:
    spans: list[Span] = field(default_factory=list[Span])

    @contextmanager
    def span(self, name: str, reaction_id: str, parent_reaction_id: str | None) -> Generator[None]:
        started = time.perf_counter()
        ok = False
        try:
            yield
            ok = True
        finally:
            self.spans.append(
                Span(
                    name=name,
                    reaction_id=reaction_id,
                    parent_reaction_id=parent_reaction_id,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    ok=ok,
                )
            )

    def by_reaction_id(self, reaction_id: str) -> Span:
        for span in self.spans:
            if span.reaction_id == reaction_id and span.name != "flush":
                return span
        raise LookupError(reaction_id)

    def total_ms(self) -> float:
        """Wall time of the whole run: the seed span plus the flush span."""
        return sum(s.duration_ms for s in self.spans if s.parent_reaction_id is None)
