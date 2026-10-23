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
from datetime import UTC, datetime
from typing import Any

from backend_builder.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class Span:
    name: str
    """What ran: a reaction class name, ``"flush"``, or ``"run"`` for the whole run."""
    reaction_id: str
    parent_reaction_id: str | None
    started_at: datetime
    duration_ms: float
    ok: bool
    """False when the span ended by raising."""


@dataclass(frozen=True, kw_only=True)
class Invocation:
    """What one reaction invocation received and did, captured by ``invoke``
    from its own context and nothing else. Becomes a run record row."""

    reaction: str
    """Qualified name."""
    reaction_id: str
    parent_reaction_id: str | None
    root_id: str
    started_at: datetime
    duration_ms: float
    outcome: str
    """``ok``, ``error`` or ``timeout``."""
    error: str | None
    input: dict[str, Any]
    vars: dict[str, Any]
    created: list[tuple[str, str]]
    """``(table, id)`` per row created through this context."""
    deleted: list[tuple[str, str]]
    enqueued: list[str]
    """Outbox row ids deferred through this context."""


@dataclass
class Trace:
    spans: list[Span] = field(default_factory=list[Span])
    invocations: list[Invocation] = field(default_factory=list[Invocation])
    """Every invocation of this attempt, in completion order."""
    attempt: int = 1
    """Which run attempt this trace belongs to."""

    @contextmanager
    def span(self, name: str, reaction_id: str, parent_reaction_id: str | None) -> Generator[None]:
        started_at = datetime.now(UTC)
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
                    started_at=started_at,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    ok=ok,
                )
            )

    def by_reaction_id(self, reaction_id: str) -> Span:
        for span in self.spans:
            if span.reaction_id == reaction_id and span.name not in ("flush", "run"):
                return span
        raise LookupError(reaction_id)

    def log(self, root_reaction_id: str) -> None:
        """One INFO line summarising the run; at DEBUG, one line per span
        indented by depth, in start order."""
        failed = [s.name for s in self.spans if not s.ok]
        total = next((s.duration_ms for s in self.spans if s.name == "run"), None)
        logger.info(
            "run finished reaction_id=%s total_ms=%s spans=%d failed=%s",
            root_reaction_id,
            f"{total:.1f}" if total is not None else "n/a",
            len(self.spans),
            ",".join(failed) or "none",
        )
        if not logger.isEnabledFor(10):  # logging.DEBUG
            return
        parents = {
            s.reaction_id: s.parent_reaction_id
            for s in self.spans
            if s.name not in ("flush", "run")
        }

        def depth(span: Span) -> int:
            d = 0
            parent = span.parent_reaction_id
            while parent is not None:
                d += 1
                parent = parents.get(parent)
            return d

        for span in sorted(self.spans, key=lambda s: s.started_at):
            logger.debug(
                "span %s%s reaction_id=%s duration_ms=%.1f ok=%s",
                "  " * depth(span),
                span.name,
                span.reaction_id,
                span.duration_ms,
                span.ok,
            )

    def total_ms(self) -> float:
        """Wall time of the whole run, measured as one span around the seed
        reaction and the flush."""
        for span in self.spans:
            if span.name == "run":
                return span.duration_ms
        raise LookupError("no run span; the trace did not come from run()")
