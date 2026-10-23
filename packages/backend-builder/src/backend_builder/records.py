"""Run records: one row per reaction invocation, however it was triggered.

Every invocation, the seed of a run or a child triggered from another
reaction, leaves a row in ``_run`` describing its own context: what it
received, what it saved, what it created or deleted, what it deferred, how
long it took and how it ended. The seed's row additionally carries the facts
that exist only at commit: the attempt count, the flush, every field change
the run wrote, and whether it committed. ``root_id`` ties the rows of one run
together; ``parent_reaction_id`` gives the tree.

The seed's row is written as ``running`` before the first attempt and
replaced, together with every child's row, when the run finishes, success or
failure; both writes are short transactions of their own, never the run's.
Recording never fails a run: an error while writing is logged and swallowed.
A ``running`` row whose deadline has passed means the process died mid-run;
the scheduler marks it ``lost``.

Level ``full`` stores inputs and saved vars; ``summary`` stores everything
except those; ``off`` stores nothing. A retention window, purged by the
scheduler, bounds the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from backend_builder.codec import EncodeError, to_jsonable

RUN_TABLE = "_run"
LOST_ERROR = "lost: the process stopped reporting before the run's deadline passed"
TRIGGER_SOURCE = "trigger"
"""The ``source`` of every child invocation."""

RecordLevel = Literal["full", "summary", "off"]
Outcome = Literal["running", "ok", "error", "timeout", "lost"]
"""``running`` is written when a run starts and replaced when it ends.
``lost`` is set by the scheduler when a ``running`` record's deadline has
passed and nothing replaced it: the process died mid-run."""


def run_tables(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        RUN_TABLE,
        metadata,
        sa.Column("id", pg.UUID(as_uuid=False), primary_key=True),
        sa.Column("root_id", pg.UUID(as_uuid=False), nullable=False),
        sa.Column("parent_reaction_id", pg.UUID(as_uuid=False), nullable=True),
        sa.Column("reaction", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("committed", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("flush_ms", sa.Float(), nullable=True),
        sa.Column("input", pg.JSONB(), nullable=True),
        sa.Column("vars", pg.JSONB(), nullable=True),
        sa.Column("changes", pg.JSONB(), nullable=False),
        sa.Column("enqueued", pg.JSONB(), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Index("_run_started_idx", "started_at"),
        sa.Index("_run_reaction_idx", "reaction", "started_at"),
        sa.Index("_run_root_idx", "root_id"),
        sa.Index(
            "_run_running_idx",
            "deadline_at",
            postgresql_where=sa.text("outcome = 'running'"),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class RunRecord:
    """One invocation's row. The seed of a run has ``id == root_id``."""

    id: str
    """The invocation's reaction id."""
    root_id: str
    """The run's seed invocation; equal to ``id`` for the seed itself."""
    parent_reaction_id: str | None
    reaction: str
    source: str
    """Where the invocation came from: ``endpoint POST /orders``, ``generic``,
    ``cli``, ``outbox``, ``schedule``, ``direct`` for a seed; ``trigger`` for
    a child."""
    attempt: int
    """For a child, the run attempt it ran in. For the seed, how many
    attempts the run made."""
    outcome: Outcome
    error: str | None
    committed: bool
    """Whether the run this invocation belongs to committed. A child of a
    failed attempt never did, even if it ended ``ok`` itself."""
    started_at: datetime
    duration_ms: float
    flush_ms: float | None
    """Seed only: how long the committing flush took."""
    input: dict[str, Any] | None
    vars: dict[str, Any] | None
    """What this invocation saved. Never its parent's or children's."""
    changes: list[dict[str, Any]]
    """``{"entity", "id", "kind", "changed"}`` rows. For a child, what it
    created or deleted through its context. For the seed, every row the run's
    flush wrote, field changes included, since those are only known at
    flush."""
    enqueued: list[str]
    """For a child, the outbox rows it deferred. For the seed, every outbox
    row the run's flush wrote: children's deferrals and data-change trigger
    matches included, since the latter belong to the flush, not to any
    invocation."""
    deadline_at: datetime | None = None
    """Seed only: when the run's budget expires. A ``running`` record past it
    is lost."""

    @property
    def is_root(self) -> bool:
        return self.id == self.root_id


def safe_jsonable(value: Any) -> Any:
    """Encode for storage; a value the codec rejects becomes its ``repr``
    rather than failing the record."""
    try:
        return to_jsonable(value)
    except EncodeError:
        if isinstance(value, dict):
            return {str(k): safe_jsonable(v) for k, v in value.items()}  # type: ignore[union-attr]
        if isinstance(value, list | tuple):
            return [safe_jsonable(v) for v in value]  # type: ignore[union-attr]
        return repr(value)
