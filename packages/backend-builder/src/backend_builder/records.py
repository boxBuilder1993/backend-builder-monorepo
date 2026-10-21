"""Run records: what each run was, received, decided, wrote and deferred.

Written in a short transaction of their own after the run finishes, success
or failure, so a failed or timed-out run is recorded too, and never on the
run's own transaction. Recording never fails a run: an error while writing
a record is logged and swallowed.

Level ``full`` stores the input and the saved vars; ``summary`` stores
everything except those; ``off`` stores nothing. A retention window, purged
by the scheduler, bounds the tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from backend_builder.codec import EncodeError, to_jsonable

RUN_TABLE = "_run"
SPAN_TABLE = "_span"

RecordLevel = Literal["full", "summary", "off"]
Outcome = Literal["ok", "error", "timeout"]


def run_tables(metadata: sa.MetaData) -> tuple[sa.Table, sa.Table]:
    run = sa.Table(
        RUN_TABLE,
        metadata,
        sa.Column("id", pg.UUID(as_uuid=False), primary_key=True),
        sa.Column("reaction", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("input", pg.JSONB(), nullable=True),
        sa.Column("vars", pg.JSONB(), nullable=True),
        sa.Column("changes", pg.JSONB(), nullable=False),
        sa.Column("enqueued", pg.JSONB(), nullable=False),
        sa.Index("_run_started_idx", "started_at"),
        sa.Index("_run_reaction_idx", "reaction", "started_at"),
    )
    span = sa.Table(
        SPAN_TABLE,
        metadata,
        sa.Column("id", pg.UUID(as_uuid=False), primary_key=True),
        sa.Column("run_id", pg.UUID(as_uuid=False), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("reaction_id", pg.UUID(as_uuid=False), nullable=False),
        sa.Column("parent_reaction_id", pg.UUID(as_uuid=False), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("ok", sa.Boolean(), nullable=False),
        sa.Index("_span_run_idx", "run_id"),
    )
    return run, span


@dataclass(frozen=True, kw_only=True)
class SpanRecord:
    attempt: int
    name: str
    reaction_id: str
    parent_reaction_id: str | None
    started_at: datetime
    duration_ms: float
    ok: bool
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, kw_only=True)
class RunRecord:
    id: str
    """The run's reaction id."""
    reaction: str
    source: str
    """Where the run came from: ``endpoint POST /orders``, ``generic``,
    ``cli``, ``outbox``, ``schedule``, or ``direct``."""
    outcome: Outcome
    error: str | None
    attempts: int
    started_at: datetime
    duration_ms: float
    input: dict[str, Any] | None
    vars: dict[str, Any] | None
    changes: list[dict[str, Any]]
    """``{"entity", "id", "kind", "changed"}`` per row the run wrote."""
    enqueued: list[str]
    """Ids of the outbox rows the run produced."""
    spans: list[SpanRecord]


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
