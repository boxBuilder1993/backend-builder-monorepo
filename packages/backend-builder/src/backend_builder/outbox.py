"""The outbox: deferred work, stored in the app's database.

A row is written in the same transaction as the run that produced it, so it
exists if and only if that run committed. Workers claim rows with a lease,
run them as ordinary runs, delete them inside the run's transaction on
success, and release them with backoff on failure. After ``max_attempts``
failed deliveries a row is marked dead and kept for inspection. Every row
carries its delivery budget, ``timeout_seconds``; the lease a worker holds
on a row is that budget plus a grace period, so a row is never claimed
twice while it is legitimately running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

OUTBOX_TABLE = "_outbox"
SCHEDULE_TABLE = "_schedule"


def outbox_table(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        OUTBOX_TABLE,
        metadata,
        sa.Column("id", pg.UUID(as_uuid=False), primary_key=True),
        sa.Column("reaction", sa.Text(), nullable=False),
        sa.Column("input", pg.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("claimed_by", sa.Text(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dead_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("timeout_seconds", sa.Float(), nullable=False, server_default="60"),
        sa.Index("_outbox_claim_idx", "available_at", postgresql_where=sa.text("dead_at IS NULL")),
        sa.Index(
            "_outbox_schedule_idx",
            "reaction",
            "scheduled_for",
            unique=True,
            postgresql_where=sa.text("scheduled_for IS NOT NULL"),
        ),
    )


def schedule_table(metadata: sa.MetaData) -> sa.Table:
    """One row per scheduled reaction: the cron it was last seeded with and
    the last occurrence seeded. The scheduler advances it under a row lock."""
    return sa.Table(
        SCHEDULE_TABLE,
        metadata,
        sa.Column("reaction", sa.Text(), primary_key=True),
        sa.Column("cron", sa.Text(), nullable=False),
        sa.Column("last_scheduled_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


@dataclass(frozen=True, kw_only=True)
class OutboxEntry:
    """What a run wants delivered later. Becomes a row at flush."""

    reaction: str
    """Qualified name of the reaction."""
    input: dict[str, Any]
    """JSON-encodable."""
    available_at: datetime | None = None
    """Not before this time; ``None`` means as soon as possible."""
    scheduled_for: datetime | None = None
    """Set for rows produced by a schedule. Unique per reaction, so two
    workers seeding the same occurrence write one row."""
    timeout_seconds: float | None = None
    """The delivery's budget. ``None`` means the app default, filled in by
    the session at enqueue."""
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, kw_only=True)
class OutboxRow:
    """A claimed row, as a worker sees it."""

    id: str
    reaction: str
    input: dict[str, Any]
    attempts: int
    available_at: datetime
    created_at: datetime
    timeout_seconds: float
    scheduled_for: datetime | None = None


@dataclass(frozen=True, kw_only=True)
class OutboxStats:
    pending: int
    claimed: int
    dead: int
    oldest_pending: datetime | None


def now() -> datetime:
    return datetime.now(UTC)
