"""one run row per invocation

Every reaction invocation, child or seed, gets a _run row; _span folds into it.
Existing rows become the seeds of their own runs.

Revision ID: 2647ccdc9592
Revises: 37136244dd34
Create Date: 2026-10-23 09:53:15.934311
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2647ccdc9592"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing rows are seeds of their own runs: backfill, then tighten.
    op.add_column("_run", sa.Column("root_id", sa.UUID(as_uuid=False), nullable=True))
    op.add_column("_run", sa.Column("parent_reaction_id", sa.UUID(as_uuid=False), nullable=True))
    op.add_column("_run", sa.Column("attempt", sa.Integer(), nullable=True))
    op.add_column("_run", sa.Column("committed", sa.Boolean(), nullable=True))
    op.add_column("_run", sa.Column("flush_ms", sa.Float(), nullable=True))
    op.execute("update _run set root_id = id, attempt = attempts, committed = (outcome = 'ok')")
    op.alter_column("_run", "root_id", nullable=False)
    op.alter_column("_run", "attempt", nullable=False)
    op.alter_column("_run", "committed", nullable=False)
    op.create_index("_run_root_idx", "_run", ["root_id"], unique=False)
    op.drop_column("_run", "attempts")
    op.drop_index("_span_run_idx", table_name="_span")
    op.drop_table("_span")


def downgrade() -> None:
    # Child rows have no place in the old shape; the seeds keep their attempt count.
    op.execute("delete from _run where id <> root_id")
    op.add_column("_run", sa.Column("attempts", sa.Integer(), nullable=True))
    op.execute("update _run set attempts = attempt")
    op.alter_column("_run", "attempts", nullable=False)
    op.drop_index("_run_root_idx", table_name="_run")
    op.drop_column("_run", "flush_ms")
    op.drop_column("_run", "committed")
    op.drop_column("_run", "attempt")
    op.drop_column("_run", "parent_reaction_id")
    op.drop_column("_run", "root_id")
    op.create_table(
        "_span",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("reaction_id", sa.UUID(), nullable=False),
        sa.Column("parent_reaction_id", sa.UUID(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("ok", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="_span_pkey"),
    )
    op.create_index("_span_run_idx", "_span", ["run_id"], unique=False)
