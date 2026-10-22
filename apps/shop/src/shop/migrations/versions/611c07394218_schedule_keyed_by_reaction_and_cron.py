"""schedule keyed by reaction and cron

A reaction may run on several schedules, so the schedule record is keyed by
(reaction, cron) rather than reaction alone. Existing rows keep their last
occurrence.

Revision ID: 611c07394218
Revises: ecab9173d146
Create Date: 2026-10-22 11:17:10.255972
"""

from collections.abc import Sequence

from alembic import op

revision: str = "611c07394218"
down_revision: str | None = "ecab9173d146"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("_schedule_pkey", "_schedule", type_="primary")
    op.create_primary_key("_schedule_pkey", "_schedule", ["reaction", "cron"])


def downgrade() -> None:
    # Keep one row per reaction, the most recently updated, before narrowing the key.
    op.execute(
        "delete from _schedule s using _schedule t "
        "where s.reaction = t.reaction and s.updated_at < t.updated_at"
    )
    op.drop_constraint("_schedule_pkey", "_schedule", type_="primary")
    op.create_primary_key("_schedule_pkey", "_schedule", ["reaction"])
