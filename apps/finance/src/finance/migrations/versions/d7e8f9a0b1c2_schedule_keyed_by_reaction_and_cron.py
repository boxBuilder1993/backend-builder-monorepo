"""schedule keyed by reaction and cron

The runtime's _schedule table is keyed by (reaction, cron) since #24, so a reaction may run on
several schedules; this is the same change the shop's 611c07394218 carries, in this app's chain.

Revision ID: d7e8f9a0b1c2
Revises: c1a2b3d4e5f6
Create Date: 2026-10-22 11:40:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "d7e8f9a0b1c2"
down_revision: str | None = "c1a2b3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("_schedule_pkey", "_schedule", type_="primary")
    op.create_primary_key("_schedule_pkey", "_schedule", ["reaction", "cron"])


def downgrade() -> None:
    op.execute(
        "delete from _schedule s using _schedule t "
        "where s.reaction = t.reaction and s.updated_at < t.updated_at"
    )
    op.drop_constraint("_schedule_pkey", "_schedule", type_="primary")
    op.create_primary_key("_schedule_pkey", "_schedule", ["reaction"])
