"""data_file indexes: path unique, (source, day) for the gap report and the ledger

Revision ID: c1a2b3d4e5f6
Revises: bd4f04f4e81d
Create Date: 2026-10-22 10:50:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c1a2b3d4e5f6"
down_revision: str | None = "bd4f04f4e81d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("create unique index data_file_path_key on data_file (path)")
    op.execute("create index data_file_source_day_idx on data_file (source, day)")
    op.execute(
        "create index data_file_unpublished_idx on data_file (path) where published_at is null"
    )


def downgrade() -> None:
    op.execute("drop index data_file_unpublished_idx")
    op.execute("drop index data_file_source_day_idx")
    op.execute("drop index data_file_path_key")
