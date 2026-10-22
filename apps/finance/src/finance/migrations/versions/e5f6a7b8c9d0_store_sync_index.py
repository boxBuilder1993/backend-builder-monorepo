"""store_sync index: the latest sync per repo is what PullStore reads

Revision ID: e5f6a7b8c9d0
Revises: ffbb8de6fabf
Create Date: 2026-10-22 19:05:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "ffbb8de6fabf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("create index store_sync_repo_created_idx on store_sync (repo, created_at desc)")


def downgrade() -> None:
    op.execute("drop index store_sync_repo_created_idx")
