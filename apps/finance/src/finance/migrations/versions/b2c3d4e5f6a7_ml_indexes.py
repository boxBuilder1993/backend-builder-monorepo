"""ML indexes: fit name and content hash unique; the run, model and infer run lookups

Revision ID: b2c3d4e5f6a7
Revises: a17b4a14a5ae
Create Date: 2026-10-22 20:30:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a17b4a14a5ae"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("create unique index fit_name_key on fit (name)")
    op.execute("create unique index fit_content_hash_key on fit (content_hash)")
    op.execute("create index fit_run_fit_idx on fit_run (fit_id, created_at desc)")
    op.execute("create index model_fit_run_idx on model (fit_run_id)")
    op.execute("create index infer_run_model_idx on infer_run (model_id, created_at desc)")
    op.execute("create index evaluation_infer_run_idx on evaluation (infer_run_id)")


def downgrade() -> None:
    for name in (
        "evaluation_infer_run_idx",
        "infer_run_model_idx",
        "model_fit_run_idx",
        "fit_run_fit_idx",
        "fit_content_hash_key",
        "fit_name_key",
    ):
        op.execute(f"drop index {name}")
