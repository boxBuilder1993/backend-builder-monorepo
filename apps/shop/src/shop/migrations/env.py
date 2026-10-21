"""Written by backend-builder. The App supplies the metadata; do not edit."""

from alembic import context
from backend_builder.migrations import run_env

run_env(context)
