"""The admin console: the app explaining itself to the people running it.

A separate process, ``backend-builder admin <app>``, on its own port, never
the API's. Server-rendered pages over the run records, the outbox and the
schedule table, plus the app's own declarations. Postgres only, since that
is where the records live. Optionally gated by ``BACKEND_BUILDER_ADMIN_TOKEN``.
"""

from backend_builder.admin.app import create_admin_app

__all__ = ["create_admin_app"]
