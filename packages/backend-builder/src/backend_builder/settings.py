"""Runtime settings, read from the environment.

``RUNTIME_ENV`` selects ``test``, ``dev`` or ``prod``; unset means ``dev``.
``BACKEND_BUILDER_DATABASE_URL`` selects the store: unset means in-memory,
set means Postgres at that URL. ``prod`` refuses to run without a database.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

RUNTIME_ENV = "RUNTIME_ENV"
DATABASE_URL = "BACKEND_BUILDER_DATABASE_URL"
LOG_LEVEL = "BACKEND_BUILDER_LOG_LEVEL"
OUTBOX_POLL_SECONDS = "BACKEND_BUILDER_OUTBOX_POLL_SECONDS"
OUTBOX_BATCH = "BACKEND_BUILDER_OUTBOX_BATCH"
OUTBOX_LEASE_GRACE_SECONDS = "BACKEND_BUILDER_OUTBOX_LEASE_GRACE_SECONDS"
OUTBOX_MAX_ATTEMPTS = "BACKEND_BUILDER_OUTBOX_MAX_ATTEMPTS"
OUTBOX_CONCURRENCY = "BACKEND_BUILDER_OUTBOX_CONCURRENCY"
SCHEDULER_TICK_SECONDS = "BACKEND_BUILDER_SCHEDULER_TICK_SECONDS"


class RuntimeEnv(StrEnum):
    TEST = "test"
    DEV = "dev"
    PROD = "prod"


class SettingsError(Exception):
    pass


@dataclass(frozen=True, kw_only=True)
class Settings:
    runtime_env: RuntimeEnv = RuntimeEnv.DEV
    database_url: str | None = None
    log_level: str = "INFO"
    outbox_poll_seconds: float = 1.0
    outbox_batch: int = 10
    outbox_lease_grace_seconds: float = 10.0
    """Added to a row's own timeout to form the lease a worker holds on it."""
    outbox_max_attempts: int = 5
    outbox_concurrency: int = 4
    scheduler_tick_seconds: float = 15.0
    """Must stay under the finest cron granularity, one minute, so no
    occurrence is missed while the scheduler runs."""

    @property
    def store_name(self) -> str:
        return "PostgresStore" if self.database_url else "InMemoryStore"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        raw_env = env.get(RUNTIME_ENV, RuntimeEnv.DEV.value).strip().lower()
        try:
            runtime_env = RuntimeEnv(raw_env)
        except ValueError:
            allowed = ", ".join(e.value for e in RuntimeEnv)
            raise SettingsError(f"{RUNTIME_ENV}={raw_env!r}; expected one of {allowed}") from None
        database_url = env.get(DATABASE_URL, "").strip() or None
        if runtime_env is RuntimeEnv.PROD and database_url is None:
            raise SettingsError(f"{RUNTIME_ENV}=prod requires {DATABASE_URL}")
        log_level = env.get(LOG_LEVEL, "INFO").strip().upper() or "INFO"
        return cls(
            runtime_env=runtime_env,
            database_url=database_url,
            log_level=log_level,
            outbox_poll_seconds=_number(env, OUTBOX_POLL_SECONDS, 1.0),
            outbox_batch=int(_number(env, OUTBOX_BATCH, 10)),
            outbox_lease_grace_seconds=_number(env, OUTBOX_LEASE_GRACE_SECONDS, 10.0),
            outbox_max_attempts=int(_number(env, OUTBOX_MAX_ATTEMPTS, 5)),
            outbox_concurrency=int(_number(env, OUTBOX_CONCURRENCY, 4)),
            scheduler_tick_seconds=_number(env, SCHEDULER_TICK_SECONDS, 15.0),
        )


def _number(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SettingsError(f"{name}={raw!r} is not a number") from None
    if value <= 0:
        raise SettingsError(f"{name} must be positive")
    return value
