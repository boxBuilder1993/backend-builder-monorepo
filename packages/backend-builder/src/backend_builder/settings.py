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
        return cls(runtime_env=runtime_env, database_url=database_url, log_level=log_level)
