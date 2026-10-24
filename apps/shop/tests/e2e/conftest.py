"""End-to-end: the real stack from compose.yaml, driven over HTTP.

The ``stack`` fixture builds the image and brings up Postgres, migrate, api,
worker and scheduler under their own compose project on their own ports,
with fast poll and tick settings, and tears everything down afterwards.
Tests talk to the published API port and inspect the database through the
published Postgres port. Skipped when Docker is unavailable, except in CI,
where it must be.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from backend_builder.migrations import async_url
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

REPO = Path(__file__).resolve().parents[4]
PROJECT = "backend-builder-e2e"
API_PORT = int(os.environ.get("E2E_API_PORT", "18080"))
POSTGRES_PORT = int(os.environ.get("E2E_POSTGRES_PORT", "15432"))
METRICS_PORTS = {"api": 19100, "worker": 19101, "scheduler": 19102}
ADMIN_PORT = int(os.environ.get("E2E_ADMIN_PORT", "19200"))
LOKI_PORT = int(os.environ.get("E2E_LOKI_PORT", "13100"))
GRAFANA_PORT = int(os.environ.get("E2E_GRAFANA_PORT", "13000"))
NOTIFY_KEY = "e2e-notify-key-5f3a9c1d"

ENV = {
    **os.environ,
    "COMPOSE_PROJECT_NAME": PROJECT,
    "API_PORT": str(API_PORT),
    "POSTGRES_PORT": str(POSTGRES_PORT),
    "ADMIN_PORT": str(ADMIN_PORT),
    "LOKI_PORT": str(LOKI_PORT),
    "GRAFANA_PORT": str(GRAFANA_PORT),
    "API_METRICS_PORT": str(METRICS_PORTS["api"]),
    "WORKER_METRICS_PORT": str(METRICS_PORTS["worker"]),
    "SCHEDULER_METRICS_PORT": str(METRICS_PORTS["scheduler"]),
    "BACKEND_BUILDER_OUTBOX_POLL_SECONDS": "0.2",
    "BACKEND_BUILDER_SCHEDULER_TICK_SECONDS": "1",
    "BACKEND_BUILDER_OUTBOX_MAX_ATTEMPTS": "2",
    "SECRET_NOTIFY_API_KEY": NOTIFY_KEY,
}


def compose(
    *args: str, check: bool = True, timeout: float = 600
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "--profile", "shop", "--profile", "logs", *args],
        cwd=REPO,
        env=ENV,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


@dataclass(frozen=True)
class Stack:
    base_url: str
    database_url: str
    admin_url: str = f"http://127.0.0.1:{ADMIN_PORT}"
    loki_url: str = f"http://127.0.0.1:{LOKI_PORT}"

    async def sql(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        """Run a statement against the stack's database and return rows."""
        engine = create_async_engine(async_url(self.database_url), isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as connection:
                result = await connection.execute(text(statement), params)
                if not result.returns_rows:
                    return []
                return [dict(row) for row in result.mappings().all()]
        finally:
            await engine.dispose()

    def logs(self, service: str) -> str:
        return compose("logs", "--no-log-prefix", service, check=False).stdout

    def cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a backend-builder command in a one-off container on the stack."""
        return compose("run", "--rm", "--no-deps", "-T", "api", *args, check=False, timeout=120)

    def metrics_url(self, service: str) -> str:
        return f"http://127.0.0.1:{METRICS_PORTS[service]}/metrics"

    def status(self) -> dict[str, str]:
        out = compose("ps", "--all", "--format", "{{.Service}}\t{{.Status}}", check=False).stdout
        return dict(line.split("\t", 1) for line in out.strip().splitlines() if "\t" in line)


async def wait_for[T](
    probe: Callable[[], Awaitable[T]],
    ok: Callable[[T], bool],
    *,
    timeout: float = 20,
    every: float = 0.25,
) -> T:
    """Poll ``probe`` until ``ok`` accepts its value or ``timeout`` passes."""
    deadline = time.monotonic() + timeout
    last: T = await probe()
    while not ok(last):
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout}s; last value: {last!r}")
        await asyncio.sleep(every)
        last = await probe()
    return last


@pytest.fixture(scope="session")
def stack() -> Iterator[Stack]:
    if shutil.which("docker") is None or compose("version", check=False).returncode != 0:
        if os.environ.get("CI"):
            raise RuntimeError("docker compose is required in CI")
        pytest.skip("docker compose not available")

    compose("down", "--volumes", "--remove-orphans", check=False)
    up = compose("up", "--build", "--detach", "--wait", check=False)
    if up.returncode != 0:
        logs = compose("logs", "--no-log-prefix", check=False).stdout
        compose("down", "--volumes", "--remove-orphans", check=False)
        raise RuntimeError(f"stack failed to start:\n{up.stderr}\n--- logs ---\n{logs[-4000:]}")
    try:
        yield Stack(
            base_url=f"http://127.0.0.1:{API_PORT}",
            database_url=(
                f"postgresql://backend_builder:backend_builder@127.0.0.1:{POSTGRES_PORT}/backend_builder"
            ),
        )
    finally:
        compose("down", "--volumes", "--remove-orphans", check=False)
