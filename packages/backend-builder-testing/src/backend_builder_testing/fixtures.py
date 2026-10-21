"""Postgres fixtures.

``postgres_url``: the maintenance database on the local server, session
scoped. Skips dependants when the server is unreachable, except in CI.

``database_url``: a fresh, empty database for one test, dropped afterwards.

``migrated_database_url``: the same, with an app's migration chain applied.
Needs the test suite to define a ``backend_builder_app`` fixture returning
its ``App``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from backend_builder import App, migrations
from backend_builder.migrations import async_url
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

TEST_DATABASE_URL = os.environ.get(
    "BACKEND_BUILDER_TEST_DATABASE_URL",
    "postgresql://backend_builder:backend_builder@localhost:5432/backend_builder",
)


async def _execute(url: str, statement: str) -> None:
    """One statement outside a transaction, as CREATE DATABASE requires."""
    engine = create_async_engine(async_url(url), isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


def create_database(maintenance_url: str, prefix: str = "test") -> str:
    """Create an empty database with a unique name and return its URL."""
    name = f"{prefix}_{uuid4().hex[:12]}"
    asyncio.run(_execute(maintenance_url, f'create database "{name}"'))
    return make_url(maintenance_url).set(database=name).render_as_string(hide_password=False)


def drop_database(maintenance_url: str, url: str) -> None:
    name = make_url(url).database
    asyncio.run(_execute(maintenance_url, f'drop database "{name}" with (force)'))


@pytest.fixture(scope="session")
def postgres_url() -> str:
    try:
        asyncio.run(_execute(TEST_DATABASE_URL, "select 1"))
    except OSError as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"postgres not reachable at {TEST_DATABASE_URL}: {exc}")
    return TEST_DATABASE_URL


@pytest.fixture
def database_url(postgres_url: str) -> Iterator[str]:
    url = create_database(postgres_url)
    try:
        yield url
    finally:
        drop_database(postgres_url, url)


@pytest.fixture
def migrated_database_url(postgres_url: str, backend_builder_app: App) -> Iterator[str]:
    url = create_database(postgres_url, prefix=backend_builder_app.name)
    try:
        migrations.migrate(backend_builder_app, url)
        yield url
    finally:
        drop_database(postgres_url, url)
