import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from backend_builder.migrations import async_url
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

# Make the fixture package importable by name, as an installed app would be.
sys.path.insert(0, str(Path(__file__).parent))

TEST_DATABASE_URL = os.environ.get(
    "BACKEND_BUILDER_TEST_DATABASE_URL",
    "postgresql://backend_builder:backend_builder@localhost:5432/backend_builder",
)


async def _run(url: str, statement: str) -> None:
    """Execute one statement outside a transaction, as CREATE DATABASE needs."""
    engine = create_async_engine(async_url(url), isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """The maintenance database on the local Postgres. Skips every test that
    needs it when the server is unreachable, except in CI where it must exist."""
    try:
        asyncio.run(_run(TEST_DATABASE_URL, "select 1"))
    except OSError as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"postgres not reachable at {TEST_DATABASE_URL}: {exc}")
    return TEST_DATABASE_URL


@pytest.fixture
def database_url(postgres_url: str) -> Iterator[str]:
    """A fresh, empty database for one test, dropped afterwards."""
    name = "test_" + uuid4().hex[:12]
    asyncio.run(_run(postgres_url, f'create database "{name}"'))
    try:
        yield make_url(postgres_url).set(database=name).render_as_string(hide_password=False)
    finally:
        asyncio.run(_run(postgres_url, f'drop database "{name}" with (force)'))
