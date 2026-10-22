"""The app runs on Postgres (it declares schedules and its reactions query). Fixtures come from the
backend_builder_testing plugin; this file names the app, builds its store, and points the data
directory at a temporary one."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from backend_builder import App, PostgresStore
from finance import app, reactions


@pytest.fixture(scope="session")
def backend_builder_app() -> App:
    return app


@pytest.fixture
async def store(migrated_database_url: str) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(migrated_database_url, app.entities)
    yield store
    await store.close()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(reactions, "DATA_DIR", tmp_path)
    return tmp_path
