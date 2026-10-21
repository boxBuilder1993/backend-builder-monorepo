"""The shop runs on Postgres. Fixtures come from the backend_builder_testing
plugin; this file only names the app and builds its store."""

from collections.abc import AsyncIterator

import pytest
from backend_builder import App, PostgresStore, Settings, Worker
from shop import app


@pytest.fixture(scope="session")
def backend_builder_app() -> App:
    return app


@pytest.fixture(autouse=True)
def notify_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_NOTIFY_API_KEY", "test-notify-key-00000000")


@pytest.fixture
async def store(migrated_database_url: str) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(migrated_database_url, app.entities)
    yield store
    await store.close()


@pytest.fixture
def worker(store: PostgresStore) -> Worker:
    return Worker(app, store, Settings(outbox_batch=20))
