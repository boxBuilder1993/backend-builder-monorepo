import pytest
from backend_builder import App, InMemoryStore, PostgresStore, RuntimeEnv, Settings, SettingsError
from backend_builder.cli import http_factory


def test_defaults() -> None:
    s = Settings.from_env({})
    assert s.runtime_env is RuntimeEnv.DEV and s.database_url is None
    assert s.log_level == "INFO" and s.store_name == "InMemoryStore"


def test_values_are_read_and_normalised() -> None:
    s = Settings.from_env(
        {
            "RUNTIME_ENV": " Test ",
            "BACKEND_BUILDER_DATABASE_URL": "postgresql://u:p@h/db ",
            "BACKEND_BUILDER_LOG_LEVEL": "debug",
        }
    )
    assert s.runtime_env is RuntimeEnv.TEST
    assert s.database_url == "postgresql://u:p@h/db" and s.store_name == "PostgresStore"
    assert s.log_level == "DEBUG"


def test_invalid_env_is_rejected() -> None:
    with pytest.raises(SettingsError, match="expected one of test, dev, prod"):
        Settings.from_env({"RUNTIME_ENV": "staging"})


def test_prod_requires_a_database() -> None:
    with pytest.raises(SettingsError, match="requires BACKEND_BUILDER_DATABASE_URL"):
        Settings.from_env({"RUNTIME_ENV": "prod"})
    s = Settings.from_env({"RUNTIME_ENV": "prod", "BACKEND_BUILDER_DATABASE_URL": "postgresql://x"})
    assert s.runtime_env is RuntimeEnv.PROD


def test_app_store_follows_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    app = App(package="fixture_app")
    monkeypatch.delenv("BACKEND_BUILDER_DATABASE_URL", raising=False)
    assert isinstance(app.create_store(), InMemoryStore)

    monkeypatch.setenv("BACKEND_BUILDER_DATABASE_URL", "postgresql://u:p@localhost:1/none")
    store = app.create_store()
    assert isinstance(store, PostgresStore) and store.url.startswith("postgresql+asyncpg://")

    explicit = App(package="fixture_app", store=InMemoryStore)
    assert isinstance(explicit.create_store(), InMemoryStore)


def test_http_factory_uses_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND_BUILDER_APP", "fixture_app")
    monkeypatch.setenv("BACKEND_BUILDER_DATABASE_URL", "postgresql://u:p@localhost:1/none")
    assert http_factory().title == "fixture_app"
