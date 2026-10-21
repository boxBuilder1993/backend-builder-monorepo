import asyncio
from pathlib import Path

import pytest
from backend_builder import App, Entity, migrations
from backend_builder.cli import cli
from backend_builder.migrations import async_url
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from typer.testing import CliRunner


class Outsider(Entity):
    note: str = ""


async def _columns(url: str, table: str) -> list[str]:
    engine = create_async_engine(async_url(url))
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text(
                    "select column_name from information_schema.columns "
                    "where table_name = :table order by ordinal_position"
                ),
                {"table": table},
            )
            return [str(row[0]) for row in result]
    finally:
        await engine.dispose()


@pytest.fixture
def app(tmp_path: Path) -> App:
    return App(package="fixture_app", migrations=tmp_path / "migrations")


def test_initial_migration_creates_the_tables(app: App, database_url: str) -> None:
    path = migrations.makemigration(app, database_url, "initial")

    assert path.parent == app.migrations_dir / "versions" and path.name.endswith("_initial.py")
    source = path.read_text()
    assert "op.create_table(" in source and "'widget'" in source
    assert "sa.Column('count', sa.BigInteger(), nullable=False)" in source
    assert (app.migrations_dir / "env.py").exists()

    migrations.migrate(app, database_url)
    assert asyncio.run(_columns(database_url, "widget")) == [
        "id",
        "created_at",
        "deleted_at",
        "name",
        "count",
    ]


def test_check_passes_when_migrated_and_fails_on_drift(
    app: App, database_url: str, tmp_path: Path
) -> None:
    with pytest.raises(migrations.MigrationError, match="differ"):
        migrations.check(app, database_url)

    migrations.makemigration(app, database_url, "initial")
    migrations.migrate(app, database_url)
    migrations.check(app, database_url)

    drifted = App(package="fixture_app", include=[Outsider], migrations=app.migrations_dir)
    with pytest.raises(migrations.MigrationError, match="differ"):
        migrations.check(drifted, database_url)


def test_empty_migration_is_a_scaffold_for_sql(app: App, database_url: str) -> None:
    migrations.makemigration(app, database_url, "initial")
    migrations.migrate(app, database_url)
    path = migrations.makemigration(app, database_url, "widget name index", empty=True)
    source = path.read_text()
    assert "def upgrade() -> None:\n    pass" in source
    assert path.name.endswith("_widget_name_index.py")


def test_cli_commands(
    app: App, database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.setenv("BACKEND_BUILDER_DATABASE_URL", database_url)
    # The CLI loads the fixture app with its default migrations dir; point it elsewhere.
    monkeypatch.setattr("fixture_app.app.migrations_dir", tmp_path / "cli_migrations")

    result = runner.invoke(cli, ["makemigration", "fixture_app", "-m", "initial"])
    assert result.exit_code == 0, result.output
    assert "_initial.py" in result.output.strip().splitlines()[-1]

    result = runner.invoke(cli, ["migrate", "fixture_app"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(cli, ["migrate", "fixture_app", "--check"])
    assert result.exit_code == 0, result.output
    assert "entities match migrations" in result.output

    monkeypatch.delenv("BACKEND_BUILDER_DATABASE_URL")
    assert runner.invoke(cli, ["migrate", "fixture_app"]).exit_code != 0
