import pytest
from backend_builder.cli import cli
from typer.testing import CliRunner

runner = CliRunner()


def test_graph_prints_the_app() -> None:
    result = runner.invoke(cli, ["graph", "fixture_app"])
    assert result.exit_code == 0, result.output
    assert "entities:" in result.output and "fixture_app.entities.Widget" in result.output
    assert "POST /widgets -> fixture_app.api.MakeWidgetApi" in result.output


def test_trigger_runs_a_reaction_and_prints_its_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_BUILDER_DATABASE_URL", raising=False)  # in-memory store
    result = runner.invoke(
        cli,
        ["trigger", "fixture_app", "fixture_app.reactions.MakeWidget", "--input", '{"name": "x"}'],
    )
    assert result.exit_code == 0, result.output
    last = result.output.strip().splitlines()[-1]
    assert last.startswith("reaction_id=") and "total_ms=" in last


def test_trigger_rejects_non_object_input_and_unknown_app() -> None:
    assert (
        runner.invoke(
            cli, ["trigger", "fixture_app", "fixture_app.reactions.MakeWidget", "--input", "[1]"]
        ).exit_code
        != 0
    )
    assert runner.invoke(cli, ["graph", "no_such_package"]).exit_code != 0
