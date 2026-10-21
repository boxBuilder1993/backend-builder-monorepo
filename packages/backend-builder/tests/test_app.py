import asyncio

import pytest
from backend_builder import App, Entity, InMemoryStore, Reaction
from backend_builder.entity import qualified_name as entity_name
from backend_builder.reaction import qualified_name as reaction_name
from fastapi.testclient import TestClient


class Outsider(Entity):
    pass


class OutsideReaction(Reaction):
    async def calculate(self, ctx) -> None:  # type: ignore[no-untyped-def]
        pass


def test_app_derives_reactions_and_entities_from_its_package() -> None:
    from fixture_app import app
    from fixture_app.api import MakeWidgetApi
    from fixture_app.entities import Widget
    from fixture_app.reactions import Bump, MakeWidget

    assert set(app.reactions) == {Bump, MakeWidget, MakeWidgetApi}
    assert app.entities == [Widget]
    assert Outsider not in app.entities and OutsideReaction not in app.reactions
    assert app.name == "fixture_app"


def test_app_include_brings_in_shared_classes() -> None:
    app = App(package="fixture_app", include=[Outsider, OutsideReaction])
    assert Outsider in app.entities and OutsideReaction in app.reactions


def test_resolve_is_scoped_to_the_app() -> None:
    from fixture_app import app
    from fixture_app.reactions import Bump

    assert app.resolve(reaction_name(Bump)) is Bump
    with pytest.raises(LookupError):
        app.resolve(reaction_name(OutsideReaction))


def test_http_serves_endpoints_and_scopes_the_generic_route() -> None:
    from fixture_app import app
    from fixture_app.entities import Widget
    from fixture_app.reactions import MakeWidget

    store = InMemoryStore()
    client = TestClient(app.http(store), raise_server_exceptions=False)

    res = client.post("/widgets", json={"name": "w1"})
    assert res.status_code == 200 and res.json()["name"] == "w1"
    assert len(asyncio.run(store.get(Widget))) == 1

    assert (
        client.post(f"/reactions/{reaction_name(MakeWidget)}", json={"name": "w2"}).status_code
        == 200
    )
    assert client.post(f"/reactions/{reaction_name(OutsideReaction)}", json={}).status_code == 500
    assert client.get("/openapi.json").json()["info"]["title"] == "fixture_app"


def test_describe_lists_everything() -> None:
    from fixture_app import app

    described = app.describe()
    assert entity_name(app.entities[0]) in described["entities"]
    assert len(described["reactions"]) == 3
    assert described["endpoints"] == [
        f"POST /widgets -> {reaction_name(app.endpoints[0].reaction)}"
    ]
