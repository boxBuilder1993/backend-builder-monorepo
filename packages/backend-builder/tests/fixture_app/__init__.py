from backend_builder import App

from fixture_app.api import endpoints

app = App(package="fixture_app", endpoints=endpoints)
