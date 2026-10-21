"""The App: the single declaration of an application's scope.

An app is a Python package. Its reactions and entities are the ones defined
under that package, discovered by importing every submodule and filtering the
registries by name prefix. Shared classes from other packages are brought in
explicitly with ``include``. The app owns its endpoints and its store, and the
CLI derives every process it runs from this object and nothing else.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from backend_builder.engine import DEFAULT_MAX_ATTEMPTS
from backend_builder.entity import Entity, registered_entities
from backend_builder.entity import qualified_name as entity_name
from backend_builder.http import Endpoint, create_app
from backend_builder.log import get_logger
from backend_builder.reaction import Reaction, registered_reactions
from backend_builder.reaction import qualified_name as reaction_name
from backend_builder.schema import build_metadata
from backend_builder.settings import Settings
from backend_builder.store import InMemoryStore, Store

logger = get_logger(__name__)


class App:
    def __init__(
        self,
        package: str,
        *,
        endpoints: Sequence[Endpoint] = (),
        store: Callable[[], Store] | None = None,
        include: Sequence[type[Reaction] | type[Entity]] = (),
        migrations: Path | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.package = package
        self.endpoints = list(endpoints)
        self.max_attempts = max_attempts
        """How many times a run is attempted before its error propagates.
        One policy for every reaction of the app, over HTTP and the CLI."""
        self._store_factory = store
        self._include = list(include)
        root = self._import_package()
        self.migrations_dir = migrations if migrations is not None else root / "migrations"
        """Where this app's migration chain lives. Default: ``migrations/``
        inside the package."""
        self.metadata  # noqa: B018  # validate the schema now, not at first use

    @property
    def name(self) -> str:
        return self.package.rsplit(".", 1)[-1]

    def _import_package(self) -> Path:
        """Import every module under the package so all its reactions and
        entities are registered before anything is derived. Returns the
        package directory."""
        root = importlib.import_module(self.package)
        path: list[str] | None = getattr(root, "__path__", None)
        if path is None:
            raise ValueError(f"{self.package!r} is a module, not a package")
        for module in pkgutil.walk_packages(path, prefix=self.package + "."):
            importlib.import_module(module.name)
        return Path(path[0])

    def _owns(self, name: str) -> bool:
        return name.startswith(self.package + ".")

    @property
    def reactions(self) -> list[type[Reaction]]:
        owned = [r for r in registered_reactions() if self._owns(reaction_name(r))]
        included = [r for r in self._include if issubclass(r, Reaction)]
        return owned + [r for r in included if r not in owned]

    @property
    def entities(self) -> list[type[Entity]]:
        owned = [e for e in registered_entities() if self._owns(entity_name(e))]
        included = [e for e in self._include if issubclass(e, Entity)]
        return owned + [e for e in included if e not in owned]

    @cached_property
    def metadata(self) -> sa.MetaData:
        """One table per entity; raises ``SchemaError`` for an unmappable field."""
        return build_metadata(self.entities)

    def resolve(self, name: str) -> type[Reaction]:
        """A reaction of this app by qualified name."""
        for reaction in self.reactions:
            if reaction_name(reaction) == name:
                return reaction
        raise LookupError(f"{self.package} has no reaction {name!r}")

    def create_store(self) -> Store:
        """A store for this app. An explicit factory wins; otherwise the
        settings decide: Postgres when ``BACKEND_BUILDER_DATABASE_URL`` is
        set, in-memory when it is not."""
        if self._store_factory is not None:
            return self._store_factory()
        settings = Settings.from_env()
        if settings.database_url:
            from backend_builder.postgres import PostgresStore

            return PostgresStore(settings.database_url, self.entities)
        return InMemoryStore()

    def http(self, store: Store | None = None, *, generic: bool = True) -> Any:
        """The FastAPI app for this app's endpoints over ``store``, or over a
        fresh store from the factory."""
        return create_app(
            store if store is not None else self.create_store(),
            self.endpoints,
            generic=generic,
            resolver=self.resolve,
            title=self.name,
            max_attempts=self.max_attempts,
        )

    def describe(self) -> dict[str, list[str]]:
        return {
            "entities": [entity_name(e) for e in self.entities],
            "reactions": [reaction_name(r) for r in self.reactions],
            "endpoints": [
                f"{e.method} {e.path} -> {reaction_name(e.reaction)}" for e in self.endpoints
            ],
        }
