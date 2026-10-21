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
from typing import Any

from backend_builder.entity import Entity, registered_entities
from backend_builder.entity import qualified_name as entity_name
from backend_builder.http import Endpoint, create_app
from backend_builder.log import get_logger
from backend_builder.reaction import Reaction, registered_reactions
from backend_builder.reaction import qualified_name as reaction_name
from backend_builder.store import InMemoryStore, Store

logger = get_logger(__name__)


class App:
    def __init__(
        self,
        package: str,
        *,
        endpoints: Sequence[Endpoint] = (),
        store: Callable[[], Store] = InMemoryStore,
        include: Sequence[type[Reaction] | type[Entity]] = (),
    ) -> None:
        self.package = package
        self.endpoints = list(endpoints)
        self._store_factory = store
        self._include = list(include)
        self._import_package()

    @property
    def name(self) -> str:
        return self.package.rsplit(".", 1)[-1]

    def _import_package(self) -> None:
        """Import every module under the package so all its reactions and
        entities are registered before anything is derived."""
        root = importlib.import_module(self.package)
        path: list[str] | None = getattr(root, "__path__", None)
        if path is None:
            return
        for module in pkgutil.walk_packages(path, prefix=self.package + "."):
            importlib.import_module(module.name)

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

    def resolve(self, name: str) -> type[Reaction]:
        """A reaction of this app by qualified name."""
        for reaction in self.reactions:
            if reaction_name(reaction) == name:
                return reaction
        raise LookupError(f"{self.package} has no reaction {name!r}")

    def create_store(self) -> Store:
        return self._store_factory()

    def http(self, store: Store | None = None, *, generic: bool = True) -> Any:
        """The FastAPI app for this app's endpoints over ``store``, or over a
        fresh store from the factory."""
        return create_app(
            store if store is not None else self.create_store(),
            self.endpoints,
            generic=generic,
            resolver=self.resolve,
            title=self.name,
        )

    def describe(self) -> dict[str, list[str]]:
        return {
            "entities": [entity_name(e) for e in self.entities],
            "reactions": [reaction_name(r) for r in self.reactions],
            "endpoints": [
                f"{e.method} {e.path} -> {reaction_name(e.reaction)}" for e in self.endpoints
            ],
        }
