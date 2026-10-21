"""The context: what a reaction sees."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend_builder.store import Predicate, Row, Store

if TYPE_CHECKING:
    from backend_builder.entity import Entity
    from backend_builder.reaction import Reaction


class Context:
    """An input plus a store. One is built per reaction invocation; all
    invocations in a run share the same store."""

    def __init__(self, store: Store, input: Any) -> None:
        self.store = store
        self.input = input

    def get(self, entity: type[Entity], where: Predicate | None = None) -> list[Row]:
        return self.store.get(entity, where)

    def get_by_id(self, entity: type[Entity], row_id: str) -> Row | None:
        return self.store.get_by_id(entity, row_id)

    def create(self, entity: type[Entity], **fields: Any) -> Row:
        return self.store.create(entity, **fields)

    async def trigger(self, reaction: type[Reaction], input: Any) -> None:
        """Run ``reaction`` now, against this store, with ``input``."""
        await reaction().calculate(Context(self.store, input))
