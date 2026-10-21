"""The context: what a reaction sees."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from backend_builder.store import Store

if TYPE_CHECKING:
    from backend_builder.entity import Entity
    from backend_builder.reaction import Reaction


class Context:
    """An input plus a store, for one reaction invocation.

    Every invocation gets its own ``reaction_id``. A context created by
    ``trigger`` records the triggering invocation as ``parent_reaction_id``;
    the seed invocation of a run has none. All invocations in a run share the
    same store.
    """

    def __init__(
        self,
        store: Store,
        input: dict[str, Any],
        *,
        reaction_id: str | None = None,
        parent_reaction_id: str | None = None,
    ) -> None:
        self._session = store
        self.input = input
        self.reaction_id = reaction_id or str(uuid4())
        self.parent_reaction_id = parent_reaction_id

    def get[E: Entity](
        self,
        entity: type[E],
        where: Callable[[E], bool] | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[E]:
        return self._session.get(entity, where, include_deleted=include_deleted)

    def get_by_id[E: Entity](
        self, entity: type[E], row_id: str, *, include_deleted: bool = False
    ) -> E | None:
        return self._session.get_by_id(entity, row_id, include_deleted=include_deleted)

    def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        return self._session.create(entity, **fields)

    async def trigger(self, reaction: type[Reaction], input: dict[str, Any]) -> Context:
        """Run ``reaction`` now, against this store, with ``input``.

        Returns the child context the reaction ran in.
        """
        child = Context(self._session, input, parent_reaction_id=self.reaction_id)
        await reaction().calculate(child)
        return child
