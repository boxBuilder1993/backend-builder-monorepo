"""The context: what a reaction sees."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from backend_builder.store import Session, Store
from backend_builder.trace import Trace

if TYPE_CHECKING:
    from backend_builder.entity import Entity
    from backend_builder.reaction import Reaction


class Context:
    """An input plus a store, for one reaction invocation.

    Every invocation gets its own ``reaction_id``. A context created by
    ``trigger`` records the triggering invocation as ``parent_reaction_id``;
    the seed invocation of a run has none. All invocations in a run share the
    same store and the same trace.
    """

    def __init__(
        self,
        store: Store,
        input: dict[str, Any],
        *,
        reaction_id: str | None = None,
        parent_reaction_id: str | None = None,
        trace: Trace | None = None,
    ) -> None:
        self._session = store
        self.input = input
        self.reaction_id = reaction_id or str(uuid4())
        self.parent_reaction_id = parent_reaction_id
        self.trace = trace if trace is not None else Trace()

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

    def save(self, name: str, value: Any) -> None:
        """Keep ``value`` under ``name`` in the session for the rest of the run."""
        self._vars().save(name, value)

    def load(self, name: str) -> Any:
        """The value saved under ``name`` by any reaction in this run."""
        return self._vars().load(name)

    def has(self, name: str) -> bool:
        return self._vars().has(name)

    def _vars(self) -> Session:
        if not isinstance(self._session, Session):
            raise TypeError("save/load need a Session; this context was built over a bare Store")
        return self._session

    async def trigger(self, reaction: type[Reaction], input: dict[str, Any]) -> Context:
        """Run ``reaction`` now, against this store, with ``input``.

        Returns the child context the reaction ran in.
        """
        from backend_builder.reaction import invoke

        child = Context(self._session, input, parent_reaction_id=self.reaction_id, trace=self.trace)
        await invoke(reaction, child)
        return child
