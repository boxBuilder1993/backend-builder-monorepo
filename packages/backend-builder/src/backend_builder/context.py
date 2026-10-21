"""The context: what a reaction sees."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from backend_builder.store import Session
from backend_builder.trace import Trace

if TYPE_CHECKING:
    from backend_builder.entity import Entity
    from backend_builder.reaction import Reaction


class Context:
    """An input plus a session, for one reaction invocation.

    Every invocation gets its own ``reaction_id``. A context created by
    ``trigger`` records the triggering invocation as ``parent_reaction_id``;
    the seed invocation of a run has none. All invocations in a run share the
    same session and the same trace.
    """

    def __init__(
        self,
        session: Session,
        input: dict[str, Any],
        *,
        reaction_id: str | None = None,
        parent_reaction_id: str | None = None,
        trace: Trace | None = None,
    ) -> None:
        self._session = session
        self.input = input
        self.reaction_id = reaction_id or str(uuid4())
        self.parent_reaction_id = parent_reaction_id
        self.trace = trace if trace is not None else Trace()

    async def get[E: Entity](
        self,
        entity: type[E],
        where: Callable[[E], bool] | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[E]:
        """Rows matching a Python predicate. In-memory stores only."""
        return await self._session.get(entity, where, include_deleted=include_deleted)

    async def get_by_id[E: Entity](
        self, entity: type[E], row_id: str, *, include_deleted: bool = False
    ) -> E | None:
        return await self._session.get_by_id(entity, row_id, include_deleted=include_deleted)

    async def query[E: Entity](self, entity: type[E], statement: str, **params: Any) -> list[E]:
        """Rows of ``entity`` returned by a native statement. SQL stores only."""
        return await self._session.query(entity, statement, **params)

    async def query_raw(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        """Untracked rows returned by a native statement. SQL stores only."""
        return await self._session.query_raw(statement, **params)

    def create[E: Entity](self, entity: type[E], **fields: Any) -> E:
        return self._session.create(entity, **fields)

    def delete(self, row: Entity) -> None:
        """Soft-delete ``row``. It disappears from reads unless asked for."""
        self._session.delete(row)

    def save(self, name: str, value: Any) -> None:
        """Keep ``value`` under ``name`` in the session for the rest of the run."""
        self._session.vars[name] = value

    def load(self, name: str) -> Any:
        """The value saved under ``name`` by any reaction in this run."""
        return self._session.vars[name]

    async def trigger(
        self, reaction: type[Reaction], input: dict[str, Any], *, deferred: bool = False
    ) -> Context | None:
        """Run ``reaction`` with ``input``.

        Immediate (the default): runs now, against this session, depth-first,
        and returns the child context the reaction ran in.

        Deferred: records an outbox entry that the run's flush writes, so the
        reaction runs after commit, by a worker, as a run of its own. Returns
        ``None``. ``input`` must be JSON-encodable. Postgres only.
        """
        from backend_builder.reaction import invoke, qualified_name

        if deferred:
            self._session.defer(qualified_name(reaction), input)
            return None
        child = Context(self._session, input, parent_reaction_id=self.reaction_id, trace=self.trace)
        await invoke(reaction, child)
        return child
