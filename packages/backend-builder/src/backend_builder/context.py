"""The context: what a reaction sees."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from backend_builder import secrets as _secrets
from backend_builder.schema import table_name
from backend_builder.store import Session
from backend_builder.trace import Trace

if TYPE_CHECKING:
    from backend_builder.entity import Entity
    from backend_builder.reaction import Reaction


class Context:
    """An input plus a session, for one reaction invocation.

    Every invocation gets its own ``reaction_id`` and its own ``vars``. A
    context created by ``trigger`` records the triggering invocation as
    ``parent_reaction_id``; the seed invocation of a run has none. All
    invocations in a run share the same session, the entities, and the same
    trace; nothing else crosses between them. A parent passes what a child
    needs in its input and reads what the child saved from the context
    ``trigger`` returns.
    """

    def __init__(
        self,
        session: Session,
        input: dict[str, Any],
        *,
        reaction_id: str | None = None,
        parent_reaction_id: str | None = None,
        root_id: str | None = None,
        trace: Trace | None = None,
        secrets: _secrets.Secrets | None = None,
    ) -> None:
        self._session = session
        self._secrets = secrets if secrets is not None else _secrets.default
        self.input = input
        self.reaction_id = reaction_id or str(uuid4())
        self.parent_reaction_id = parent_reaction_id
        self.root_id = root_id or self.reaction_id
        """The run's seed invocation; this id for the seed itself."""
        self.trace = trace if trace is not None else Trace()
        self.vars: dict[str, Any] = {}
        """This invocation's saved values, the dict ``save`` and ``load``
        manage. Not shared with parents or children; recorded on its row."""
        self.created: list[tuple[str, str]] = []
        """``(table, id)`` of every row created through this context."""
        self.deleted: list[tuple[str, str]] = []
        self.enqueued: list[str] = []
        """Ids of the outbox rows deferred through this context."""

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
        row = self._session.create(entity, **fields)
        self.created.append((table_name(entity), row.id))
        return row

    def delete(self, row: Entity) -> None:
        """Soft-delete ``row``. It disappears from reads unless asked for."""
        self._session.delete(row)
        self.deleted.append((table_name(type(row)), row.id))

    def save(self, name: str, value: Any) -> None:
        """Keep ``value`` under ``name`` for this invocation. A parent reads
        it from the context ``trigger`` returns; a builder reads the seed's."""
        self.vars[name] = value

    def load(self, name: str) -> Any:
        """The value this invocation saved under ``name``. ``KeyError`` if absent."""
        return self.vars[name]

    def secret(self, name: str) -> str:
        """The secret ``name``, read from the environment as ``SECRET_<NAME>``.
        ``SecretError`` if unset. Values read this way are redacted from run
        records and logs."""
        return self._secrets.get(name)

    async def trigger(
        self,
        reaction: type[Reaction],
        input: dict[str, Any],
        *,
        deferred: bool = False,
        timeout: float | None = None,
    ) -> Context | None:
        """Run ``reaction`` with ``input``.

        Immediate (the default): runs now, against this session, depth-first,
        and returns the child context the reaction ran in, so the parent can
        ``load`` what the child saved. The child sees only ``input`` and the
        shared entities, never the parent's vars. ``timeout`` bounds
        it, within the run's own deadline; it can only tighten. Expiry aborts
        the whole run, because a cancelled transaction cannot be resumed.

        Deferred: records an outbox entry that the run's flush writes, so the
        reaction runs after commit, by a worker, as a run of its own. Returns
        ``None``. ``input`` must be JSON-encodable. ``timeout`` is that run's
        whole budget and may be any value; None is the app default. Postgres
        only.
        """
        import asyncio

        from backend_builder.reaction import invoke, qualified_name

        if deferred:
            row_id = self._session.defer(qualified_name(reaction), input, timeout=timeout)
            self.enqueued.append(row_id)
            return None
        child = Context(
            self._session,
            input,
            parent_reaction_id=self.reaction_id,
            root_id=self.root_id,
            trace=self.trace,
            secrets=self._secrets,
        )
        if timeout is None:
            await invoke(reaction, child)
        else:
            async with asyncio.timeout(timeout):
                await invoke(reaction, child)
        return child
