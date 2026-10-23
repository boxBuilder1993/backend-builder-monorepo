"""Run entry point: the seed of a run."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from backend_builder import metrics, secrets
from backend_builder.context import Context
from backend_builder.log import get_logger
from backend_builder.reaction import Reaction, invoke, qualified_name
from backend_builder.records import TRIGGER_SOURCE, Outcome, RecordLevel, RunRecord, safe_jsonable
from backend_builder.schema import table_name
from backend_builder.store import NotSupported, Session, Store
from backend_builder.trace import Trace

if TYPE_CHECKING:
    from backend_builder.triggers import Trigger

logger = get_logger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT = 60.0


async def run(
    store: Store,
    reaction: type[Reaction],
    input: dict[str, Any],
    *,
    reaction_id: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout: float = DEFAULT_TIMEOUT,
    outbox_row_id: str | None = None,
    triggers: Sequence[Trigger] = (),
    source: str = "direct",
    records: RecordLevel = "full",
) -> Context:
    """Run ``reaction`` as the seed of a new run against ``store``.

    Opens a session over ``store``, runs the reaction in a root context with
    ``reaction_id`` (fresh if not given) and no parent, and flushes the session
    on success. The whole run happens inside ``store.run_scope()``.

    Any exception discards the session, leaving ``store`` untouched, and the
    run starts again from the seed with a fresh session, up to
    ``max_attempts`` times with a short jittered pause between attempts. A run
    is a function of its input and the store, so a retry is safe. The last
    attempt's exception propagates. The trace is logged for every attempt.

    ``timeout`` is the budget for the whole run, every attempt included. When
    it expires the current attempt is cancelled, its transaction rolls back,
    and ``TimeoutError`` propagates. A cancelled transaction is never resumed.

    ``outbox_row_id`` names the outbox row this run delivers; it is deleted
    inside the run's transaction after the flush, so the run's writes and
    the row's removal commit together. ``triggers`` are the app's data-change
    triggers, matched at flush.

    A ``running`` record for the seed is written to the store before the
    first attempt and replaced after the run, whatever its outcome, together
    with one record per invocation the run made, every attempt included,
    each write in a transaction of its own, at ``records`` level. ``source``
    names where the run came from. Recording never fails the run.
    """
    attempts = max(1, max_attempts)
    reaction_id = reaction_id or str(uuid4())  # one id for the run, whatever the attempt count
    state = _RunState(started_at=datetime.now(UTC), clock=time.perf_counter())
    state.deadline_at = state.started_at + timedelta(seconds=timeout)
    outcome: Outcome = "ok"
    error: str | None = None
    if records != "off":
        await _record(store, reaction, reaction_id, input, source, records, state, "running", None)
    try:
        async with asyncio.timeout(timeout):
            return await _attempts(
                store,
                reaction,
                input,
                reaction_id,
                attempts,
                timeout,
                outbox_row_id,
                triggers,
                state,
            )
    except TimeoutError:
        outcome, error = "timeout", f"TimeoutError: budget of {timeout}s exhausted"
        logger.error(
            "run timed out reaction=%s reaction_id=%s timeout_s=%s",
            reaction.__name__,
            reaction_id,
            timeout,
        )
        raise
    except Exception as exc:
        outcome, error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metrics.observe_run(
            qualified_name(reaction),
            source,
            outcome,
            (time.perf_counter() - state.clock) * 1000,
            state.attempts,
        )
        for trace in state.traces:
            for span in trace.spans:
                metrics.observe_span(span.name, span.duration_ms)
        if records != "off":
            await _record(
                store, reaction, reaction_id, input, source, records, state, outcome, error
            )


class _RunState:
    """What the attempts leave behind for the record."""

    def __init__(self, *, started_at: datetime, clock: float) -> None:
        self.started_at = started_at
        self.clock = clock
        self.attempts = 0
        self.traces: list[Trace] = []
        self.session: Session | None = None
        self.root: Context | None = None
        self.deadline_at: datetime | None = None


async def _attempts(
    store: Store,
    reaction: type[Reaction],
    input: dict[str, Any],
    reaction_id: str,
    attempts: int,
    timeout: float,
    outbox_row_id: str | None,
    triggers: Sequence[Trigger],
    state: _RunState,
) -> Context:
    for attempt in range(1, attempts + 1):
        session = Session(store, triggers=triggers, default_timeout=timeout)
        ctx = Context(session, input, reaction_id=reaction_id)
        ctx.trace.attempt = attempt
        state.attempts = attempt
        state.traces.append(ctx.trace)
        state.session = session
        state.root = ctx
        try:
            async with store.run_scope():
                with ctx.trace.span("run", ctx.reaction_id, None):
                    await invoke(reaction, ctx)
                    with ctx.trace.span("flush", ctx.reaction_id, None):
                        await session.flush(reaction_id=ctx.reaction_id)
                    if outbox_row_id is not None:
                        await store.ack(outbox_row_id)
            return ctx
        except TimeoutError:
            raise  # the run's deadline, or a trigger's: never retried
        except Exception as exc:
            if attempt == attempts:
                logger.error(
                    "run gave up reaction=%s reaction_id=%s attempts=%d error=%s: %s",
                    reaction.__name__,
                    ctx.reaction_id,
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                raise
            delay = random.uniform(0, 0.05 * attempt)
            logger.warning(
                "run retrying reaction=%s reaction_id=%s attempt=%d delay_ms=%.0f error=%s: %s",
                reaction.__name__,
                ctx.reaction_id,
                attempt,
                delay * 1000,
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(delay)
        finally:
            ctx.trace.log(ctx.reaction_id)
    raise AssertionError("unreachable")


async def _record(
    store: Store,
    reaction: type[Reaction],
    reaction_id: str,
    input: dict[str, Any],
    source: str,
    level: RecordLevel,
    state: _RunState,
    outcome: Outcome,
    error: str | None,
) -> None:
    """One row per invocation, every attempt included, plus the seed's row
    with the commit-level facts. Inputs and vars at ``full`` level only."""
    session = state.session
    redact = secrets.default.redact_jsonable
    full = level == "full"

    def payload(value: Any) -> dict[str, Any] | None:
        return redact(safe_jsonable(value)) if full else None

    committed = session is not None and outcome == "ok"
    rows: list[RunRecord] = []
    for trace in state.traces:
        last = trace is state.traces[-1]
        for inv in trace.invocations:
            if inv.reaction_id == reaction_id:
                continue  # the seed's row is built below from the run's outcome
            rows.append(
                RunRecord(
                    id=inv.reaction_id,
                    root_id=reaction_id,
                    parent_reaction_id=inv.parent_reaction_id,
                    reaction=inv.reaction,
                    source=TRIGGER_SOURCE,
                    attempt=trace.attempt,
                    outcome="timeout" if inv.outcome == "timeout" else inv.outcome,  # type: ignore[arg-type]
                    error=secrets.default.redact(inv.error) if inv.error else None,
                    committed=committed and last,
                    started_at=inv.started_at,
                    duration_ms=inv.duration_ms,
                    flush_ms=None,
                    input=payload(inv.input),
                    vars=payload(inv.vars),
                    changes=_own_changes(inv.created, inv.deleted),
                    enqueued=list(inv.enqueued) if committed and last else [],
                )
            )
    root = state.root
    seed = (
        next(
            (i for i in reversed(state.traces[-1].invocations) if i.reaction_id == reaction_id),
            None,
        )
        if state.traces
        else None
    )
    flush = (
        next((sp for sp in state.traces[-1].spans if sp.name == "flush"), None)
        if state.traces
        else None
    )
    changes = (
        [
            {"entity": table_name(c.entity), "id": c.id, "kind": c.kind, "changed": list(c.changed)}
            for c in session.changes_made
        ]
        if committed and session is not None
        else []
    )
    rows.append(
        RunRecord(
            id=reaction_id,
            root_id=reaction_id,
            parent_reaction_id=None,
            reaction=qualified_name(reaction),
            source=source,
            attempt=state.attempts,
            outcome=outcome,
            error=secrets.default.redact(error) if error is not None else None,
            committed=committed,
            started_at=state.started_at,
            duration_ms=(time.perf_counter() - state.clock) * 1000,
            flush_ms=flush.duration_ms if flush is not None and committed else None,
            input=payload(input),
            vars=payload(seed.vars if seed is not None else (root.vars if root else {})),
            changes=changes,
            enqueued=list(session.enqueued) if committed and session is not None else [],
            deadline_at=state.deadline_at,
        )
    )
    try:
        await store.record_run(rows)
    except NotSupported:
        pass
    except Exception as exc:  # recording must never fail the run
        logger.error(
            "run record failed reaction_id=%s error=%s: %s", reaction_id, type(exc).__name__, exc
        )


def _own_changes(
    created: list[tuple[str, str]], deleted: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, pairs in (("created", created), ("deleted", deleted)):
        for table, row_id in pairs:
            rows.append({"entity": table, "id": row_id, "kind": kind, "changed": []})
    return rows
