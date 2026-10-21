"""Run entry point: the seed of a run."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from backend_builder import metrics, secrets
from backend_builder.context import Context
from backend_builder.log import get_logger
from backend_builder.reaction import Reaction, invoke, qualified_name
from backend_builder.records import Outcome, RecordLevel, RunRecord, SpanRecord, safe_jsonable
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

    After the run, whatever its outcome, a record of it is written to the
    store in a transaction of its own, at ``records`` level: what came in,
    what was saved, what was written, what was deferred, and every attempt's
    spans. ``source`` names where the run came from. Recording never fails
    the run.
    """
    attempts = max(1, max_attempts)
    reaction_id = reaction_id or str(uuid4())  # one id for the run, whatever the attempt count
    state = _RunState(started_at=datetime.now(UTC), clock=time.perf_counter())
    outcome: Outcome = "ok"
    error: str | None = None
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
        state.attempts = attempt
        state.traces.append(ctx.trace)
        state.session = session
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
    session = state.session
    redact = secrets.default.redact_jsonable
    spans = [
        SpanRecord(
            attempt=number,
            name=s.name,
            reaction_id=s.reaction_id,
            parent_reaction_id=s.parent_reaction_id,
            started_at=s.started_at,
            duration_ms=s.duration_ms,
            ok=s.ok,
        )
        for number, trace in enumerate(state.traces, start=1)
        for s in trace.spans
    ]
    committed = session is not None and outcome == "ok"
    changes = (
        [
            {"entity": table_name(c.entity), "id": c.id, "kind": c.kind, "changed": list(c.changed)}
            for c in session.changes_made
        ]
        if committed and session is not None
        else []
    )
    record = RunRecord(
        id=reaction_id,
        reaction=qualified_name(reaction),
        source=source,
        outcome=outcome,
        error=secrets.default.redact(error) if error is not None else None,
        attempts=state.attempts,
        started_at=state.started_at,
        duration_ms=(time.perf_counter() - state.clock) * 1000,
        input=redact(safe_jsonable(input)) if level == "full" else None,
        vars=redact(safe_jsonable(session.vars))
        if level == "full" and session is not None
        else None,
        changes=changes,
        enqueued=list(session.enqueued) if committed and session is not None else [],
        spans=spans,
    )
    try:
        await store.record_run(record)
    except NotSupported:
        pass
    except Exception as exc:  # recording must never fail the run
        logger.error(
            "run record failed reaction_id=%s error=%s: %s", reaction_id, type(exc).__name__, exc
        )
