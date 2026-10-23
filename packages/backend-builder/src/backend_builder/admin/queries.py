"""Read models for the console, as SQL over the system tables."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from backend_builder.store import Store


async def overview(store: Store, since_hours: float = 24) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(hours=since_hours)
    outcomes = await store.query_raw(
        "select outcome, count(*) as n from _run "
        "where id = root_id and started_at >= :since group by outcome",
        since=since,
    )
    stats = await store.outbox_stats()
    running = await runs(store, outcome="running", limit=100)
    return {
        "outcomes": {str(r["outcome"]): int(r["n"]) for r in outcomes},
        "running": running,
        "outbox": stats,
        "since_hours": since_hours,
    }


async def runs(
    store: Store,
    *,
    reaction: str | None = None,
    outcome: str | None = None,
    source: str | None = None,
    limit: int = 50,
    roots_only: bool = True,
) -> list[dict[str, Any]]:
    """Recent rows, newest first. Runs (seed rows) by default; with
    ``roots_only`` false, every invocation, children included."""
    clauses: list[str] = ["id = root_id"] if roots_only else []
    params: dict[str, Any] = {"limit": limit}
    if reaction:
        clauses.append("reaction = :reaction")
        params["reaction"] = reaction
    if outcome:
        clauses.append("outcome = :outcome")
        params["outcome"] = outcome
    if source:
        clauses.append("source like :source")
        params["source"] = f"{source}%"
    where = ("where " + " and ".join(clauses)) if clauses else ""
    return await store.query_raw(
        f"select id, root_id, reaction, source, outcome, error, attempt, started_at, duration_ms "
        f"from _run {where} order by started_at desc limit :limit",
        **params,
    )


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


async def run_detail(store: Store, run_id: str) -> dict[str, Any] | None:
    """One invocation's row plus its subtree: for a seed, every invocation of
    the run grouped by attempt; for a child, its descendants in its attempt.
    ``outbox_rows`` are the outbox rows the subtree deferred."""
    if not _is_uuid(run_id):
        return None
    rows = await store.query_raw("select * from _run where id = :id", id=run_id)
    if not rows:
        return None
    record = rows[0]
    family = await store.query_raw(
        "select id, root_id, parent_reaction_id, reaction, attempt, outcome, error, started_at, "
        "duration_ms, enqueued from _run where root_id = :root order by attempt, started_at",
        root=record["root_id"],
    )
    subtree = _subtree(family, record)
    record["attempts_tree"] = _trees(subtree)
    record["is_root"] = record["id"] == record["root_id"]
    ids: list[str] = []
    for r in subtree:
        enqueued: list[Any] = list(r.get("enqueued") or [])
        for row_id in enqueued:
            if str(row_id) not in ids:
                ids.append(str(row_id))
    record["enqueued_all"] = ids
    record["outbox_rows"] = (
        await store.query_raw(
            "select id, reaction, attempts, available_at, claimed_at, dead_at, scheduled_for "
            "from _outbox where id = any(:ids)",
            ids=ids,
        )
        if ids
        else []
    )
    return record


def _subtree(family: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    """``record`` and its descendants; the whole family for the seed."""
    if record["id"] == record["root_id"]:
        return family
    keep = {record["id"]}
    changed = True
    while changed:
        changed = False
        for r in family:
            if r["id"] not in keep and r["parent_reaction_id"] in keep:
                keep.add(r["id"])
                changed = True
    return [r for r in family if r["id"] in keep]


def _trees(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by attempt and order each attempt's rows parent-first, depth
    annotated, so a template can indent them. Rows are in start order, and
    children always start after their parent."""
    by_attempt: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_attempt.setdefault(int(row["attempt"]), []).append(row)
    trees: list[dict[str, Any]] = []
    for attempt, items in sorted(by_attempt.items()):
        parents = {r["id"]: r["parent_reaction_id"] for r in items}
        for r in items:
            r["depth"] = _depth(r, parents)
        trees.append({"attempt": attempt, "rows": items})
    return trees


def _depth(row: dict[str, Any], parents: dict[Any, Any]) -> int:
    d, parent = 0, row["parent_reaction_id"]
    while parent is not None and parent in parents:
        d, parent = d + 1, parents[parent]
    return d


async def reaction_stats(store: Store, reaction: str, since_hours: float = 24) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(hours=since_hours)
    rows = await store.query_raw(
        "select outcome, count(*) as n, "
        "percentile_cont(0.5) within group (order by duration_ms) as p50, "
        "percentile_cont(0.95) within group (order by duration_ms) as p95 "
        "from _run where reaction = :reaction and started_at >= :since group by outcome",
        reaction=reaction,
        since=since,
    )
    pending = await store.query_raw(
        "select count(*) as n from _outbox where reaction = :reaction and dead_at is null",
        reaction=reaction,
    )
    return {
        "by_outcome": rows,
        "pending": int(pending[0]["n"]) if pending else 0,
        "since_hours": since_hours,
    }


async def outbox_rows(store: Store, state: str, limit: int = 100) -> list[dict[str, Any]]:
    conditions = {
        "pending": "dead_at is null and claimed_at is null",
        "claimed": "dead_at is null and claimed_at is not null",
        "dead": "dead_at is not null",
    }
    return await store.query_raw(
        f"select id, reaction, input, attempts, last_error, created_at, available_at, "
        f"claimed_by, claimed_at, dead_at, scheduled_for, timeout_seconds "
        f"from _outbox where {conditions[state]} order by available_at limit :limit",
        limit=limit,
    )


async def outbox_row(store: Store, row_id: str) -> dict[str, Any] | None:
    if not _is_uuid(row_id):
        return None
    rows = await store.query_raw("select * from _outbox where id = :id", id=row_id)
    return rows[0] if rows else None


async def requeue(store: Store, row_id: str) -> bool:
    if not _is_uuid(row_id):
        return False
    rows = await store.query_raw(
        "update _outbox set dead_at = null, attempts = 0, claimed_by = null, claimed_at = null, "
        "available_at = now() where id = :id returning id",
        id=row_id,
    )
    return bool(rows)


async def discard(store: Store, row_id: str) -> bool:
    if not _is_uuid(row_id):
        return False
    rows = await store.query_raw("delete from _outbox where id = :id returning id", id=row_id)
    return bool(rows)


async def schedules(store: Store) -> list[dict[str, Any]]:
    return await store.query_raw(
        "select s.reaction, s.cron, s.last_scheduled_for, s.updated_at, "
        "(select min(available_at) from _outbox o where o.reaction = s.reaction "
        " and o.scheduled_for is not null and o.dead_at is null) as next_due "
        "from _schedule s order by s.reaction, s.cron"
    )
