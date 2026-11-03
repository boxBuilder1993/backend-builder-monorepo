"""The poller's reactions, NightlySync running the legs in order and then Publish in one run, and
PullStore, which brings the local mirror of the store to the store's current revision.

Files are written by the legs and the store, outside the transaction, so the order is always file
first, row second, and every file write is idempotent; the rows are the only thing the transaction
covers. Reactions never do HTTP or Parquet themselves.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from backend_builder import Context, Reaction

from finance import mirror, settings, store
from finance.entities import DataFile, StoreSync
from finance.settings import HF_REPO, RECENT_DAYS
from finance.sources import BUDGETS, LEGS, SOURCES, Ledger, LegResult


def nodata_path(source: str, day: date) -> str:
    return f"{source}/.nodata/{day.isoformat()}"


def data_root() -> Path:
    """Where the data is on this host: the store's mirror. Patched in tests."""
    return mirror.mirror_dir(HF_REPO)


class NightlySync(Reaction):
    """Scheduled 20:00 and 22:00 IST. Triggers one SyncLeg per leg, in order, then Publish: one run,
    one transaction, the night. A leg's failure is an outcome saved under its name, never an
    exception, so the other legs run and the publish happens. Input: {"scheduled_for": ...}.

    Each child's summary is read from the context ``trigger`` returns; vars belong to one
    invocation and nothing crosses between reactions except through input and that context."""

    LEG_KEYS = ("implemented", "written", "no_data", "error", "notes")
    PUBLISH_KEYS = ("files", "commits", "bytes", "error")

    async def calculate(self, ctx: Context) -> None:
        legs: dict[str, Any] = {}
        for leg, budget in BUDGETS.items():
            child = await ctx.trigger(SyncLeg, {"leg": leg}, timeout=budget)
            assert child is not None
            legs[leg] = {k: child.vars.get(k) for k in self.LEG_KEYS}
        ctx.save("legs", legs)
        published = await ctx.trigger(Publish, {})
        assert published is not None
        ctx.save("published", {k: published.vars.get(k) for k in self.PUBLISH_KEYS})


class SyncLeg(Reaction):
    """Input: {"leg": "reference" | "nse" | "us" | "kite"}. Runs the leg's function in a thread,
    then writes one DataFile row per file written and per day found empty. Idempotent: a path
    already in the ledger is updated, never duplicated."""

    async def calculate(self, ctx: Context) -> None:
        leg = ctx.input["leg"]
        if leg not in LEGS:
            raise LookupError(f"unknown leg {leg!r}; known: {sorted(LEGS)}")
        today = date.today()
        ledger, rows = await self._ledger(ctx, SOURCES[leg], today)
        ctx.save("leg", leg)
        try:
            result: LegResult = await asyncio.to_thread(LEGS[leg], data_root(), today, ledger)
        except Exception as exc:  # a source that is down is an outcome, not the night's failure
            ctx.save("error", f"{type(exc).__name__}: {exc}")
            ctx.save("implemented", True)
            return
        ctx.save("implemented", result.implemented)
        if not result.implemented:
            return
        now = datetime.now(UTC)
        for w in result.written:
            row = rows.get(w.path)
            if row is None:
                ctx.create(
                    DataFile,
                    path=w.path,
                    kind=w.kind,
                    source=w.source,
                    day=w.day,
                    status="ok",
                    bytes=w.bytes,
                    sha256=w.sha256,
                    written_at=now,
                    checked_at=now,
                )
            else:
                changed = row.bytes != w.bytes or row.sha256 != w.sha256 or row.status != "ok"
                row.status, row.bytes, row.sha256, row.checked_at = "ok", w.bytes, w.sha256, now
                if changed:  # a rewritten reference file or a day that gained data goes out again
                    row.written_at, row.published_at = now, None
        for nd in result.no_data:
            path = nodata_path(nd.source, nd.day)
            row = rows.get(path)
            if row is None:
                ctx.create(
                    DataFile,
                    path=path,
                    kind="raw",
                    source=nd.source,
                    day=nd.day,
                    status="no_data",
                    checked_at=now,
                )
            else:
                row.checked_at = now
        ctx.save("written", len(result.written))
        ctx.save("no_data", len(result.no_data))
        ctx.save("notes", result.notes)

    @staticmethod
    async def _ledger(
        ctx: Context, sources: tuple[str, ...], today: date
    ) -> tuple[Ledger, dict[str, DataFile]]:
        """The leg's view of the ledger, and every row of its sources by path. Loaded in one query
        per source before anything is written: a query flushes the session's pending changes first,
        so a lookup per file inside the write loop would make the run quadratic."""
        ok: set[tuple[str, date]] = set()
        no_data: set[tuple[str, date]] = set()
        by_path: dict[str, DataFile] = {}
        recent = today - timedelta(days=RECENT_DAYS)
        for source in sources:
            rows = await ctx.query(
                DataFile, "select * from data_file where source = :source", source=source
            )
            for r in rows:
                by_path[r.path] = r
                if r.day is None:
                    continue
                if r.status == "ok":
                    ok.add((r.source, r.day))
                elif r.day < recent:  # recent no_data days are asked again
                    no_data.add((r.source, r.day))
        return Ledger(ok=frozenset(ok), no_data=frozenset(no_data)), by_path


class Publish(Reaction):
    """Uploads every ok DataFile with published_at null to the store in chunked commits and stamps
    published_at. Triggered by NightlySync after the legs; runnable alone from the CLI. The token is
    SECRET_HF_TOKEN. A failed upload is an outcome: the rows stay unstamped and go out next time.
    FINANCE_PUBLISH=off leaves them unstamped on purpose, for the shadow phase."""

    async def calculate(self, ctx: Context) -> None:
        rows = await ctx.query(
            DataFile,
            "select * from data_file where status = 'ok' and published_at is null order by path",
        )
        ctx.save("files", len(rows))
        if not rows:
            return
        if not settings.PUBLISH:
            ctx.save("disabled", True)
            return
        token = ctx.secret("hf_token")
        paths = [r.path for r in rows]
        try:
            commits = await asyncio.to_thread(
                store.upload, data_root(), paths, HF_REPO, token, f"data: sync {date.today()}"
            )
        except Exception as exc:  # noqa: BLE001 - the store being down must not undo the legs' rows
            ctx.save("error", f"{type(exc).__name__}: {exc}")
            return
        now = datetime.now(UTC)
        for r in rows:
            r.published_at = now
        ctx.save("commits", [c.sha for c in commits])
        ctx.save("bytes", sum(c.bytes for c in commits))


class PullStore(Reaction):
    """Bring the mirror to the store's current revision, or find it already there. Input:
    {"patterns": [...]} (default parquet/**; the ML never needs raw). One API call when nothing
    moved; one tree listing and a download of only the files whose blob differs when it did. Every
    outcome is a StoreSync row: current, pulled, or failed with the error, so the next read tries
    again. Saves data_revision too: the market data's content hash, unmoved by the pipeline's own
    commits (models, inference tables), for callers that cache what they derive from the data."""

    DEFAULT_PATTERNS = ["parquet/**"]

    async def calculate(self, ctx: Context) -> None:
        patterns: list[str] = list(ctx.input.get("patterns") or self.DEFAULT_PATTERNS)
        token = ctx.secret("hf_token")
        root = data_root()
        last = await ctx.query(
            StoreSync,
            "select * from store_sync where repo = :repo and status in ('current', 'pulled') "
            "order by created_at desc limit 1",
            repo=HF_REPO,
        )
        known = last[0] if last else None
        try:
            revision = await asyncio.to_thread(mirror.store_revision, HF_REPO, token)
            covered = known is not None and set(patterns) <= set(known.patterns)
            if known is not None and known.revision == revision and covered and root.is_dir():
                ctx.create(
                    StoreSync,
                    repo=HF_REPO,
                    revision=revision,
                    patterns=patterns,
                    data_revision=known.data_revision,
                )
                ctx.save("status", "current")
                ctx.save("revision", revision)
                ctx.save("data_revision", known.data_revision)
                return
            pulled = await asyncio.to_thread(mirror.pull, HF_REPO, token, revision, patterns, root)
        except Exception as exc:  # noqa: BLE001 - the store being down is an outcome
            ctx.create(
                StoreSync,
                repo=HF_REPO,
                revision=known.revision if known is not None else "",
                patterns=patterns,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            ctx.save("status", "failed")
            ctx.save("error", f"{type(exc).__name__}: {exc}")
            return
        ctx.create(
            StoreSync,
            repo=HF_REPO,
            revision=revision,
            patterns=patterns,
            data_revision=pulled.data_revision,
            files=pulled.files,
            bytes=pulled.bytes,
            seconds=round(pulled.seconds, 1),
            status="pulled",
        )
        ctx.save("status", "pulled")
        ctx.save("revision", revision)
        ctx.save("data_revision", pulled.data_revision)
        ctx.save("files", pulled.files)
        ctx.save("bytes", pulled.bytes)
