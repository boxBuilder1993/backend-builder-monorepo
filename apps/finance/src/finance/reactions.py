"""The three reactions: NightlySync fans out, SyncLeg runs one leg and writes rows, Publish uploads.

Files are written by the legs and the store, outside the transaction, so the order is always file
first, row second, and every file write is idempotent; the rows are the only thing the transaction
covers. Reactions never do HTTP or Parquet themselves.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

from backend_builder import Context, Reaction

from finance import store
from finance.entities import DataFile
from finance.settings import DATA_DIR, HF_REPO, RECENT_DAYS
from finance.sources import BUDGETS, LEGS, SOURCES, Ledger, LegResult


def nodata_path(source: str, day: date) -> str:
    return f"{source}/.nodata/{day.isoformat()}"


class NightlySync(Reaction):
    """Scheduled 20:00 and 22:00 IST. Defers one SyncLeg per leg, each with its own budget.
    Input: {"scheduled_for": ...}. Writes nothing itself."""

    async def calculate(self, ctx: Context) -> None:
        for leg, budget in BUDGETS.items():
            await ctx.trigger(SyncLeg, {"leg": leg}, deferred=True, timeout=budget)
        ctx.save("deferred", list(BUDGETS))


class SyncLeg(Reaction):
    """Input: {"leg": "reference" | "nse" | "us" | "kite"}. Runs the leg's function in a thread,
    then writes one DataFile row per file written and per day found empty. Idempotent: a path
    already in the ledger is updated, never duplicated."""

    async def calculate(self, ctx: Context) -> None:
        leg = ctx.input["leg"]
        if leg not in LEGS:
            raise LookupError(f"unknown leg {leg!r}; known: {sorted(LEGS)}")
        today = date.today()
        ledger = await self._ledger(ctx, SOURCES[leg], today)
        result: LegResult = await asyncio.to_thread(LEGS[leg], DATA_DIR, today, ledger)
        ctx.save("leg", leg)
        ctx.save("implemented", result.implemented)
        if not result.implemented:
            return
        now = datetime.now(UTC)
        for w in result.written:
            row = await self._row(ctx, w.path)
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
            row = await self._row(ctx, path)
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
    async def _row(ctx: Context, path: str) -> DataFile | None:
        rows = await ctx.query(DataFile, "select * from data_file where path = :path", path=path)
        return rows[0] if rows else None

    @staticmethod
    async def _ledger(ctx: Context, sources: tuple[str, ...], today: date) -> Ledger:
        ok: set[tuple[str, date]] = set()
        no_data: set[tuple[str, date]] = set()
        recent = today - timedelta(days=RECENT_DAYS)
        for source in sources:
            rows = await ctx.query(
                DataFile,
                "select * from data_file where source = :source and day is not null",
                source=source,
            )
            for r in rows:
                assert r.day is not None
                if r.status == "ok":
                    ok.add((r.source, r.day))
                elif r.day < recent:  # recent no_data days are asked again
                    no_data.add((r.source, r.day))
        return Ledger(ok=frozenset(ok), no_data=frozenset(no_data))


class Publish(Reaction):
    """Scheduled every 30 minutes. Uploads every ok DataFile with published_at null to the store in
    chunked commits and stamps published_at. The token is SECRET_HF_TOKEN. Nothing to upload is the
    common case and costs one query."""

    async def calculate(self, ctx: Context) -> None:
        rows = await ctx.query(
            DataFile,
            "select * from data_file where status = 'ok' and published_at is null order by path",
        )
        ctx.save("files", len(rows))
        if not rows:
            return
        token = ctx.secret("hf_token")
        paths = [r.path for r in rows]
        commits = await asyncio.to_thread(
            store.upload, DATA_DIR, paths, HF_REPO, token, f"data: sync {date.today()}"
        )
        now = datetime.now(UTC)
        for r in rows:
            r.published_at = now
        ctx.save("commits", [c.sha for c in commits])
        ctx.save("bytes", sum(c.bytes for c in commits))
