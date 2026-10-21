"""The worker: claims outbox rows and runs them.

Any number of workers may run against one database; ``FOR UPDATE SKIP
LOCKED`` divides the rows between them without coordination. Each claimed
row is delivered as an ordinary run with the app's retry policy, and the row
is deleted inside that run's transaction. A failed delivery is released
with backoff; at the outbox attempt limit the row is marked dead.
"""

from __future__ import annotations

import asyncio
import os
import socket
from uuid import uuid4

from backend_builder.app import App
from backend_builder.engine import run
from backend_builder.log import get_logger
from backend_builder.outbox import OutboxRow, now
from backend_builder.reaction import qualified_name
from backend_builder.settings import Settings
from backend_builder.store import Store

logger = get_logger(__name__)


class Worker:
    def __init__(self, app: App, store: Store, settings: Settings | None = None) -> None:
        self.app = app
        self.store = store
        self.settings = settings if settings is not None else Settings.from_env()
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:6]}"

    async def seed_schedules(self) -> None:
        """Ensure each schedule has its next occurrence enqueued. Idempotent:
        the outbox's unique index on (reaction, scheduled_for) makes a second
        seed of the same occurrence a no-op, so any number of workers may
        call this at any time."""
        moment = now()
        entries = [s.entry(s.next_after(moment)) for s in self.app.schedules]
        if entries:
            await self.store.enqueue(entries)

    async def run_once(self) -> int:
        """Seed schedules, claim one batch and deliver it. Returns how many
        rows were claimed."""
        await self.seed_schedules()
        rows = await self.store.claim(
            self.settings.outbox_batch, self.worker_id, self.settings.outbox_lease_seconds
        )
        if not rows:
            return 0
        logger.info("worker claimed worker_id=%s rows=%d", self.worker_id, len(rows))
        semaphore = asyncio.Semaphore(max(1, self.settings.outbox_concurrency))

        async def deliver(row: OutboxRow) -> None:
            async with semaphore:
                await self._deliver(row)

        await asyncio.gather(*(deliver(row) for row in rows))
        return len(rows)

    async def run_forever(self) -> None:
        """Poll until cancelled. Sleeps the poll interval when the queue is empty."""
        logger.info(
            "worker started worker_id=%s app=%s poll_s=%s batch=%d concurrency=%d",
            self.worker_id,
            self.app.name,
            self.settings.outbox_poll_seconds,
            self.settings.outbox_batch,
            self.settings.outbox_concurrency,
        )
        while True:
            claimed = await self.run_once()
            if claimed == 0:
                await asyncio.sleep(self.settings.outbox_poll_seconds)

    async def _enqueue_next(self, row: OutboxRow) -> None:
        """After a scheduled delivery, enqueue the occurrence after it. The
        unique index makes this safe if seeding already did."""
        assert row.scheduled_for is not None
        for schedule in self.app.schedules:
            if qualified_name(schedule.reaction) == row.reaction:
                await self.store.enqueue([schedule.entry(schedule.next_after(row.scheduled_for))])

    async def _deliver(self, row: OutboxRow) -> None:
        try:
            reaction = self.app.resolve(row.reaction)
        except LookupError as exc:
            await self.store.release(row.id, str(exc), self.settings.outbox_max_attempts)
            logger.error("worker unknown reaction row=%s reaction=%s", row.id, row.reaction)
            return
        try:
            await run(
                self.store,
                reaction,
                row.input,
                max_attempts=self.app.max_attempts,
                outbox_row_id=row.id,
                triggers=self.app.triggers,
            )
            logger.info("worker delivered row=%s reaction=%s", row.id, reaction.__name__)
            if row.scheduled_for is not None:
                await self._enqueue_next(row)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            await self.store.release(row.id, error, self.settings.outbox_max_attempts)
            logger.error(
                "worker delivery failed row=%s reaction=%s attempt=%d error=%s",
                row.id,
                reaction.__name__,
                row.attempts + 1,
                error,
            )
