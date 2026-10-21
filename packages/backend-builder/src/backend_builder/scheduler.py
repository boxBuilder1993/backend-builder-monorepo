"""The scheduler: keeps each schedule's next occurrence in the outbox.

A separate process from the worker. On every tick it reads each schedule's
``last_scheduled_for`` from the ``_schedule`` system table and, if that time
has passed, enqueues the next occurrence after now and advances the record,
all in one transaction under a row lock, so several schedulers never
double-seed and a restart never re-seeds an occurrence already delivered.
Occurrences that fall while no scheduler is running are skipped, as cron
does. The worker is a pure consumer and knows nothing about schedules.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from backend_builder.app import App
from backend_builder.log import get_logger
from backend_builder.outbox import now
from backend_builder.settings import Settings
from backend_builder.store import NotSupported, Store

logger = get_logger(__name__)


class Scheduler:
    def __init__(self, app: App, store: Store, settings: Settings | None = None) -> None:
        self.app = app
        self.store = store
        self.settings = settings if settings is not None else Settings.from_env()
        self._last_purge: float | None = None

    async def tick(self) -> int:
        """Seed every schedule whose last seeded occurrence has passed.
        Returns how many rows were enqueued."""
        seeded = 0
        moment = now()
        for schedule in self.app.schedules:
            if await self.store.seed_schedule(schedule, moment):
                seeded += 1
        if seeded:
            logger.info("scheduler seeded app=%s rows=%d", self.app.name, seeded)
        return seeded

    async def run_forever(self) -> None:
        logger.info(
            "scheduler started app=%s schedules=%d tick_s=%s",
            self.app.name,
            len(self.app.schedules),
            self.settings.scheduler_tick_seconds,
        )
        while True:
            await self.tick()
            await self.purge_if_due()
            await asyncio.sleep(self.settings.scheduler_tick_seconds)

    async def purge_if_due(self, *, every_seconds: float = 3600.0) -> int:
        """Delete run records older than the retention window, at most once
        per ``every_seconds``. Returns how many were deleted."""
        loop_time = asyncio.get_running_loop().time()
        if self._last_purge is not None and loop_time - self._last_purge < every_seconds:
            return 0
        self._last_purge = loop_time
        cutoff = now() - timedelta(days=self.settings.run_retention_days)
        try:
            purged = await self.store.purge_runs(cutoff)
        except NotSupported:
            return 0
        if purged:
            logger.info("scheduler purged run records app=%s rows=%d", self.app.name, purged)
        return purged
