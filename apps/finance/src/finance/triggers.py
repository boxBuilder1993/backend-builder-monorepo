"""When things run. Cron is UTC: 14:30 and 16:30 UTC are 20:00 and 22:00 IST (no DST in India).

A scheduled row carries its own budget: one without a timeout gets the outbox default of 60 s, not
the App's, and the night timed out at 60 s on 2026-10-23 for that reason. The night's budget is the
App's four hours, stated here."""

from backend_builder import OnChange, OnSchedule

from finance.ml.entities import Model
from finance.ml.reactions import CloseFitRun
from finance.reactions import NightlySync

NIGHT_TIMEOUT = 4 * 3600

triggers = [
    OnSchedule("30 14 * * *", NightlySync, timeout=NIGHT_TIMEOUT),
    OnSchedule("30 16 * * *", NightlySync, timeout=NIGHT_TIMEOUT),
    OnChange(Model, ["status"], CloseFitRun),
]
