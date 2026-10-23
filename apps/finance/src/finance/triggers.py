"""When things run. Cron is UTC: 14:30 and 16:30 UTC are 20:00 and 22:00 IST (no DST in India)."""

from backend_builder import OnChange, OnSchedule

from finance.ml.entities import Model
from finance.ml.reactions import CloseFitRun
from finance.reactions import NightlySync

triggers = [
    OnSchedule("30 14 * * *", NightlySync),
    OnSchedule("30 16 * * *", NightlySync),
    OnChange(Model, ["status"], CloseFitRun),
]
