"""When things run. Cron is UTC: 14:30 and 16:30 UTC are 20:00 and 22:00 IST (no DST in India)."""

from backend_builder import OnSchedule

from finance.reactions import NightlySync, Publish

triggers = [
    OnSchedule("30 14 * * *", NightlySync, timeout=60),
    OnSchedule("30 16 * * *", NightlySync, timeout=60),
    OnSchedule("*/30 * * * *", Publish, timeout=3600),
]
