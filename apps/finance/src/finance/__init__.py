"""finance: the market-data poller on Backend Builder. See docs/finance-app.md."""

from backend_builder import App

from finance.triggers import triggers

# One attempt per run: a leg that fails is released to the outbox with backoff, not re-run inside
# its budget. The 600 s default covers NightlySync and Publish; legs set their own on the trigger.
app = App(package="finance", triggers=triggers, max_attempts=1, timeout=600)

__all__ = ["app"]
