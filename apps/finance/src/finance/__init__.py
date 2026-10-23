"""finance: the market-data poller on Backend Builder. See docs/finance-app.md."""

from backend_builder import App

from finance.ml.api import endpoints
from finance.triggers import triggers

# One attempt per run and a budget of four hours for the night: the legs catch their own failures
# as outcomes, and a night that still fails is caught up by the next scheduled run, not by a retry.
app = App(
    package="finance", endpoints=endpoints, triggers=triggers, max_attempts=1, timeout=4 * 3600
)

__all__ = ["app"]
