"""finance: the market-data poller on Backend Builder. See docs/finance-app.md."""

from backend_builder import App

from finance.ml.api import endpoints
from finance.triggers import triggers

# A budget of four hours for the night. The legs, the fits and the uploads catch their own failures
# as outcomes, so the runtime's retries (the default three) only ever replay infrastructure faults:
# a serialization conflict when two models of one sweep finish together, a dropped connection.
app = App(package="finance", endpoints=endpoints, triggers=triggers, timeout=4 * 3600)

__all__ = ["app"]
