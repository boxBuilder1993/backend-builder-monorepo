"""Prometheus metrics, and the small server every process exposes them on.

Metrics live in this process's memory and are updated where things happen:
a run ends, a row is delivered, a request completes. ``/metrics`` renders
them as text for a scraper to poll; nothing is pushed. Every process serves
``/metrics`` and ``/health`` on ``BACKEND_BUILDER_METRICS_PORT``, separate
from the API's own port, so metrics never sit on a public listener.

Labels are bounded: reaction names, sources, outcomes, routes, statuses.
Never a reaction id.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from backend_builder.log import get_logger

logger = get_logger(__name__)

registry = CollectorRegistry()

_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300)

runs_total = Counter(
    "bb_runs_total",
    "Runs by reaction, source and outcome",
    ["reaction", "source", "outcome"],
    registry=registry,
)
run_duration_seconds = Histogram(
    "bb_run_duration_seconds",
    "Whole-run duration",
    ["reaction"],
    registry=registry,
    buckets=_DURATION_BUCKETS,
)
run_attempts = Histogram(
    "bb_run_attempts",
    "Attempts a run took",
    ["reaction"],
    registry=registry,
    buckets=(1, 2, 3, 5, 10),
)
span_duration_seconds = Histogram(
    "bb_span_duration_seconds",
    "Duration of one span",
    ["name"],
    registry=registry,
    buckets=_DURATION_BUCKETS,
)
http_requests_total = Counter(
    "bb_http_requests_total", "HTTP requests", ["route", "method", "status"], registry=registry
)
http_request_duration_seconds = Histogram(
    "bb_http_request_duration_seconds",
    "HTTP request duration",
    ["route", "method"],
    registry=registry,
    buckets=_DURATION_BUCKETS,
)
outbox_deliveries_total = Counter(
    "bb_outbox_deliveries_total",
    "Outbox deliveries by reaction and outcome",
    ["reaction", "outcome"],
    registry=registry,
)
outbox_pending = Gauge("bb_outbox_pending", "Outbox rows waiting", registry=registry)
outbox_claimed = Gauge("bb_outbox_claimed", "Outbox rows being delivered", registry=registry)
outbox_dead = Gauge("bb_outbox_dead", "Outbox rows marked dead", registry=registry)
outbox_oldest_pending_seconds = Gauge(
    "bb_outbox_oldest_pending_seconds", "Age of the oldest waiting outbox row", registry=registry
)
schedule_seeds_total = Counter(
    "bb_schedule_seeds_total", "Schedule occurrences seeded", registry=registry
)
run_records_purged_total = Counter(
    "bb_run_records_purged_total", "Run records purged", registry=registry
)
runs_lost_total = Counter(
    "bb_runs_lost_total",
    "Runs whose process stopped reporting before the deadline",
    registry=registry,
)


def render() -> bytes:
    return generate_latest(registry)


def app(ping: Callable[[], Awaitable[bool]] | None = None) -> Starlette:
    """The metrics server's app: ``/metrics`` and ``/health``."""

    async def metrics(request: Request) -> Response:
        return Response(render(), media_type=CONTENT_TYPE_LATEST)

    async def health(request: Request) -> Response:
        ok = await ping() if ping is not None else True
        return JSONResponse(
            {"status": "ok" if ok else "unavailable"}, status_code=200 if ok else 503
        )

    return Starlette(routes=[Route("/metrics", metrics), Route("/health", health)])


async def serve(
    port: int, ping: Callable[[], Awaitable[bool]] | None = None, host: str = "0.0.0.0"
) -> None:
    """Serve the metrics app until cancelled. Runs alongside a process's
    main loop as a task."""
    import uvicorn

    config = uvicorn.Config(app(ping), host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    logger.info("metrics serving host=%s port=%d", host, port)
    await server.serve()


def start(port: int, ping: Callable[[], Awaitable[bool]] | None = None) -> asyncio.Task[None]:
    """Start the metrics server as a background task on the running loop."""
    return asyncio.create_task(serve(port, ping), name="metrics")


def observe_span(name: str, duration_ms: float) -> None:
    span_duration_seconds.labels(name=name).observe(duration_ms / 1000)


def observe_run(
    reaction: str, source: str, outcome: str, duration_ms: float, attempts: int
) -> None:
    runs_total.labels(reaction=reaction, source=source, outcome=outcome).inc()
    run_duration_seconds.labels(reaction=reaction).observe(duration_ms / 1000)
    run_attempts.labels(reaction=reaction).observe(attempts)


def observe_outbox(stats: Any) -> None:
    """Refresh the outbox gauges from an ``OutboxStats``."""
    from datetime import UTC, datetime

    outbox_pending.set(stats.pending)
    outbox_claimed.set(stats.claimed)
    outbox_dead.set(stats.dead)
    oldest = stats.oldest_pending
    outbox_oldest_pending_seconds.set(
        max(0.0, (datetime.now(UTC) - oldest).total_seconds()) if oldest is not None else 0.0
    )
