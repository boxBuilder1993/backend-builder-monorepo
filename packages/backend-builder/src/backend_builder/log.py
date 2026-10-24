"""Logging: one line per event, stamped with the run it belongs to.

The package logs under the ``backend_builder`` hierarchy. ``configure``
attaches one handler to the root logger, so an app's own
``logging.getLogger(__name__)`` lines travel the same way as the runtime's.

Every record emitted inside a run carries ``root_id``, ``reaction_id``,
``reaction`` and ``source``: ``invoke`` sets a context variable on entry and
restores it on exit, and a filter copies the fields onto each record. Lines
from a child invocation carry the child's ``reaction_id`` and the run's
``root_id``. Processes add ``app`` and ``process`` through ``identify``.

Two formats. ``human`` is one readable line to stderr, for a terminal.
``json`` is one JSON object per line to stdout, for a collector; see
``docs/logs.md``. Both redact every secret the process has read.

Conventions:
  INFO   one line per lifecycle event, identifiers only: reaction name,
         reaction id, parent id, counts, durations. Never payloads.
  DEBUG  payloads: the input a reaction received, the rows a flush wrote.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TextIO

ROOT = "backend_builder"
Format = Literal["human", "json"]

logging.getLogger(ROOT).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """A logger under the package hierarchy. ``name`` is usually ``__name__``."""
    if name == ROOT or name.startswith(ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT}.{name}")


# --- what a line carries ---------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RunFields:
    """The run an invocation belongs to, stamped on every record it emits."""

    root_id: str
    reaction_id: str
    reaction: str
    source: str


_current: ContextVar[RunFields | None] = ContextVar("backend_builder_log_run", default=None)
FIELD_NAMES = ("root_id", "reaction_id", "reaction", "source")

_identity: dict[str, str] = {}
"""``app`` and ``process`` for this process; constant once set."""


def enter(fields: RunFields) -> Token[RunFields | None]:
    """Make ``fields`` current. ``invoke`` calls this; ``leave`` undoes it."""
    return _current.set(fields)


def leave(token: Token[RunFields | None]) -> None:
    _current.reset(token)


def current() -> RunFields | None:
    return _current.get()


def identify(*, app: str | None = None, process: str | None = None) -> None:
    """Name the process: ``app`` is the app's name, ``process`` its role
    (``api``, ``worker``, ``scheduler``, ``admin``, ``cli``)."""
    if app is not None:
        _identity["app"] = app
    if process is not None:
        _identity["process"] = process


_previous_factory = logging.getLogRecordFactory()


def _stamping_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    """Every record, whichever logger made it and whichever handler sees it,
    carries the current run's fields, or None outside a run."""
    record = _previous_factory(*args, **kwargs)
    fields = _current.get()
    for name in FIELD_NAMES:
        setattr(record, name, getattr(fields, name) if fields is not None else None)
    return record


if logging.getLogRecordFactory() is not _stamping_factory:
    logging.setLogRecordFactory(_stamping_factory)


# --- formats ---------------------------------------------------------------------------------


def _redact(text: str) -> str:
    from backend_builder import secrets

    return secrets.default.redact(text)


class HumanFormatter(logging.Formatter):
    """``HH:MM:SS LEVEL logger message [reaction_id=…]``, redacted."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-5s %(name)s %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        reaction_id = getattr(record, "reaction_id", None)
        if reaction_id and f"reaction_id={reaction_id}" not in line:
            line = f"{line} reaction_id={reaction_id}"
        return _redact(line)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, the process's
    identity, the run's fields when inside one, and the exception if any."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **_identity,
        }
        for name in FIELD_NAMES:
            value = getattr(record, name, None)
            if value is not None:
                line[name] = value
        if record.exc_info and record.exc_info[1] is not None:
            line["exception"] = self.formatException(record.exc_info)
        return _redact(json.dumps(line, default=str))


class _CurrentStream:
    """Always writes to the current ``sys.stdout`` or ``sys.stderr``, so a
    handler survives test runners and tools that swap or close them."""

    def __init__(self, name: Literal["stdout", "stderr"]) -> None:
        self._name = name

    def write(self, text: str) -> int:
        return getattr(sys, self._name).write(text)

    def flush(self) -> None:
        getattr(sys, self._name).flush()


class _Handler(logging.StreamHandler):  # type: ignore[type-arg]
    """Marks the handler ``configure`` owns, so a second call replaces it."""


def configure(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    format: Format = "human",
) -> None:
    """Send every logger's records at ``level`` and above to ``stream``:
    ``human`` lines to stderr, ``json`` lines to stdout, unless ``stream`` is
    given. Attaches to the root logger so app loggers are included. Replaces
    what a previous call attached, so it is safe to call more than once.
    """
    root = logging.getLogger()
    target = (
        stream if stream is not None else _CurrentStream("stdout" if format == "json" else "stderr")
    )
    handler = _Handler(target)  # type: ignore[arg-type]
    handler.setFormatter(JsonFormatter() if format == "json" else HumanFormatter())
    root.handlers = [h for h in root.handlers if not isinstance(h, _Handler)]
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger(ROOT).setLevel(logging.NOTSET)  # defer to the root's level
    for noisy in ("httpx", "httpcore", "hpack", "asyncio"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def as_dict(fields: RunFields) -> dict[str, str]:
    return asdict(fields)
