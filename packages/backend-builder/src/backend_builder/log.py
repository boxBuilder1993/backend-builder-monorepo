"""Logging for the backend_builder package.

The package logs under the ``backend_builder`` logger hierarchy and, as a
library, attaches only a ``NullHandler`` by default. Applications and scripts
call ``configure`` to see output.

Conventions:
  INFO   one line per lifecycle event, identifiers only: reaction name,
         reaction id, parent id, counts, durations. Never payloads.
  DEBUG  payloads: the input a reaction received, the rows a flush wrote.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

ROOT = "backend_builder"

logging.getLogger(ROOT).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """A logger under the package hierarchy. ``name`` is usually ``__name__``."""
    if name == ROOT or name.startswith(ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT}.{name}")


class _CurrentStderr:
    """A stream-like object that always writes to the current ``sys.stderr``,
    so a handler survives test runners and tools that swap or close it."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


def configure(level: int | str = logging.INFO, *, stream: TextIO | None = None) -> None:
    """Send package logs at ``level`` and above to ``stream``, or to whatever
    ``sys.stderr`` is at the time of each message when no stream is given.

    Replaces any stream handler a previous call attached, so it is safe to
    call more than once.
    """
    logger = logging.getLogger(ROOT)
    handler = logging.StreamHandler(stream if stream is not None else _CurrentStderr())  # type: ignore[arg-type]
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s %(name)s %(message)s", "%H:%M:%S")
    )
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    logger.addHandler(handler)
    logger.setLevel(level)
