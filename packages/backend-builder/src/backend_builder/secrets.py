"""Secrets: values a reaction needs but the repo must never hold.

A secret is named in lowercase snake case and read from the environment as
``SECRET_<NAME>`` upper-cased, so ``ctx.secret("stripe_api_key")`` reads
``SECRET_STRIPE_API_KEY``. The platform that runs the process (compose,
Kubernetes, a cloud secret manager injecting variables) owns storage,
rotation and audit; this module owns access and redaction.

Every value read through ``Secrets`` is remembered so it can be masked
wherever text leaves the process: run records and log lines. Redaction is by
exact value, so an encoded or truncated secret still leaks; it is a seatbelt,
not a vault. Values shorter than ``MIN_REDACTED`` are not masked, because
replacing every "test" in the logs is worse than the leak; reading one logs a
warning.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from typing import Any

ENV_PREFIX = "SECRET_"
MIN_REDACTED = 8
NAME = re.compile(r"^[a-z][a-z0-9_]*$")

logger = logging.getLogger("backend_builder.secrets")


class SecretError(LookupError):
    """A secret is missing from the environment, or its name is malformed."""


def env_name(name: str) -> str:
    """The environment variable holding ``name``; ``SecretError`` for a bad name."""
    if not NAME.match(name):
        raise SecretError(f"bad secret name {name!r}: lowercase letters, digits and underscores")
    return ENV_PREFIX + name.upper()


class Secrets:
    """Access to the environment's secrets plus the redactor for what was read.

    ``env`` defaults to the process environment, consulted on every read so a
    value set after construction (a test's monkeypatch, say) is seen.
    """

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._env = env
        self._seen: dict[str, str] = {}

    def get(self, name: str) -> str:
        """The value of ``name``. ``SecretError`` if the variable is unset or empty."""
        variable = env_name(name)
        env = os.environ if self._env is None else self._env
        value = env.get(variable)
        if not value:
            raise SecretError(f"secret {name!r} is not set: export {variable}")
        if name not in self._seen:
            if len(value) < MIN_REDACTED:
                logger.warning(
                    "secret %s is shorter than %d characters and will not be redacted",
                    name,
                    MIN_REDACTED,
                )
            self._seen[name] = value
        return value

    @property
    def seen(self) -> tuple[str, ...]:
        """Names read so far, for diagnostics. Never the values."""
        return tuple(self._seen)

    def redact(self, text: str) -> str:
        """``text`` with every secret value read so far replaced by ``[secret:name]``."""
        for name, value in self._seen.items():
            if len(value) >= MIN_REDACTED and value in text:
                text = text.replace(value, f"[secret:{name}]")
        return text

    def redact_jsonable(self, value: Any) -> Any:
        """``redact`` applied to every string inside an already JSON-encodable value."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return {k: self.redact_jsonable(v) for k, v in value.items()}  # type: ignore[union-attr]
        if isinstance(value, list):
            return [self.redact_jsonable(v) for v in value]  # type: ignore[union-attr]
        return value


default = Secrets()
"""The process's secrets. Contexts use it unless given another."""
