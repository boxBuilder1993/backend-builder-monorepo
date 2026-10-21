"""Entity base class.

An entity is a dataclass. Subclasses declare their fields as annotations and
are turned into keyword-only dataclasses automatically. Instances are the rows
a store holds, keyed by their entity class and ``id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import dataclass_transform
from uuid import uuid4


def _new_id() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass_transform(kw_only_default=True)
@dataclass(kw_only=True)
class Entity:
    """Base class for all entities.

    System fields:
      ``id``          primary key, a UUID string, generated unless supplied.
      ``created_at``  UTC time the instance was constructed.
      ``deleted_at``  set by a soft delete; ``None`` while the row is live.
    """

    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_now)
    deleted_at: datetime | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(kw_only=True)(cls)
