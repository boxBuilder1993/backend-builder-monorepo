"""Data-change and scheduled triggers, declared on the App.

A data-change trigger names an entity, optionally the fields that matter,
and a reaction. At flush the session compares every row it inserted or
changed against the app's triggers and writes one outbox row per matching
change, so the reaction runs after commit, by a worker, with an input that
says what happened and to what::

    {"entity": "order_line", "id": "...", "kind": "created"}
    {"entity": "order_line", "id": "...", "kind": "updated", "changed": ["qty"]}
    {"entity": "order_line", "id": "...", "kind": "deleted"}

The reaction loads current state itself. A scheduled trigger names a cron
expression and a reaction; the worker keeps one pending outbox row per
schedule at its next occurrence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from croniter import croniter

from backend_builder.entity import Entity
from backend_builder.outbox import OutboxEntry
from backend_builder.reaction import Reaction, qualified_name
from backend_builder.schema import persisted_fields, table_name

Kind = Literal["created", "updated", "deleted"]


class TriggerError(Exception):
    pass


@dataclass(frozen=True)
class OnCreate:
    entity: type[Entity]
    reaction: type[Reaction]


@dataclass(frozen=True)
class OnChange:
    entity: type[Entity]
    fields: Sequence[str]
    reaction: type[Reaction]

    def __post_init__(self) -> None:
        if not self.fields:
            raise TriggerError(f"OnChange({self.entity.__name__}): list at least one field")
        known = {f.name for f in persisted_fields(self.entity)}
        unknown = [f for f in self.fields if f not in known]
        if unknown:
            raise TriggerError(f"OnChange({self.entity.__name__}): no persisted field(s) {unknown}")


@dataclass(frozen=True)
class OnDelete:
    entity: type[Entity]
    reaction: type[Reaction]


@dataclass(frozen=True)
class OnSchedule:
    cron: str
    reaction: type[Reaction]

    def __post_init__(self) -> None:
        if not croniter.is_valid(self.cron):
            raise TriggerError(f"OnSchedule: {self.cron!r} is not a valid cron expression")

    def next_after(self, moment: datetime) -> datetime:
        """The first occurrence strictly after ``moment``, in UTC."""
        nxt: datetime = croniter(self.cron, moment.astimezone(UTC)).get_next(datetime)
        return nxt.replace(tzinfo=UTC) if nxt.tzinfo is None else nxt

    def entry(self, scheduled_for: datetime) -> OutboxEntry:
        return OutboxEntry(
            reaction=qualified_name(self.reaction),
            input={"scheduled_for": scheduled_for.isoformat()},
            available_at=scheduled_for,
            scheduled_for=scheduled_for,
        )


DataTrigger = OnCreate | OnChange | OnDelete
Trigger = DataTrigger | OnSchedule


@dataclass(frozen=True, kw_only=True)
class Change:
    """One row's net change in a run, as the session reports it at flush."""

    entity: type[Entity]
    id: str
    kind: Kind
    changed: tuple[str, ...] = field(default_factory=tuple)
    """Changed persisted fields, for ``updated``."""


def match(triggers: Iterable[Trigger], changes: Iterable[Change]) -> list[OutboxEntry]:
    """One outbox entry per (change, matching trigger)."""
    entries: list[OutboxEntry] = []
    for change in changes:
        for trigger in triggers:
            if isinstance(trigger, OnSchedule) or trigger.entity is not change.entity:
                continue
            payload: dict[str, Any] = {
                "entity": table_name(change.entity),
                "id": change.id,
                "kind": change.kind,
            }
            if isinstance(trigger, OnCreate) and change.kind == "created":
                pass
            elif isinstance(trigger, OnDelete) and change.kind == "deleted":
                pass
            elif isinstance(trigger, OnChange) and change.kind == "updated":
                hit = [f for f in change.changed if f in trigger.fields]
                if not hit:
                    continue
                payload["changed"] = hit
            else:
                continue
            entries.append(OutboxEntry(reaction=qualified_name(trigger.reaction), input=payload))
    return entries
