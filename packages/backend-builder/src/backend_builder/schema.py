"""Entities to tables.

A persisted entity becomes one table whose columns are its persisted fields.
Only tables and columns come from entities. Constraints, indexes and every
other database feature belong to the app's migrations, so autogenerate is
configured elsewhere to ignore them.
"""

from __future__ import annotations

import dataclasses
import re
import types
import typing
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from backend_builder.entity import Entity


class SchemaError(Exception):
    """An entity field cannot be mapped to a column."""


PERSIST = "persist"
"""Field metadata key. ``field(metadata={PERSIST: False})`` keeps a field off
the table."""


def table_name(entity: type[Entity]) -> str:
    """Class name in snake case: ``OrderLine`` becomes ``order_line``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", entity.__name__).lower()


def is_persisted(field: dataclasses.Field[Any]) -> bool:
    return bool(field.metadata.get(PERSIST, True))


def persisted_fields(entity: type[Entity]) -> list[dataclasses.Field[Any]]:
    return [f for f in dataclasses.fields(entity) if is_persisted(f)]


_SCALARS: dict[type, Any] = {
    str: sa.Text(),
    int: sa.BigInteger(),
    float: sa.Double(),
    Decimal: sa.Numeric(),
    bool: sa.Boolean(),
    datetime: sa.DateTime(timezone=True),
    date: sa.Date(),
}


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """``X | None`` to ``(X, True)``; anything else to ``(annotation, False)``.
    A union with more than one non-None member is returned whole."""
    origin = typing.get_origin(annotation)
    if origin is types.UnionType or origin is typing.Union:
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        nullable = len(members) < len(typing.get_args(annotation))
        if len(members) == 1:
            return members[0], nullable
        return typing.Union[tuple(members)], nullable  # noqa: UP007
    return annotation, False


def column_type(annotation: Any) -> Any:
    """The column type for a field annotation. Scalars map to their SQL
    types; anything else that is JSON-shaped, dicts, lists, dataclasses,
    entities, unions, maps to ``jsonb``."""
    inner, _ = _unwrap_optional(annotation)
    if inner in _SCALARS:
        return _SCALARS[inner]
    origin = typing.get_origin(inner)
    if inner in (dict, list) or origin in (dict, list, types.UnionType, typing.Union):
        return pg.JSONB()
    if isinstance(inner, type) and dataclasses.is_dataclass(inner):
        return pg.JSONB()
    if inner is Any:
        return pg.JSONB()
    raise SchemaError(f"no column type for {annotation!r}")


def table(entity: type[Entity], metadata: sa.MetaData) -> sa.Table:
    hints = typing.get_type_hints(entity)
    columns: list[sa.Column[Any]] = []
    for field in persisted_fields(entity):
        annotation = hints[field.name]
        if field.name == "id":
            columns.append(sa.Column("id", pg.UUID(as_uuid=False), primary_key=True))
            continue
        try:
            kind = column_type(annotation)
        except SchemaError as exc:
            raise SchemaError(f"{entity.__name__}.{field.name}: {exc}") from None
        _, nullable = _unwrap_optional(annotation)
        columns.append(sa.Column(field.name, kind, nullable=nullable))
    return sa.Table(table_name(entity), metadata, *columns)


def build_metadata(entities: Iterable[type[Entity]]) -> sa.MetaData:
    """One table per entity. Raises ``SchemaError`` for an unmappable field."""
    metadata = sa.MetaData()
    for entity in entities:
        table(entity, metadata)
    return metadata
