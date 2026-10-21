"""Values to and from JSON-shaped data, for ``jsonb`` columns.

Encoding walks a value and produces something ``json.dumps`` accepts: dicts
and lists recurse, dataclasses become dicts of their fields, datetimes and
dates become ISO strings, ``Decimal`` becomes a string. Anything else raises
``EncodeError``. Decoding reverses it using the field's annotation, so a
nested dataclass comes back as that dataclass and an ISO string inside it
comes back as a ``datetime`` where the nested field says so.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from datetime import date, datetime
from decimal import Decimal
from typing import Any


class EncodeError(Exception):
    """A value cannot be stored as JSON."""


def to_jsonable(value: Any, *, path: str = "") -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_jsonable(getattr(value, f.name), path=f"{path}.{f.name}")
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():  # type: ignore[union-attr]
            if not isinstance(k, str):
                raise EncodeError(f"{path or 'value'}: dict key {k!r} is not a string")
            out[k] = to_jsonable(v, path=f"{path}.{k}")
        return out
    if isinstance(value, list | tuple):
        return [to_jsonable(v, path=f"{path}[{i}]") for i, v in enumerate(value)]  # type: ignore[union-attr]
    raise EncodeError(f"{path or 'value'}: {type(value).__name__} is not JSON-storable")


def _unwrap_optional(annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin is types.UnionType or origin is typing.Union:
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return members[0] if len(members) == 1 else annotation
    return annotation


def from_jsonable(annotation: Any, value: Any) -> Any:
    """Rebuild ``value`` as ``annotation`` describes. Unknown or ambiguous
    annotations return the value unchanged."""
    if value is None:
        return None
    target = _unwrap_optional(annotation)
    origin = typing.get_origin(target)
    args = typing.get_args(target)

    if target is datetime and isinstance(value, str):
        return datetime.fromisoformat(value)
    if target is date and isinstance(value, str):
        return date.fromisoformat(value)
    if target is Decimal and isinstance(value, str | int | float):
        return Decimal(str(value))
    if isinstance(target, type) and dataclasses.is_dataclass(target) and isinstance(value, dict):
        hints = typing.get_type_hints(target)
        data: dict[str, Any] = value  # type: ignore[assignment]
        return target(**{k: from_jsonable(hints.get(k, Any), v) for k, v in data.items()})
    if origin is list and isinstance(value, list):
        item = args[0] if args else Any
        items: list[Any] = value  # type: ignore[assignment]
        return [from_jsonable(item, v) for v in items]
    if origin is dict and isinstance(value, dict):
        item = args[1] if len(args) == 2 else Any
        mapping: dict[str, Any] = value  # type: ignore[assignment]
        return {k: from_jsonable(item, v) for k, v in mapping.items()}
    return value
