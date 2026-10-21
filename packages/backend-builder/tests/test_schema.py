from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from backend_builder import Entity
from backend_builder.schema import PERSIST, SchemaError, build_metadata, table_name
from sqlalchemy.dialects import postgresql as pg


@dataclass
class Address:
    city: str


class Order(Entity):
    pass


class Everything(Entity):
    text: str
    big: int
    real: float
    money: Decimal
    flag: bool
    when: datetime
    day: date
    maybe: str | None = None
    blob: dict[str, Any] = field(default_factory=dict[str, Any])
    items: list[int] = field(default_factory=list[int])
    address: Address | None = None
    order: Order | None = None
    either: int | str = 0
    scratch: list[str] = field(default_factory=list[str], metadata={PERSIST: False})


class Unmappable(Entity):
    weird: set[int] = field(default_factory=set[int])


def test_table_name_is_snake_case() -> None:
    assert table_name(Order) == "order"
    assert table_name(Everything) == "everything"

    class OrderLineItem(Entity):
        pass

    assert table_name(OrderLineItem) == "order_line_item"


def test_columns_follow_the_mapping() -> None:
    metadata = build_metadata([Everything])
    t = metadata.tables["everything"]
    kinds = {c.name: type(c.type) for c in t.columns}

    assert kinds["id"] is pg.UUID and t.c.id.primary_key
    assert kinds["created_at"] is sa.DateTime and t.c.created_at.type.timezone  # type: ignore[attr-defined]
    assert kinds["deleted_at"] is sa.DateTime and t.c.deleted_at.nullable
    assert kinds["text"] is sa.Text
    assert kinds["big"] is sa.BigInteger
    assert kinds["real"] is sa.Double
    assert kinds["money"] is sa.Numeric
    assert kinds["flag"] is sa.Boolean
    assert kinds["when"] is sa.DateTime
    assert kinds["day"] is sa.Date
    assert kinds["maybe"] is sa.Text and t.c.maybe.nullable
    for name in ("blob", "items", "address", "order", "either"):
        assert kinds[name] is pg.JSONB, name
    assert t.c.address.nullable and not t.c.blob.nullable


def test_non_persisted_fields_are_not_columns() -> None:
    t = build_metadata([Everything]).tables["everything"]
    assert "scratch" not in t.columns


def test_required_fields_are_not_nullable() -> None:
    t = build_metadata([Everything]).tables["everything"]
    assert not t.c.text.nullable and not t.c.when.nullable


def test_unmappable_type_fails_with_the_field_named() -> None:
    with pytest.raises(SchemaError, match="Unmappable.weird"):
        build_metadata([Unmappable])
