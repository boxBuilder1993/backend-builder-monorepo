"""What data changes cause which reactions, delivered after commit by the worker."""

from backend_builder import OnChange, OnCreate, OnDelete

from shop.entities import OrderLine
from shop.reactions import OrderTotal

triggers = [
    OnCreate(OrderLine, OrderTotal),
    OnChange(OrderLine, ["price", "qty"], OrderTotal),
    OnDelete(OrderLine, OrderTotal),
]
