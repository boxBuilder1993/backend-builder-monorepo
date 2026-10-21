"""What data changes and schedules cause which reactions, delivered after
commit by the worker. Schedules are seeded by the scheduler."""

from backend_builder import OnChange, OnCreate, OnDelete, OnSchedule

from shop.entities import Order, OrderLine
from shop.reactions import ExpireUnpaidOrders, NotifyOrder, OrderTotal

triggers = [
    OnCreate(Order, NotifyOrder),
    OnCreate(OrderLine, OrderTotal),
    OnChange(OrderLine, ["price", "qty"], OrderTotal),
    OnDelete(OrderLine, OrderTotal),
    OnSchedule("* * * * *", ExpireUnpaidOrders),
]
