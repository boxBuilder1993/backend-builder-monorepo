from backend_builder import Entity


class Order(Entity):
    user_id: str
    tenant: str = "default"
    total: int = 0
    status: str = "new"
    """``new`` until paid or expired. ``ExpireUnpaidOrders`` moves stale
    ``new`` orders to ``expired`` on a schedule."""


class OrderLine(Entity):
    order_id: str
    price: int
    qty: int
