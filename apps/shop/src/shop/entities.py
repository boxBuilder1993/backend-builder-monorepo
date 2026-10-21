from backend_builder import Entity


class Order(Entity):
    user_id: str
    tenant: str = "default"
    total: int = 0


class OrderLine(Entity):
    order_id: str
    price: int
    qty: int
