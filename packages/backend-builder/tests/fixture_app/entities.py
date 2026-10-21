from backend_builder import Entity


class Widget(Entity):
    name: str
    count: int = 0
