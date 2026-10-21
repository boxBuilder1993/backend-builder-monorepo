"""Shop: an order-to-cash example app on backend-builder."""

from backend_builder import App

from shop.api import endpoints
from shop.triggers import triggers

app = App(package="shop", endpoints=endpoints, triggers=triggers)

__all__ = ["app"]
