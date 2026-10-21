"""Shop: an order-to-cash example app on backend-builder."""

from backend_builder import App

from shop.api import endpoints

app = App(package="shop", endpoints=endpoints)

__all__ = ["app"]
