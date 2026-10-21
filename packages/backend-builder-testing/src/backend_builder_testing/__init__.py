"""pytest fixtures for apps built on backend-builder. Registered as a pytest
plugin, so any package in the environment gets them without configuration."""

from backend_builder_testing.fixtures import TEST_DATABASE_URL, create_database, drop_database

__all__ = ["TEST_DATABASE_URL", "create_database", "drop_database"]
