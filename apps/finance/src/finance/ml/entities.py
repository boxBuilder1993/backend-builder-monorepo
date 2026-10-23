"""The ML state: a Fit is how a model is made, a FitRun is one execution, a Model is one result,
an InferRun is one scoring, an Evaluation is metrics over one scoring. Nothing about judging a model
lives on the Fit or the Model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from backend_builder import Entity


class Fit(Entity):
    """A definition. Immutable once run; a change is a new Fit."""

    name: str
    description: str = ""
    query: str
    """SQL over the DuckDB views; yields key + feature + target columns. Params as :name."""
    params: dict[str, Any]
    """The query's named parameters and their defaults."""
    key: list[str]
    features: list[dict[str, Any]]
    """[{"name": ..., "kind": "numeric" | "categorical"}, ...]"""
    target: dict[str, Any]
    """{"columns": [...], "task": "regression" | "binary" | "multiclass"}"""
    search: list[dict[str, Any]]
    """[{"algorithm": ..., "params": {name: value | [values]}}, ...]"""
    content_hash: str
    """Of everything above except name and description; registering it twice returns this row."""


class FitRun(Entity):
    """One execution of a Fit."""

    fit_id: str
    params: dict[str, Any]
    status: str = "queued"
    """``queued`` | ``running`` | ``done`` | ``failed``."""
    specs: int = 0
    rows: int = 0
    data_hash: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class Model(Entity):
    """One fitted model. The id is the model id; the store is where it lives."""

    fit_run_id: str
    fit_id: str
    algorithm: str
    params: dict[str, Any]
    """The resolved point of the search, defaults filled in."""
    schema: dict[str, Any] | None = None
    """Key, features with kinds and categorical levels, target with classes: the contract."""
    status: str = "queued"
    """``queued`` | ``fitting`` | ``uploading`` | ``fitted`` | ``failed``."""
    repo: str | None = None
    path: str | None = None
    revision: str | None = None
    sha256: str | None = None
    bytes: int = 0
    cache_path: str | None = None
    fit_seconds: float | None = None
    device: str | None = None
    error: str | None = None


class InferRun(Entity):
    """One scoring of a model: a model id, a label and a query."""

    model_id: str
    label: str
    query: str
    params: dict[str, Any]
    status: str = "running"
    """``running`` | ``done`` | ``failed``."""
    rows: int = 0
    unknown_levels: dict[str, Any] | None = None
    path: str | None = None
    """parquet/inference/<infer run id>.parquet in the mirror, a DataFile."""
    error: str | None = None


class Evaluation(Entity):
    """Metrics over one InferRun's prediction table. Several may sit on one run."""

    infer_run_id: str
    metrics: list[str]
    params: dict[str, Any]
    results: dict[str, Any] | None = None
    status: str = "done"
    """``done`` | ``failed``."""
    error: str | None = None
