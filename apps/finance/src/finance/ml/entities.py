"""The ML state: a Fit is how a model is made, a FitRun is one execution, a Model is one result,
an InferRun is one scoring, an Evaluation is metrics over one scoring. Nothing about judging a model
lives on the Fit or the Model."""

from __future__ import annotations

from datetime import date, datetime
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


class BacktestGrid(Entity):
    """Every rule combination for one source and capital over one period, evaluated in one run.
    See docs/finance-portfolio.md §6."""

    label: str
    source: dict[str, Any]
    """{"baseline": "momentum_12_1" | "low_vol" | "equal_weight" | "niftybees"} or, later,
    {"walk_forward_id": ...}."""
    capital: float
    costs: str
    """A schedule name in finance.ml.costs.SCHEDULES."""
    start_date: date
    end_date: date
    grid: dict[str, Any]
    """{rule name: [values]} for finance.ml.backtest.Rules."""
    draws: int = 1
    status: str = "running"
    """``running`` | ``done`` | ``failed``."""
    combinations: int = 0
    results_path: str | None = None
    """parquet/backtests/grid_<id>.parquet in the mirror: one row per combination."""
    best: dict[str, Any] | None = None
    """The combination with the highest end value, and its numbers."""
    error: str | None = None


class BacktestRun(Entity):
    """One account kept in full: its trades and its daily equity."""

    label: str
    source: dict[str, Any]
    rules: dict[str, Any]
    costs: str
    capital: float
    start_date: date
    end_date: date
    status: str = "running"
    equity_path: str | None = None
    """parquet/backtests/run_<id>_equity.parquet: date, equity, cash."""
    trades_path: str | None = None
    """parquet/backtests/run_<id>_trades.parquet: date, symbol, side, shares, price, value,
    charges, why."""
    summary: dict[str, Any] | None = None
    error: str | None = None
