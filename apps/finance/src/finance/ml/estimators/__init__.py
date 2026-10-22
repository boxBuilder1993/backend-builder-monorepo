"""One protocol, adapters behind it, a registry by algorithm name. An adapter turns a prepared
matrix into a fitted estimator, predicts a dict of output columns, and saves and loads itself in a
directory.
Each declares its Params dataclass, whether it fits a vector target jointly, the devices it can use
and a default budget. The task picks the concrete class inside the adapter; a search entry names
only the algorithm."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from finance.ml.schema import Schema


class Fitted(Protocol):
    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Output column -> values. Names come from ``output_columns``."""
        ...

    def save(self, directory: Path) -> None: ...


class Adapter(Protocol):
    name: ClassVar[str]
    Params: ClassVar[type[Any]]
    multi_output: ClassVar[bool]
    devices: ClassVar[frozenset[str]]
    budget: ClassVar[float]

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        schema: Schema,
        params: dict[str, Any],
        device: str,
        seed: int,
    ) -> Fitted: ...

    @classmethod
    def load(cls, directory: Path, schema: Schema) -> Fitted: ...


REGISTRY: dict[str, type[Adapter]] = {}
DEFAULT_SEED = 42


def register(adapter: type[Adapter]) -> type[Adapter]:
    REGISTRY[adapter.name] = adapter
    return adapter


def resolve_params(algorithm: str, point: dict[str, Any]) -> dict[str, Any]:
    """A search point completed with the adapter's defaults and checked against its Params: unknown
    names and wrong types raise."""
    cls = REGISTRY[algorithm].Params
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(point) - set(fields)
    if unknown:
        raise ValueError(f"unknown params {sorted(unknown)}; accepted: {sorted(fields)}")
    for name, value in point.items():
        want = fields[name].type
        if want in ("int", int) and not (isinstance(value, int) and not isinstance(value, bool)):
            raise TypeError(f"{name} must be an int, not {value!r}")
        if want in ("float", float) and not isinstance(value, int | float):
            raise TypeError(f"{name} must be a number, not {value!r}")
        if want in ("str", str) and not isinstance(value, str):
            raise TypeError(f"{name} must be a string, not {value!r}")
    return dataclasses.asdict(cls(**point))


def output_columns(schema: Schema) -> list[str]:
    """pred_<col> per target column; for classification also pred_<col>_<class> probabilities."""
    out: list[str] = []
    for c in schema.target.columns:
        out.append(f"pred_{c}")
        if schema.target.task != "regression":
            out += [f"pred_{c}_{k}" for k in schema.target.classes or ()]
    return out


def pick_device(adapter: type[Adapter], wanted: str) -> str:
    """auto: cuda if the adapter supports it and a GPU is present, else cpu."""
    if wanted != "auto":
        if wanted not in adapter.devices:
            raise ValueError(f"{adapter.name} does not support device {wanted!r}")
        return wanted
    if "cuda" in adapter.devices and _has_cuda():
        return "cuda"
    return "cpu"


def _has_cuda() -> bool:
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


@dataclass
class PerColumn:
    """A vector target on an adapter that fits one column at a time: one fitted estimator per
    column,    the same dict of outputs as a joint fit."""

    parts: dict[str, Fitted]

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for p in self.parts.values():
            out.update(p.predict(X))
        return out

    def save(self, directory: Path) -> None:
        for col, p in self.parts.items():
            sub = directory / f"col_{col}"
            sub.mkdir(parents=True, exist_ok=True)
            p.save(sub)


def classification_outputs(
    col: str, classes: tuple[str, ...], proba: np.ndarray
) -> dict[str, np.ndarray]:
    """From class probabilities (n, k) to pred_<col> (the label) and pred_<col>_<class>."""
    out: dict[str, np.ndarray] = {}
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    labels = np.array(classes, dtype=object)[proba.argmax(axis=1)]
    out[f"pred_{col}"] = labels
    for i, k in enumerate(classes):
        out[f"pred_{col}_{k}"] = proba[:, i].astype("float64")
    return out


def load(algorithm: str, directory: Path, schema: Schema) -> Fitted:
    return REGISTRY[algorithm].load(directory, schema)


def _register_builtin() -> None:
    import importlib

    for module in ("xgboost_", "lightgbm_", "linear"):
        importlib.import_module(f"finance.ml.estimators.{module}")


_register_builtin()
