"""lightgbm: the second boosting library, so a finding can be told from a library quirk. One column
at a time; a vector target is one booster per column inside the model. CPU."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from finance.ml import estimators as E
from finance.ml.schema import Schema


@dataclass(frozen=True)
class LightgbmParams:
    n_estimators: int = 300
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.1
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_lambda: float = 0.0


@dataclass
class LightgbmFitted:
    model: Any
    column: str
    schema: Schema

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if self.schema.target.task == "regression":
            return {f"pred_{self.column}": np.asarray(self.model.predict(X), dtype="float64")}
        proba = np.asarray(self.model.predict_proba(X))
        return E.classification_outputs(self.column, self.schema.target.classes or (), proba)

    def save(self, directory: Path) -> None:
        self.model.booster_.save_model(str(directory / "model.txt"))


@E.register
class Lightgbm:
    name: ClassVar[str] = "lightgbm"
    Params: ClassVar[type[Any]] = LightgbmParams
    multi_output: ClassVar[bool] = False
    devices: ClassVar[frozenset[str]] = frozenset({"cpu"})
    budget: ClassVar[float] = 1800

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        schema: Schema,
        params: dict[str, Any],
        device: str,
        seed: int,
    ) -> E.Fitted:
        import lightgbm as lgb

        common = dict(params, random_state=seed, verbose=-1)
        parts: dict[str, E.Fitted] = {}
        for i, col in enumerate(schema.target.columns):
            if schema.target.task == "regression":
                m = lgb.LGBMRegressor(**common)
                m.fit(X, Y[:, i])
            else:
                m = lgb.LGBMClassifier(**common)
                m.fit(X, Y[:, i].astype(int))
            parts[col] = LightgbmFitted(m, col, schema)
        return parts[schema.target.columns[0]] if len(parts) == 1 else E.PerColumn(parts)

    @classmethod
    def fit_batches(
        cls,
        batches: E.Batches,
        schema: Schema,
        params: dict[str, Any],
        device: str,
        seed: int,
    ) -> E.Fitted:
        """No streaming form here: the chunks are stacked and fitted in memory."""
        X, Y = E.concat(batches)
        return cls.fit(X, Y, schema, params, device, seed)

    @classmethod
    def load(cls, directory: Path, schema: Schema) -> E.Fitted:
        import lightgbm as lgb

        def one(d: Path, col: str) -> E.Fitted:
            booster = lgb.Booster(model_file=str(d / "model.txt"))
            wrapper = _BoosterWrapper(booster, schema.target.task != "regression")
            return LightgbmFitted(wrapper, col, schema)

        cols = schema.target.columns
        if len(cols) == 1:
            return one(directory, cols[0])
        return E.PerColumn({c: one(directory / f"col_{c}", c) for c in cols})


class _BoosterWrapper:
    """A loaded Booster answering like the sklearn estimator it was saved from."""

    def __init__(self, booster: Any, classifier: bool) -> None:
        self.booster_ = booster
        self.classifier = classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.booster_.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.asarray(self.booster_.predict(X))
        return np.column_stack([1 - p, p]) if p.ndim == 1 else p
