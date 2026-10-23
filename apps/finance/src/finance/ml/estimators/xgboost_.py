"""xgboost: gradient-boosted trees, the workhorse. Regression fits a vector target jointly (one tree
set per target); classification is one column. Categoricals arrive as integer codes with NaN for
missing and are treated as ordinal in this version."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from finance.ml import estimators as E
from finance.ml.schema import Schema


@dataclass(frozen=True)
class XgboostParams:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0


@dataclass
class XgboostFitted:
    model: Any
    schema: Schema

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        t = self.schema.target
        if t.task == "regression":
            p = np.asarray(self.model.predict(X), dtype="float64")
            p = p.reshape(len(X), -1)
            return {f"pred_{c}": p[:, i] for i, c in enumerate(t.columns)}
        proba = np.asarray(self.model.predict_proba(X))
        return E.classification_outputs(t.columns[0], t.classes or (), proba)

    def save(self, directory: Path) -> None:
        self.model.save_model(str(directory / "model.ubj"))


@E.register
class Xgboost:
    name: ClassVar[str] = "xgboost"
    Params: ClassVar[type[Any]] = XgboostParams
    multi_output: ClassVar[bool] = True
    devices: ClassVar[frozenset[str]] = frozenset({"cpu", "cuda"})
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
        import xgboost as xgb

        common = dict(params, device=device, tree_method="hist", random_state=seed)
        if schema.target.task == "regression":
            model = xgb.XGBRegressor(**common)
            model.fit(X, Y if Y.shape[1] > 1 else Y[:, 0])
        else:
            model = xgb.XGBClassifier(**common)
            model.fit(X, Y[:, 0].astype(int))
        return XgboostFitted(model, schema)

    @classmethod
    def load(cls, directory: Path, schema: Schema) -> E.Fitted:
        import xgboost as xgb

        model = xgb.XGBRegressor() if schema.target.task == "regression" else xgb.XGBClassifier()
        model.load_model(str(directory / "model.ubj"))
        return XgboostFitted(model, schema)
