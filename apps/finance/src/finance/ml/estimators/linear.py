"""sklearn.linear: ridge for regression, logistic for classification. Deliberately weak: the
baseline every boosted model must beat on the validation tables before it means anything."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from finance.ml import estimators as E
from finance.ml.schema import Schema


@dataclass(frozen=True)
class LinearParams:
    alpha: float = 1.0
    """Ridge regularisation; for logistic the inverse, C = 1 / alpha."""
    max_iter: int = 1000


@dataclass
class LinearFitted:
    model: Any
    imputer: Any
    schema: Schema

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        Xf = self.imputer.transform(X)
        t = self.schema.target
        if t.task == "regression":
            p = np.asarray(self.model.predict(Xf), dtype="float64").reshape(len(X), -1)
            return {f"pred_{c}": p[:, i] for i, c in enumerate(t.columns)}
        proba = np.asarray(self.model.predict_proba(Xf))
        return E.classification_outputs(t.columns[0], t.classes or (), proba)

    def save(self, directory: Path) -> None:
        (directory / "model.pkl").write_bytes(pickle.dumps((self.model, self.imputer)))


@E.register
class Linear:
    name: ClassVar[str] = "sklearn.linear"
    Params: ClassVar[type[Any]] = LinearParams
    multi_output: ClassVar[bool] = True
    devices: ClassVar[frozenset[str]] = frozenset({"cpu"})
    budget: ClassVar[float] = 600

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
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge

        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Xf = imputer.transform(X)
        if schema.target.task == "regression":
            model = Ridge(alpha=params["alpha"], random_state=seed)
            model.fit(Xf, Y if Y.shape[1] > 1 else Y[:, 0])
        else:
            model = LogisticRegression(C=1.0 / params["alpha"], max_iter=params["max_iter"])
            model.fit(Xf, Y[:, 0].astype(int))
        return LinearFitted(model, imputer, schema)

    @classmethod
    def load(cls, directory: Path, schema: Schema) -> E.Fitted:
        model, imputer = pickle.loads((directory / "model.pkl").read_bytes())
        return LinearFitted(model, imputer, schema)
