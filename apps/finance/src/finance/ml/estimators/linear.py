"""sklearn.linear: ridge for regression, logistic for classification. Deliberately weak: the
baseline every boosted model must beat on the validation tables before it means anything.

The ridge is solved in closed form from sums accumulated over chunks (n, Σx, Σy, ΣxxT, ΣxyT in
float64), centred so the intercept is unpenalised, exactly as scikit-learn's ``Ridge`` does on the
whole matrix; the two agree to floating point. Nulls are imputed with per-column medians taken
from a reservoir sample of up to ``MEDIAN_SAMPLE`` rows on a first pass (the whole data when it is
smaller). Logistic regression has no closed form and concatenates the chunks."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from finance.ml import estimators as E
from finance.ml.schema import Schema

MEDIAN_SAMPLE = 1_000_000


@dataclass(frozen=True)
class LinearParams:
    alpha: float = 1.0
    """Ridge regularisation; for logistic the inverse, C = 1 / alpha."""
    max_iter: int = 1000


class MedianImputer:
    """Per-column medians; NaN in X becomes the column's median (0 where a column was all null)."""

    def __init__(self, medians: np.ndarray) -> None:
        self.medians = np.where(np.isnan(medians), 0.0, medians).astype("float64")

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.asarray(X, dtype="float64").copy()
        mask = np.isnan(out)
        if mask.any():
            out[mask] = np.broadcast_to(self.medians, out.shape)[mask]
        return out


class RidgeSolution:
    """coef_ (targets, features) and intercept_ (targets,), predicting like a fitted Ridge."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray) -> None:
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = np.asarray(X, dtype="float64") @ self.coef_.T + self.intercept_
        return p[:, 0] if p.shape[1] == 1 else p


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


def _reservoir_medians(batches: E.Batches, seed: int) -> np.ndarray:
    """Per-column nan-medians of a uniform sample of up to MEDIAN_SAMPLE rows, one pass. Each
    arriving row replaces a random slot with probability MEDIAN_SAMPLE / rows seen so far."""
    rng = np.random.default_rng(seed)
    sample: np.ndarray | None = None
    seen = 0
    for X, _ in batches():
        rest = np.asarray(X, dtype="float32")
        if sample is None:
            sample = rest[:MEDIAN_SAMPLE].copy()
            seen = len(sample)
            rest = rest[MEDIAN_SAMPLE:]
        if len(sample) < MEDIAN_SAMPLE and len(rest):
            take = min(MEDIAN_SAMPLE - len(sample), len(rest))
            sample = np.vstack([sample, rest[:take]])
            seen += take
            rest = rest[take:]
        if len(rest):
            slots = rng.integers(0, seen + np.arange(1, len(rest) + 1))
            keep = slots < MEDIAN_SAMPLE
            sample[slots[keep]] = rest[keep]
            seen += len(rest)
    if sample is None or not len(sample):
        raise ValueError("no rows to fit")
    with np.errstate(all="ignore"):
        return np.nanmedian(sample.astype("float64"), axis=0)


def _ridge(batches: E.Batches, imputer: MedianImputer, alpha: float) -> RidgeSolution:
    n = 0
    sx = sy = sxx = sxy = None
    for X, Y in batches():
        Xf = imputer.transform(X)
        Yf = np.asarray(Y, dtype="float64")
        if sx is None:
            d, k = Xf.shape[1], Yf.shape[1]
            sx, sy = np.zeros(d), np.zeros(k)
            sxx, sxy = np.zeros((d, d)), np.zeros((d, k))
        n += len(Xf)
        sx += Xf.sum(axis=0)
        sy += Yf.sum(axis=0)
        sxx += Xf.T @ Xf
        sxy += Xf.T @ Yf
    if sx is None or sy is None or sxx is None or sxy is None or n == 0:
        raise ValueError("no rows to fit")
    mx, my = sx / n, sy / n
    gram = sxx - n * np.outer(mx, mx) + alpha * np.eye(len(mx))
    cross = sxy - n * np.outer(mx, my)
    coef = np.linalg.solve(gram, cross).T  # (targets, features)
    intercept = my - coef @ mx
    return RidgeSolution(coef, intercept)


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
        return cls.fit_batches(E.single(X, Y), schema, params, device, seed)

    @classmethod
    def fit_batches(
        cls,
        batches: E.Batches,
        schema: Schema,
        params: dict[str, Any],
        device: str,
        seed: int,
    ) -> E.Fitted:
        imputer = MedianImputer(_reservoir_medians(batches, seed))
        if schema.target.task == "regression":
            return LinearFitted(_ridge(batches, imputer, float(params["alpha"])), imputer, schema)
        from sklearn.linear_model import LogisticRegression

        X, Y = E.concat(batches)
        model = LogisticRegression(C=1.0 / params["alpha"], max_iter=params["max_iter"])
        model.fit(imputer.transform(X), Y[:, 0].astype(int))
        return LinearFitted(model, imputer, schema)

    @classmethod
    def load(cls, directory: Path, schema: Schema) -> E.Fitted:
        model, imputer = pickle.loads((directory / "model.pkl").read_bytes())
        return LinearFitted(model, imputer, schema)
