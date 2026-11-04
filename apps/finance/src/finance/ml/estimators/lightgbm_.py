"""lightgbm: the second boosting library, so a finding can be told from a library quirk. One column
at a time; a vector target is one booster per column inside the model. CPU.

A fit streams. The chunks are read three times, for the labels, for LightGBM's binning sample and
for the rows, through a ``lgb.Sequence`` that holds one chunk at a time, and what stays in memory is
the binned matrix, about a byte per cell: 46 million rows of 89 features bin to some 4 GB where the
float32 matrix would be 16 GB. The result is the same single fit as from the whole matrix; LightGBM
bins from a sample of 200,000 rows either way and then pushes every row."""

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


def _labels(batches: E.Batches) -> tuple[np.ndarray, int]:
    """Every chunk's Y stacked (small: one float per row per target) and the first chunk's length,
    the granularity LightGBM pushes rows at."""
    ys: list[np.ndarray] = []
    first = 0
    for X, Y in batches():
        if not ys:
            first = len(X)
        ys.append(np.asarray(Y))
    if not ys:
        raise ValueError("no rows to fit")
    return np.vstack(ys), max(first, 1)


def chunk_sequence_class() -> type[Any]:
    import lightgbm as lgb

    class ChunkSequence(lgb.Sequence):
        """Random access over a re-iterable factory of chunks, holding one chunk. LightGBM reads
        its binning sample in ascending row order and then the rows in slices from the start, so
        each pass is one forward iteration; an index behind the held chunk restarts the factory."""

        def __init__(self, batches: E.Batches, nrow: int, batch_size: int) -> None:
            self.batch_size = batch_size
            self._batches = batches
            self._nrow = nrow
            self._it: Any = None
            self._start = 0
            self._chunk: np.ndarray | None = None

        def __len__(self) -> int:
            return self._nrow

        def _load(self, i: int) -> np.ndarray:
            if self._chunk is None or i < self._start:
                self._it = iter(self._batches())
                self._start = 0
                self._chunk = np.asarray(next(self._it)[0])
            while i >= self._start + len(self._chunk):
                self._start += len(self._chunk)
                self._chunk = np.asarray(next(self._it)[0])
            return self._chunk

        def __getitem__(self, idx: Any) -> np.ndarray:
            if isinstance(idx, slice):
                start, stop, step = idx.indices(self._nrow)
                if step != 1:
                    raise ValueError("a chunk sequence is read in order")
                parts: list[np.ndarray] = []
                i = start
                while i < stop:
                    chunk = self._load(i)
                    off = i - self._start
                    take = min(stop - i, len(chunk) - off)
                    parts.append(chunk[off : off + take])
                    i += take
                return parts[0] if len(parts) == 1 else np.vstack(parts)
            if isinstance(idx, list):
                return np.stack([self[int(i)] for i in idx])
            chunk = self._load(int(idx))  # the binning sample is read a row at a time, as double
            return chunk[int(idx) - self._start].astype("float64")

    return ChunkSequence


def _native(params: dict[str, Any], schema: Schema, seed: int) -> dict[str, Any]:
    """The sklearn-style names are aliases LightGBM's native API accepts; the objective is what the
    sklearn estimators would have chosen for the task."""
    out = {k: v for k, v in params.items() if k != "n_estimators"}
    out.update(seed=seed, verbose=-1)
    if schema.target.task == "regression":
        out["objective"] = "regression"
    elif len(schema.target.classes or ()) <= 2:
        out["objective"] = "binary"
    else:
        out.update(objective="multiclass", num_class=len(schema.target.classes or ()))
    return out


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
        import lightgbm as lgb

        Y, batch_size = _labels(batches)
        seq = chunk_sequence_class()(batches, len(Y), batch_size)
        native = _native(params, schema, seed)
        rounds = int(params["n_estimators"])
        classifier = schema.target.task != "regression"
        parts: dict[str, E.Fitted] = {}
        dataset: Any = None
        for i, col in enumerate(schema.target.columns):
            label = Y[:, i].astype("float64")
            if dataset is None:
                dataset = lgb.Dataset([seq], label=label, params=native)
            else:
                dataset.set_label(label)
            booster = lgb.train(native, dataset, num_boost_round=rounds)
            parts[col] = LightgbmFitted(_BoosterWrapper(booster, classifier), col, schema)
        return parts[schema.target.columns[0]] if len(parts) == 1 else E.PerColumn(parts)

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
    """A Booster answering like the sklearn estimator the first version of this adapter used."""

    def __init__(self, booster: Any, classifier: bool) -> None:
        self.booster_ = booster
        self.classifier = classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.booster_.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.asarray(self.booster_.predict(X))
        return np.column_stack([1 - p, p]) if p.ndim == 1 else p
