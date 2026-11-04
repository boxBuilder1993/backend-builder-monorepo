"""xgboost: gradient-boosted trees, the workhorse. Regression fits a vector target jointly (one tree
set per target); classification is one column. Categoricals arrive as integer codes with NaN for
missing and are treated as ordinal in this version.

Fits through the Booster API on a quantised matrix fed chunk by chunk: the data is quantised
into at most 256 bins per feature as it streams in (the same sketch ``tree_method=hist`` builds
from a full matrix), so a fit holds about a byte per cell instead of three float32 copies. On the
GPU that is an in-core ``QuantileDMatrix``, the card's memory being the limit anyway. On the CPU it
is an ``ExtMemQuantileDMatrix``: the quantised pages live in files under the work directory and
stream through training, so a fit of any size holds a page or two of them (a 46-million-row fit
built in core peaked past 24 GB and the OOM killer took the worker). The cache is removed when the
fit ends. Models saved by the earlier scikit-learn wrapper load into a Booster unchanged."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from finance.ml import data
from finance.ml import estimators as E
from finance.ml.schema import Schema

MAX_BIN = 256


def cache_root() -> Path:
    """Where a CPU fit keeps its quantised pages while it trains."""
    root = data.work_dir() / "xgb-cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


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
        raw = np.asarray(self.model.inplace_predict(np.ascontiguousarray(X, dtype="float32")))
        if t.task == "regression":
            p = raw.astype("float64").reshape(len(X), -1)
            return {f"pred_{c}": p[:, i] for i, c in enumerate(t.columns)}
        return E.classification_outputs(t.columns[0], t.classes or (), raw)

    def save(self, directory: Path) -> None:
        self.model.save_model(str(directory / "model.ubj"))


def _objective(schema: Schema) -> dict[str, Any]:
    t = schema.target
    if t.task == "regression":
        return {"objective": "reg:squarederror"}
    k = len(t.classes or ())
    if k == 2:
        return {"objective": "binary:logistic"}
    return {"objective": "multi:softprob", "num_class": k}


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
        import xgboost as xgb

        regression = schema.target.task == "regression"

        class Chunks(xgb.DataIter):
            def __init__(self, cache_prefix: str | None) -> None:
                super().__init__(cache_prefix=cache_prefix, release_data=True)
                self.it: Iterator[tuple[np.ndarray, np.ndarray]] = iter(())

            def reset(self) -> None:
                self.it = batches()

            def next(self, input_data: Any) -> bool:
                try:
                    X, Y = next(self.it)
                except StopIteration:
                    return False
                label = Y if regression and Y.shape[1] > 1 else Y[:, 0]
                if not regression:
                    label = label.astype("int32")
                input_data(data=np.ascontiguousarray(X, dtype="float32"), label=label)
                return True

        rounds = int(params["n_estimators"])
        train = {k: v for k, v in params.items() if k != "n_estimators"}
        train.update(_objective(schema), device=device, tree_method="hist", seed=seed)
        if regression and len(schema.target.columns) > 1:
            train["multi_strategy"] = "one_output_per_tree"
        cache: Path | None = None
        try:
            if device == "cpu":
                cache = Path(tempfile.mkdtemp(prefix="fit-", dir=cache_root()))
                dm: Any = xgb.ExtMemQuantileDMatrix(
                    Chunks(str(cache / "page")), max_bin=MAX_BIN, missing=np.nan
                )
            else:
                dm = xgb.QuantileDMatrix(Chunks(None), max_bin=MAX_BIN, missing=np.nan)
            booster = xgb.train(train, dm, num_boost_round=rounds)
            del dm
        finally:
            if cache is not None:
                shutil.rmtree(cache, ignore_errors=True)
        return XgboostFitted(booster, schema)

    @classmethod
    def load(cls, directory: Path, schema: Schema) -> E.Fitted:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(directory / "model.ubj"))
        return XgboostFitted(booster, schema)
