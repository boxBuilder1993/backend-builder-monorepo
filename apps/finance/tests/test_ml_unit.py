"""The ML functions on frames, with no runtime and no network: schema, search, coercion and encoding,
estimators on a small frame, artifacts, metrics, the query parameter binding."""

# ruff: noqa: E501  (test fixtures read better on one line)

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from finance.ml import artifacts, data, metrics, search
from finance.ml import estimators as E
from finance.ml import schema as S

KEY = ["symbol", "date"]
FEATURES = [{"name": "ret_5d", "kind": "numeric"}, {"name": "sector", "kind": "categorical"}]


def frame(n: int = 200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sector = rng.choice(["bank", "it", "pharma"], n)
    ret = rng.normal(0, 0.02, n)
    y = ret * 2 + np.where(sector == "bank", 0.01, 0.0) + rng.normal(0, 0.001, n)
    return pl.DataFrame(
        {
            "symbol": [f"S{i % 20}" for i in range(n)],
            "date": [date(2025, 1, 1 + i // 20) for i in range(n)],
            "ret_5d": ret,
            "sector": sector,
            "fwd": y,
            "up": (y > 0).astype(int),
        }
    )


def test_declare_validates_the_definition() -> None:
    s = S.declare(KEY, FEATURES, {"columns": ["fwd"], "task": "regression"})
    assert s.feature_names == ("ret_5d", "sector") and s.categorical == ("sector",)
    for bad, msg in [
        ({"columns": ["fwd"], "task": "ranking"}, "task"),
        ({"columns": ["a", "b"], "task": "multiclass"}, "exactly one"),
        ({"columns": ["ret_5d"], "task": "regression"}, "several"),
    ]:
        with pytest.raises(ValueError, match=msg):
            S.declare(KEY, FEATURES, bad)
    with pytest.raises(ValueError, match="kind"):
        S.declare(
            KEY, [{"name": "x", "kind": "boolean"}], {"columns": ["fwd"], "task": "regression"}
        )
    assert S.Schema.from_dict(s.to_dict()) == s


def test_search_expands_axes_and_validates_params() -> None:
    specs = search.expand(
        [
            {
                "algorithm": "xgboost",
                "params": {"max_depth": [3, 5], "learning_rate": [0.03, 0.1], "n_estimators": 50},
            },
            {"algorithm": "sklearn.linear", "params": {"alpha": [0.1, 1.0]}},
        ]
    )
    assert [(s.algorithm, s.params["max_depth"], s.params["learning_rate"]) for s in specs[:4]] == [
        ("xgboost", 3, 0.03), ("xgboost", 3, 0.1), ("xgboost", 5, 0.03), ("xgboost", 5, 0.1)
    ]  # fmt: skip
    assert all(s.params["n_estimators"] == 50 and s.params["subsample"] == 1.0 for s in specs[:4])
    assert [s.params["alpha"] for s in specs[4:]] == [0.1, 1.0] and specs[4].params[
        "max_iter"
    ] == 1000
    with pytest.raises(ValueError, match="unknown params"):
        search.expand([{"algorithm": "xgboost", "params": {"max_dpeth": 3}}])
    with pytest.raises(ValueError, match="must be an int"):
        search.expand([{"algorithm": "xgboost", "params": {"max_depth": 3.5}}])
    with pytest.raises(ValueError, match="unknown algorithm"):
        search.expand([{"algorithm": "catboost"}])


def test_prepare_encodes_and_counts_unknown_levels() -> None:
    s = S.declare(KEY, FEATURES, {"columns": ["fwd"], "task": "regression"})
    fitted = data.prepare(frame(), s, fitting=True)
    assert fitted.X.shape == (200, 2) and fitted.X.dtype == np.float32
    assert fitted.schema.features[1].levels == ("bank", "it", "pharma")
    assert fitted.Y is not None and fitted.Y.shape == (200, 1)
    new = frame(seed=1).with_columns(pl.Series("sector", ["auto"] * 5 + ["bank"] * 195))
    scored = data.prepare(new.drop("fwd"), fitted.schema, fitting=False)
    assert scored.unknown_levels == {"sector": 5} and np.isnan(scored.X[:, 1]).sum() == 5
    assert scored.Y is None and scored.truth is None
    with pytest.raises(ValueError, match="did not yield"):
        data.prepare(new.drop("sector"), fitted.schema, fitting=False)
    with pytest.raises(ValueError, match="not numeric"):
        data.prepare(frame().with_columns(pl.lit("x").alias("ret_5d")), s, fitting=True)


def test_prepare_classification_records_classes() -> None:
    s = S.declare(KEY, FEATURES, {"columns": ["up"], "task": "binary"})
    p = data.prepare(frame(), s, fitting=True)
    assert (
        p.schema.target.classes == ("0", "1") and p.Y is not None and set(p.Y[:, 0]) <= {0.0, 1.0}
    )
    assert E.output_columns(p.schema) == ["pred_up", "pred_up_0", "pred_up_1"]


@pytest.mark.parametrize("algorithm", ["xgboost", "lightgbm", "sklearn.linear"])
def test_each_adapter_fits_predicts_saves_and_loads(algorithm: str, tmp_path: Path) -> None:
    s = S.declare(KEY, FEATURES, {"columns": ["fwd"], "task": "regression"})
    p = data.prepare(frame(), s, fitting=True)
    params = E.resolve_params(
        algorithm, {"n_estimators": 30} if algorithm != "sklearn.linear" else {}
    )
    adapter = E.REGISTRY[algorithm]
    assert p.Y is not None
    fitted = adapter.fit(p.X, p.Y, p.schema, params, "cpu", 42)
    preds = fitted.predict(p.X)
    assert list(preds) == ["pred_fwd"] and preds["pred_fwd"].shape == (200,)
    assert np.corrcoef(preds["pred_fwd"], p.Y[:, 0])[0, 1] > 0.8
    artifacts.save(tmp_path, fitted, p.schema, {"algorithm": algorithm, "params": params})
    loaded, schema, info = artifacts.load(tmp_path)
    assert schema == p.schema and info["algorithm"] == algorithm
    np.testing.assert_allclose(
        loaded.predict(p.X)["pred_fwd"], preds["pred_fwd"], rtol=1e-5, atol=1e-6
    )
    sha1, size = artifacts.archive_hash(tmp_path)
    assert len(sha1) == 64 and size > 0 and artifacts.archive_hash(tmp_path) == (sha1, size)


def test_vector_target_and_classification_outputs(tmp_path: Path) -> None:
    f = frame().with_columns((pl.col("fwd") * 2).alias("fwd2"))
    s = S.declare(KEY, FEATURES, {"columns": ["fwd", "fwd2"], "task": "regression"})
    p = data.prepare(f, s, fitting=True)
    assert p.Y is not None
    for algorithm in ("xgboost", "lightgbm", "sklearn.linear"):  # joint or per column, same outputs
        params = E.resolve_params(
            algorithm, {"n_estimators": 20} if algorithm != "sklearn.linear" else {}
        )
        fitted = E.REGISTRY[algorithm].fit(p.X, p.Y, p.schema, params, "cpu", 1)
        assert sorted(fitted.predict(p.X)) == ["pred_fwd", "pred_fwd2"]
        d = tmp_path / algorithm
        artifacts.save(d, fitted, p.schema, {"algorithm": algorithm})
        assert sorted(artifacts.load(d)[0].predict(p.X[:3])) == ["pred_fwd", "pred_fwd2"]
    sb = S.declare(KEY, FEATURES, {"columns": ["up"], "task": "binary"})
    pb = data.prepare(frame(), sb, fitting=True)
    assert pb.Y is not None
    fitted = E.REGISTRY["sklearn.linear"].fit(
        pb.X, pb.Y, pb.schema, E.resolve_params("sklearn.linear", {}), "cpu", 1
    )
    out = fitted.predict(pb.X)
    assert sorted(out) == ["pred_up", "pred_up_0", "pred_up_1"]
    np.testing.assert_allclose(out["pred_up_0"] + out["pred_up_1"], 1.0, atol=1e-6)
    assert set(out["pred_up"]) <= {"0", "1"}


def test_metrics_on_known_inputs() -> None:
    table = pl.DataFrame({"fwd": [0.02, -0.01, 0.03, 0.0], "pred_fwd": [0.01, 0.01, 0.02, -0.01]})
    r = metrics.evaluate(
        table, ["fwd"], ["mae", "net_return_per_trade", "trades", "hit_rate"], {"cost_bps": 100}
    )
    assert r["mae.fwd"] == pytest.approx((0.01 + 0.02 + 0.01 + 0.01) / 4)
    assert r["trades.fwd"] == 3 and r["hit_rate.fwd"] == pytest.approx(2 / 3)
    assert r["net_return_per_trade.fwd"] == pytest.approx((0.02 - 0.01 + 0.03) / 3 - 0.01)
    with pytest.raises(ValueError, match="unknown metrics"):
        metrics.evaluate(table, ["fwd"], ["sharpe"], {})
    with pytest.raises(ValueError, match="lacks"):
        metrics.evaluate(table.drop("fwd"), ["fwd"], ["mae"], {})


def test_query_params_bind_by_name_and_the_views_install(tmp_path: Path) -> None:
    root = tmp_path
    day = root / "parquet/index/year=2026"
    day.mkdir(parents=True)
    pl.DataFrame(
        {
            "date": [date(2026, 1, 2)] * 2,
            "index_name": ["Nifty 50", "Nifty Bank"],
            "close": [1.0, 2.0],
        }
    ).write_parquet(day / "index_20260102.parquet")
    con = data.connect(root)
    assert data.param_names(
        "select * from index_daily where date <= :end and close::double > :lo"
    ) == ["end", "lo"]
    df = data.run_query(
        con,
        "select index_name from index_daily where date <= :end and close > :lo order by 1",
        {"end": date(2026, 1, 31), "lo": 1.5},
    )
    assert df["index_name"].to_list() == ["Nifty Bank"]
    with pytest.raises(ValueError, match="without values"):
        data.run_query(con, "select 1 where :x = 1", {})
    assert data.resolve_params({"end": "2026-01-01"}, {"end": "2026-02-01"}) == {
        "end": "2026-02-01"
    }
    with pytest.raises(ValueError, match="not declared"):
        data.resolve_params({"end": "x"}, {"start": "y"})
    h1 = data.frame_hash(df, ("index_name",))
    assert h1 == data.frame_hash(df.reverse(), ("index_name",)) and len(h1) == 64


def test_batched_fit_matches_in_memory(tmp_path: Path) -> None:
    """The streaming path: the schema completed from the file equals the one a whole-frame prepare
    learns; the chunks cover every row; the ridge from chunks equals scikit-learn's Ridge on the
    whole imputed matrix; xgboost from many chunks agrees with xgboost from one."""
    from sklearn.linear_model import Ridge  # pyright: ignore[reportMissingTypeStubs]

    f = frame(n=500).with_columns(
        pl.when(pl.col("ret_5d") > 0.03).then(None).otherwise(pl.col("ret_5d")).alias("ret_5d")
    )
    path = tmp_path / "rows.parquet"
    f.write_parquet(path, row_group_size=64)
    declared = S.declare(KEY, FEATURES, {"columns": ["fwd"], "task": "regression"})
    whole = data.prepare(data.read_rows(path, declared, fitting=True), declared, fitting=True)
    completed = data.complete_schema(path, declared)
    assert completed == whole.schema and completed.features[1].levels == ("bank", "it", "pharma")
    chunks = list(data.iter_batches(path, completed, batch_rows=37))
    assert len(chunks) > 5 and sum(len(x) for x, _ in chunks) == 500
    assert all(x.dtype == np.float32 and y.dtype == np.float32 for x, y in chunks)
    assert data.count_rows(path) == 500

    def batches() -> object:
        return data.iter_batches(path, completed, batch_rows=37)

    lin = E.REGISTRY["sklearn.linear"]
    params = E.resolve_params("sklearn.linear", {"alpha": 0.5})
    streamed = lin.fit_batches(batches, completed, params, "cpu", 1)  # type: ignore[arg-type]
    whole_x = whole.X.astype("float64")
    med = np.nanmedian(whole_x, axis=0)
    filled = np.where(np.isnan(whole_x), med, whole_x)
    assert whole.Y is not None
    ref: Any = Ridge(alpha=0.5).fit(filled, whole.Y[:, 0])  # pyright: ignore[reportUnknownMemberType]
    expected: np.ndarray = np.asarray(ref.predict(filled), dtype="float64")
    np.testing.assert_allclose(
        streamed.predict(whole.X)["pred_fwd"], expected, rtol=1e-5, atol=1e-7
    )
    one = lin.fit(whole.X, whole.Y, completed, params, "cpu", 1)
    np.testing.assert_allclose(
        one.predict(whole.X)["pred_fwd"], streamed.predict(whole.X)["pred_fwd"], rtol=1e-6
    )

    xg = E.REGISTRY["xgboost"]
    xparams = E.resolve_params("xgboost", {"n_estimators": 40, "max_depth": 3})
    many = xg.fit_batches(batches, completed, xparams, "cpu", 7)  # type: ignore[arg-type]
    single = xg.fit(whole.X, whole.Y, completed, xparams, "cpu", 7)
    a, b = many.predict(whole.X)["pred_fwd"], single.predict(whole.X)["pred_fwd"]
    assert np.corrcoef(a, b)[0, 1] > 0.98 and np.corrcoef(a, whole.Y[:, 0])[0, 1] > 0.8
    d = tmp_path / "xgb"
    d.mkdir()
    artifacts.save(d, many, completed, {"algorithm": "xgboost"})
    np.testing.assert_allclose(artifacts.load(d)[0].predict(whole.X)["pred_fwd"], a, rtol=1e-6)


def test_fit_job_streams_the_file(tmp_path: Path) -> None:
    from finance.ml import fitting

    path = tmp_path / "rows.parquet"
    frame(n=300).write_parquet(path, row_group_size=50)
    declared = S.declare(KEY, FEATURES, {"columns": ["fwd"], "task": "regression"})
    job = fitting.FitJob(
        rows_path=str(path),
        schema=declared.to_dict(),
        algorithm="sklearn.linear",
        params=E.resolve_params("sklearn.linear", {}),
        device="cpu",
        seed=1,
        out_dir=str(tmp_path / "model"),
        info={"fit_name": "t"},
        batch_rows=50,
    )
    done = fitting.fit_job(job)
    assert done.rows == 300 and done.device == "cpu" and len(done.sha256) == 64
    fitted, schema, info = artifacts.load(tmp_path / "model")
    assert schema.features[1].levels == ("bank", "it", "pharma")
    assert info["rows"] == 300 and info["batch_rows"] == 50
    assert list(fitted.predict(np.zeros((2, 2), dtype="float32"))) == ["pred_fwd"]


def test_fit_job_falls_back_to_cpu_on_cuda_out_of_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from finance.ml import fitting

    path = tmp_path / "rows.parquet"
    frame(n=120).write_parquet(path)
    declared = S.declare(KEY, FEATURES, {"columns": ["fwd"], "task": "regression"})
    adapter = E.REGISTRY["xgboost"]
    real = adapter.fit_batches
    calls: list[str] = []

    def flaky(batches: Any, schema: Any, params: Any, device: str, seed: int) -> Any:
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("XGBoostError: cudaErrorMemoryAllocation: out of memory")
        return real(batches, schema, params, device, seed)

    monkeypatch.setattr(adapter, "fit_batches", flaky)

    def cuda(adapter: type[Any], wanted: str) -> str:
        return "cuda"

    monkeypatch.setattr(E, "pick_device", cuda)
    job = fitting.FitJob(
        rows_path=str(path),
        schema=declared.to_dict(),
        algorithm="xgboost",
        params=E.resolve_params("xgboost", {"n_estimators": 5}),
        device="auto",
        seed=1,
        out_dir=str(tmp_path / "m"),
        info={},
    )
    done = fitting.fit_job(job)
    assert calls == ["cuda", "cpu"] and done.device == "cpu"
    assert artifacts.load(tmp_path / "m")[2]["device"] == "cpu"


def test_ic_groups_by_a_list_of_columns() -> None:
    from datetime import datetime

    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for d in range(30):
        for h in (1, 5):
            for _ in range(25):
                y = rng.normal()
                rows.append(
                    {
                        "ts": datetime(2026, 1, 1 + d),
                        "h": h,
                        "y": y,
                        "pred_y": y * (1 if h == 1 else -1) + rng.normal(),
                    }
                )
    t = pl.DataFrame(rows)
    pos = metrics.evaluate(t.filter(pl.col("h") == 1), ["y"], ["ic"], {"group": ["ts", "h"]})[
        "ic.y"
    ]
    both = metrics.evaluate(t, ["y"], ["ic", "ic_points"], {"group": ["ts", "h"]})
    assert (
        pos > 0.5 and abs(both["ic.y"]) < 0.15 and both["ic_points.y"] == 60
    )  # opposite signs cancel
    assert metrics.evaluate(t, ["y"], ["ic_points"], {"group": "ts"})["ic_points.y"] == 30
