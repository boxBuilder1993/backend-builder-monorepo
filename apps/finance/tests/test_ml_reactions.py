"""The ML reactions on Postgres with a real linear estimator on a tiny mirror, the store faked:
register, run, fit, upload, close, infer, publish, evaluate; and the outcomes when things fail."""

# ruff: noqa: E501  (test fixtures read better on one line)

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from backend_builder import PostgresStore, Worker, run
from finance import mirror
from finance.ml import data, modelstore
from finance.ml.reactions import Evaluate, FitModel, Infer, LoadRows, RegisterFit, RunFit

KEY = ["symbol", "date"]
DEFINITION: dict[str, Any] = {
    "name": "toy",
    "description": "a toy fit on the index table",
    "query": "select index_name as symbol, date, close, lag_close, fwd from toy where date <= :end_date",
    "params": {"end_date": "2026-01-20"},
    "key": KEY,
    "features": [{"name": "close", "kind": "numeric"}, {"name": "lag_close", "kind": "numeric"}],
    "target": {"columns": ["fwd"], "task": "regression"},
    "search": [{"algorithm": "sklearn.linear", "params": {"alpha": [0.1, 10.0]}}],
}


def toy_frame(n_days: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for i in range(n_days):
        for name in ("A", "B", "C"):
            close = 100 + i + rng.normal(0, 1)
            rows.append(
                {"index_name": name, "date": date(2026, 1, 1 + i % 28) if i < 28 else date(2026, 2, 1 + i - 28),
                 "close": close, "lag_close": close - 1 + rng.normal(0, 0.1), "fwd": 0.5 * close + rng.normal(0, 0.1)}
            )  # fmt: skip
    return pl.DataFrame(rows)


@pytest.fixture
def toy_mirror(data_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A mirror with one table, the store faked as current, the model store faked, the cache in tmp."""
    (data_dir / "parquet" / "toy").mkdir(parents=True)
    toy_frame().write_parquet(data_dir / "parquet" / "toy" / "toy.parquet")
    original = data.install_views

    def with_toy(con: Any, root: Path) -> list[str]:
        made = original(con, root)
        con.execute(
            f"create view toy as select * from read_parquet('{(root / 'parquet/toy/toy.parquet').as_posix()}')"
        )
        return made + ["toy"]

    monkeypatch.setattr(data, "install_views", with_toy)
    monkeypatch.setenv("SECRET_HF_TOKEN", "hf_test_token_000000000000")

    def revision(repo: str, token: str) -> str:
        return "rev1"

    def no_pull(
        repo: str, token: str, revision: str, patterns: list[str], root: Path
    ) -> mirror.Pulled:
        return mirror.Pulled(0, 0, 0.0)

    monkeypatch.setattr(mirror, "store_revision", revision)
    monkeypatch.setattr(mirror, "pull", no_pull)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    uploads: list[str] = []

    def fake_upload(repo: str, token: str, model_id: str, directory: Path) -> str:
        uploads.append(model_id)
        return f"rev-{model_id[:6]}"

    monkeypatch.setattr(modelstore, "upload", fake_upload)

    def no_publish(*args: Any, **kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr("finance.store.upload", no_publish)
    monkeypatch.setattr("finance.settings.PUBLISH", True)
    return data_dir


async def test_register_is_idempotent_and_validates(store: PostgresStore, toy_mirror: Path) -> None:
    ctx = await run(store, RegisterFit, DEFINITION)
    fit_id, created = ctx.load("fit_id"), ctx.load("created")
    assert created is True and ctx.load("specs") == 2
    again = await run(store, RegisterFit, {**DEFINITION, "name": "toy again"})
    assert again.load("fit_id") == fit_id and again.load("created") is False
    with pytest.raises(ValueError, match="does not declare"):
        await run(store, RegisterFit, {**DEFINITION, "params": {}})
    with pytest.raises(ValueError, match="unknown algorithm"):
        await run(store, RegisterFit, {**DEFINITION, "search": [{"algorithm": "nope"}]})


async def test_the_whole_chain(store: PostgresStore, toy_mirror: Path) -> None:
    fit_id = (await run(store, RegisterFit, DEFINITION)).load("fit_id")

    ctx = await run(store, RunFit, {"fit_id": fit_id, "params": {"end_date": "2026-01-15"}})
    run_id, models = ctx.load("fit_run_id"), ctx.load("models")
    assert len(models) == 2 and ctx.load("rows") == 15 * 3
    rows = await store.query_raw("select status from model where fit_run_id = :r", r=run_id)
    assert [r["status"] for r in rows] == ["queued", "queued"]
    outbox = await store.query_raw("select reaction, input from _outbox order by created_at")
    assert [o["reaction"].rsplit(".", 1)[1] for o in outbox] == ["FitModel", "FitModel"]

    from finance import app

    worker = Worker(app, store)
    assert await worker.run_once() == 2  # the two fits
    assert await worker.run_once() >= 1  # CloseFitRun through OnChange(Model.status)
    while await worker.run_once():
        pass
    rows = await store.query_raw(
        "select status, revision, device, schema from model where fit_run_id = :r", r=run_id
    )
    assert all(
        r["status"] == "fitted" and r["revision"].startswith("rev-") and r["device"] == "cpu"
        for r in rows
    )
    assert rows[0]["schema"]["features"][0] == {"name": "close", "kind": "numeric", "levels": None}
    fr = await store.query_raw(
        "select status, specs, finished_at from fit_run where id = :r", r=run_id
    )
    assert fr[0]["status"] == "done" and fr[0]["finished_at"] is not None

    model_id = models[0]
    ctx = await run(
        store, Infer,
        {"model_id": model_id, "label": "validation", "query": DEFINITION["query"], "params": {"end_date": "2026-02-12"}},
    )  # fmt: skip
    assert ctx.load("status") == "done" and ctx.load("rows") == 40 * 3
    table = pl.read_parquet(toy_mirror / ctx.load("path"))
    assert set(
        ["symbol", "date", "fwd", "pred_fwd", "model_id", "infer_run_id", "scored_at"]
    ) <= set(table.columns)
    assert np.corrcoef(table["fwd"].to_numpy(), table["pred_fwd"].to_numpy())[0, 1] > 0.99
    df = await store.query_raw(
        "select kind, source, published_at from data_file where kind = 'inference'"
    )
    assert len(df) == 1 and df[0]["published_at"] is not None

    ctx = await run(
        store, Evaluate, {"infer_run_id": ctx.load("infer_run_id"), "metrics": ["mae", "r2"]}
    )
    results = ctx.load("results")
    assert results["r2.fwd"] > 0.99 and results["mae.fwd"] < 1.0


async def test_failures_are_outcomes(
    store: PostgresStore, toy_mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_id = (
        await run(store, RegisterFit, {**DEFINITION, "search": [{"algorithm": "sklearn.linear"}]})
    ).load("fit_id")
    ctx = await run(store, RunFit, {"fit_id": fit_id})
    model_id = ctx.load("models")[0]

    def broken(repo: str, token: str, model_id: str, directory: Path) -> str:
        raise ConnectionError("huggingface.co: 503")

    monkeypatch.setattr(modelstore, "upload", broken)
    ctx = await run(store, FitModel, {"model_id": model_id, "device": "cpu"})
    assert ctx.load("status") == "uploading" and ctx.load("upload") == "failed"
    row = (
        await store.query_raw(
            "select status, cache_path, error from model where id = :m", m=model_id
        )
    )[0]
    assert row["status"] == "uploading" and row["cache_path"] and "503" in row["error"]

    with pytest.raises(LookupError, match="not a fitted model"):
        await run(
            store, Infer, {"model_id": model_id, "label": "x", "query": "select 1", "params": {}}
        )

    ctx = await run(store, FitModel, {"model_id": model_id, "device": "tpu"})
    assert ctx.load("status") == "failed" and "tpu" in ctx.load("error")


async def test_infer_refuses_a_schema_mismatch(store: PostgresStore, toy_mirror: Path) -> None:
    fit_id = (
        await run(store, RegisterFit, {**DEFINITION, "search": [{"algorithm": "sklearn.linear"}]})
    ).load("fit_id")
    model_id = (await run(store, RunFit, {"fit_id": fit_id})).load("models")[0]
    ctx = await run(store, FitModel, {"model_id": model_id, "device": "cpu"})
    assert ctx.load("status") == "fitted"
    ctx = await run(
        store, Infer,
        {"model_id": model_id, "label": "bad", "query": "select index_name as symbol, date, close from toy", "params": {}},
    )  # fmt: skip
    assert ctx.load("status") == "failed" and "lag_close" in ctx.load("error")


async def test_screen_runs_as_a_reaction_and_records_its_table(
    store: PostgresStore, toy_mirror: Path, tmp_path: Path
) -> None:
    from finance.ml import forecast
    from finance.ml.reactions import Screen
    from test_forecast import (
        DAYS,
        synthetic_mirror,
    )  # the tests directory is on the path under pytest

    synthetic_mirror(toy_mirror)  # the synthetic minute, daily, reference, US and commodity tables
    con = data.connect(toy_mirror)
    params = {
        **forecast.PARAMS,
        "start": DAYS[0].isoformat(),
        "end": DAYS[-1].isoformat(),
        "min_turnover": 1.0,
    }
    rows = tmp_path / "rows.parquet"
    assert data.write_rows(con, forecast.ROWS_SQL, params, rows) > 0
    ctx = await run(store, Screen, {"grain": "minute", "horizon": 15, "rows_path": str(rows)})
    table = ctx.load("table")
    assert table.startswith("| feature |") and "| r5 |" in table and "| dow |" in table
    assert ctx.load("rows_path") == str(rows) and ctx.load("horizon") == 15
    path = ctx.load("path")
    assert path.startswith("parquet/screens/") and (toy_mirror / path).exists()
    df = pl.read_parquet(toy_mirror / path)
    assert set(df.columns) >= {
        "feature",
        "year",
        "ic",
        "se",
        "points",
        "grain",
        "market",
        "horizon",
    }
    ledger = await store.query_raw(
        "select source, kind, published_at from data_file where source = 'screens'"
    )
    assert (
        len(ledger) == 1
        and ledger[0]["kind"] == "parquet"
        and ledger[0]["published_at"] is not None
    )


async def test_screen_sweep_runs_each_spec_and_records_failures(
    store: PostgresStore, toy_mirror: Path, tmp_path: Path
) -> None:
    from finance.ml import forecast
    from finance.ml.reactions import ScreenSweep
    from test_forecast import DAYS, synthetic_mirror

    synthetic_mirror(toy_mirror)
    con = data.connect(toy_mirror)
    params = {
        **forecast.PARAMS,
        "start": DAYS[0].isoformat(),
        "end": DAYS[-1].isoformat(),
        "min_turnover": 1.0,
    }
    rows = tmp_path / "rows.parquet"
    data.write_rows(con, forecast.ROWS_SQL, params, rows)
    ctx = await run(
        store,
        ScreenSweep,
        {
            "specs": [
                {"grain": "minute", "horizon": 15, "rows_path": str(rows)},
                {
                    "grain": "minute",
                    "horizon": 60,
                    "rows_path": str(tmp_path / "missing.parquet"),
                    "params": {"min_turnover": "not a number"},
                },
            ]
        },
    )
    results = ctx.load("results")
    assert ctx.load("screens") == 2 and results[0]["spec"] == "minute nse 15" and results[0]["path"]
    assert results[1]["spec"] == "minute nse 60" and "error" in results[1]


async def test_model_search_walks_forward_and_closes(
    store: PostgresStore, toy_mirror: Path
) -> None:
    """Two specs × one feature subset × two windows on the toy Fit: each window fits inline and
    scores its test range; the closers finish the walk-forwards and the search with a leaderboard."""
    from finance import app
    from finance.ml.entities import ModelSearch, WalkForward, WalkForwardWindow
    from finance.ml.walkforward import RunModelSearch

    fit_id = (await run(store, RegisterFit, DEFINITION)).load("fit_id")
    windows = [
        {"train_end": "2026-01-15", "test_start": "2026-01-16", "test_end": "2026-01-24"},
        {"train_end": "2026-01-24", "test_start": "2026-01-25", "test_end": "2026-02-12"},
    ]
    ctx = await run(
        store,
        RunModelSearch,
        {
            "label": "toy search",
            "fit_ids": [fit_id],
            "specs": [
                {"algorithm": "sklearn.linear", "params": {"alpha": 0.1}},
                {"algorithm": "sklearn.linear", "params": {"alpha": 10.0}},
            ],
            "feature_sets": {"all": None, "close_only": ["close"]},
            "windows": windows,
        },
    )
    assert ctx.load("walk_forwards") == 4 and ctx.load("windows") == 8
    search_id = ctx.load("search_id")
    wfs = await store.query(
        WalkForward, "select * from walk_forward where search_id = :s", s=search_id
    )
    assert len(wfs) == 4 and {w.status for w in wfs} == {"running"}
    wins = await store.query(
        WalkForwardWindow, "select * from walk_forward_window order by walk_forward_id, index"
    )
    assert len(wins) == 8 and {w.status for w in wins} == {"queued"}
    worker = Worker(app, store)
    while await worker.run_once():
        pass
    wins = await store.query(
        WalkForwardWindow, "select * from walk_forward_window order by walk_forward_id, index"
    )
    assert {w.status for w in wins} == {"done"}, [w.error for w in wins]
    assert all(w.fit_run_id and w.model_id and w.infer_run_id and w.ic for w in wins)
    # the window's model was fitted on rows up to train_end only
    fr = await store.query_raw(
        "select params, features from fit_run where id = :r", r=wins[0].fit_run_id
    )
    assert fr[0]["params"]["end_date"] in ("2026-01-15", "2026-01-24")
    subset = [w for w in wfs if w.feature_set == "close_only"][0]
    sub_win = [w for w in wins if w.walk_forward_id == subset.id][0]
    fr = await store.query_raw("select features from fit_run where id = :r", r=sub_win.fit_run_id)
    assert fr[0]["features"] == ["close"]
    m = await store.query_raw("select schema from model where id = :m", m=sub_win.model_id)
    assert [f["name"] for f in m[0]["schema"]["features"]] == ["close"]
    wfs = await store.query(
        WalkForward, "select * from walk_forward where search_id = :s", s=search_id
    )
    assert {w.status for w in wfs} == {"done"}, [w.error for w in wfs]
    one = wfs[0]
    assert one.predictions_path and one.summary and len(one.summary["windows"]) == 2
    table = pl.read_parquet(toy_mirror / one.predictions_path)
    assert (
        set(table["window"].to_list()) == {0, 1}
        and "pred_fwd" in table.columns
        and table.height > 0
    )
    s = (await store.query(ModelSearch, "select * from model_search where id = :s", s=search_id))[0]
    assert s.status == "done" and s.leaderboard and len(s.leaderboard["rows"]) == 4
    assert s.leaderboard["failed"] == 0 and "selection_ic" in s.leaderboard["rows"][0]
    assert "ic_w0" in s.leaderboard["rows"][0] and "ic_w1" in s.leaderboard["rows"][0]


async def test_load_rows_reuses_the_file_for_the_same_query_at_the_same_revision(
    store: PostgresStore, toy_mirror: Path
) -> None:
    inp = {"query": DEFINITION["query"], "params": DEFINITION["params"], "key": DEFINITION["key"]}
    first = await run(store, LoadRows, inp)
    second = await run(store, LoadRows, inp)
    assert first.load("cached") is False and second.load("cached") is True
    assert second.load("rows_path") == first.load("rows_path")
    assert second.load("rows") == first.load("rows") > 0
    assert second.load("data_hash") == first.load("data_hash")
    # other parameters are other rows, in another file
    other = await run(store, LoadRows, {**inp, "params": {"end_date": "2026-01-10"}})
    assert other.load("cached") is False and other.load("rows_path") != first.load("rows_path")
    assert other.load("rows") < first.load("rows")


def test_range_params_purges_targets_past_the_cut_only_where_the_fit_can() -> None:
    from finance.ml.entities import Fit
    from finance.ml.walkforward import range_params

    def fit(params: dict[str, Any]) -> Fit:
        return Fit(
            name="f", query="q", params=params, key=[], features=[], target={}, search=[],
            content_hash="x",
        )  # fmt: skip

    pooled = fit({"start": "2022-09-01", "end": "2030-12-31", "max_exit": "2030-12-31"})
    cut = date(2024, 12, 31)
    assert range_params(pooled, None, cut, purge=True) == {
        "end": "2024-12-31",
        "max_exit": "2024-12-31",
    }
    # the test range leaves the exits open, so every row whose truth is known is scored
    assert range_params(pooled, date(2025, 1, 1), date(2025, 12, 31)) == {
        "start": "2025-01-01",
        "end": "2025-12-31",
    }
    assert range_params(fit({"end_date": "2026-01-20"}), None, cut, purge=True) == {
        "end_date": "2024-12-31"
    }
    with pytest.raises(ValueError, match="no end parameter"):
        range_params(fit({"start": "2022-09-01"}), None, cut)
