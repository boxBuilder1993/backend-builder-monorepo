"""Walk-forward fits and the model search as reactions. See docs/finance-portfolio.md §5.

A ``WalkForward`` is one Fit, one model spec and one feature set fitted on the data up to each
window's ``train_end`` and scored on that window's test range; a ``ModelSearch`` is a grid of them
judged together. ``RunModelSearch`` creates the rows and defers one ``RunWindow`` per window, so the
worker fits one window at a time and no parent holds a transaction for hours. ``CloseWalkForward``
and ``CloseModelSearch`` run on status changes and finish the parents when their children are done.

With the owner's split (train on everything except 2026, validate on 2026 once) a walk-forward has
two windows: train to 2024-12-31 and test 2025 (the year the choices are made on), train to
2025-12-31 and test 2026 (the result). The machinery takes any list of windows.
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from backend_builder import Context, Reaction

from finance import reactions as poller
from finance.ml.entities import Fit, InferRun, ModelSearch, WalkForward, WalkForwardWindow
from finance.ml.reactions import Infer, PublishArtifact, RunFit
from finance.ml.scores import SCORE_KEYS, mean_scores, window_scores

WINDOW_BUDGET = 2 * 3600
"""A window fits one model on up to about ten million rows and scores a year: well under this."""

DEFAULT_WINDOWS: list[dict[str, str]] = [
    {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31"},
    {"train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2030-12-31"},
]


def _day(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def range_params(fit: Fit, start: date | None, end: date, purge: bool = False) -> dict[str, Any]:
    """The Fit's own names for its date range: ``start``/``end`` on the forecast Fits, ``end_date``
    on others; a Fit without an end parameter cannot be walked forward. With ``purge``, a Fit that
    has a ``max_exit`` parameter gets it set to ``end`` too, so no training target resolves after
    the cut: a 126-session target decided in December 2024 is observed in mid-2025, inside the
    test year, and a model that saw it has been shown the test period."""
    out: dict[str, Any] = {}
    end_key = next((k for k in ("end", "end_date") if k in fit.params), None)
    if end_key is None:
        raise ValueError(f"Fit {fit.name!r} has no end parameter to cut training at")
    out[end_key] = end.isoformat()
    if purge and "max_exit" in fit.params:
        out["max_exit"] = end.isoformat()
    if start is not None:
        start_key = next((k for k in ("start", "start_date") if k in fit.params), None)
        if start_key:
            out[start_key] = start.isoformat()
    return out


class RunModelSearch(Reaction):
    """Input: {"label", "fit_ids": [...], "specs": [{"algorithm", "params"}, ...],
    "feature_sets": {name: [features] | null}, "windows"?: [{"train_end", "test_start",
    "test_end"}, ...] (default: the two windows of the 2026 split)}. One WalkForward per
    (Fit × spec × feature set) with its windows queued, one deferred RunWindow per window.
    Returns in seconds."""

    async def calculate(self, ctx: Context) -> None:
        fit_ids = [str(f) for f in ctx.input["fit_ids"]]
        specs = [dict(s) for s in ctx.input["specs"]]
        feature_sets: dict[str, Any] = dict(ctx.input.get("feature_sets") or {"all": None})
        windows = [dict(w) for w in ctx.input.get("windows") or DEFAULT_WINDOWS]
        for w in windows:
            if not _day(w["train_end"]) < _day(w["test_start"]) <= _day(w["test_end"]):
                raise ValueError(f"window out of order: {w}")
        fits: dict[str, Fit] = {}
        for fid in fit_ids:
            fit = await ctx.get_by_id(Fit, fid)
            if fit is None:
                raise LookupError(f"no Fit {fid}")
            fits[fid] = fit
        for name, feats in feature_sets.items():
            if feats is not None:
                for fit in fits.values():
                    known = {str(f["name"]) for f in fit.features}
                    bad = [f for f in feats if f not in known]
                    if bad:
                        raise ValueError(f"feature set {name!r}: not in Fit {fit.name!r}: {bad}")
        search = ctx.create(
            ModelSearch,
            label=str(ctx.input["label"]),
            fit_ids=fit_ids,
            specs=specs,
            feature_sets=feature_sets,
            windows=windows,
        )
        count = 0
        for fid, fit in fits.items():
            for spec in specs:
                for name, feats in feature_sets.items():
                    wf = ctx.create(
                        WalkForward,
                        search_id=search.id,
                        fit_id=fid,
                        algorithm=str(spec["algorithm"]),
                        params=dict(spec.get("params") or {}),
                        feature_set=str(name),
                        features=[str(f) for f in feats] if feats is not None else None,
                        label=f"{search.label} / {fit.name} / {spec['algorithm']} / {name}",
                        windows=len(windows),
                    )
                    for i, w in enumerate(windows):
                        win = ctx.create(
                            WalkForwardWindow,
                            walk_forward_id=wf.id,
                            index=i,
                            train_end=_day(w["train_end"]),
                            test_start=_day(w["test_start"]),
                            test_end=_day(w["test_end"]),
                        )
                        await ctx.trigger(
                            RunWindow, {"window_id": win.id}, deferred=True, timeout=WINDOW_BUDGET
                        )
                    count += 1
        search.walk_forwards = count
        ctx.save("search_id", search.id)
        ctx.save("walk_forwards", count)
        ctx.save("windows", len(windows) * count)


class RunWindow(Reaction):
    """Deferred, one per WalkForwardWindow. Input: {"window_id"}. RunFit inline with the one spec
    and the feature subset, the query's end at train_end; then Infer on the test range with
    need_target off, so every test day gets a forecast, and sample off, so every listed duration
    does. Records the ids and the window's scores."""

    async def calculate(self, ctx: Context) -> None:
        win = await ctx.get_by_id(WalkForwardWindow, ctx.input["window_id"])
        if win is None:
            raise LookupError(f"no WalkForwardWindow {ctx.input['window_id']}")
        wf = await ctx.get_by_id(WalkForward, win.walk_forward_id)
        fit = await ctx.get_by_id(Fit, wf.fit_id) if wf else None
        if wf is None or fit is None:
            raise LookupError(f"window {win.id} without its walk-forward or Fit")
        win.status = "running"
        ctx.save("window_id", win.id)
        try:
            fitted = await ctx.trigger(
                RunFit,
                {
                    "fit_id": fit.id,
                    "params": range_params(fit, None, win.train_end, purge=True),
                    "specs": [{"algorithm": wf.algorithm, "params": wf.params}],
                    "features": wf.features,
                    "inline": True,
                },
            )
            assert fitted is not None
            win.fit_run_id = str(fitted.vars["fit_run_id"])
            models = list(fitted.vars.get("models") or [])
            if len(models) != 1:
                raise RuntimeError(f"expected one model, got {len(models)}")
            win.model_id = str(models[0])
            params = range_params(fit, win.test_start, win.test_end)
            if "need_target" in fit.params:
                params["need_target"] = False
            if "sample" in fit.params:
                params["sample"] = (
                    0  # the training rows sample durations; the forecast is the full curve
                )
            inferred = await ctx.trigger(
                Infer,
                {
                    "model_id": win.model_id,
                    "label": f"wf {wf.id} w{win.index}",
                    "query": fit.query,
                    "params": {**fit.params, **params},
                },
            )
            assert inferred is not None
            if inferred.vars.get("status") != "done":
                raise RuntimeError(f"inference failed: {inferred.vars.get('error')}")
            win.infer_run_id = str(inferred.vars["infer_run_id"])
            path = poller.data_root() / str(inferred.vars["path"])
            win.ic = await asyncio.to_thread(
                window_scores, path, fit.target["columns"][0], list(fit.key)
            )
        except Exception as exc:  # noqa: BLE001 - the window's outcome
            win.status, win.error = "failed", _err(exc)
            ctx.save("error", win.error)
            return
        win.status = "done"
        ctx.save("ic", win.ic)


class CloseWalkForward(Reaction):
    """OnChange(WalkForwardWindow, ["status"]). When every window of the walk-forward is done or
    failed: the windows' inference tables concatenated into parquet/walkforward/<id>.parquet (with
    the window index) and published, the IC summary, status done (or failed if any window failed).
    Idempotent; harmless on a closed walk-forward."""

    async def calculate(self, ctx: Context) -> None:
        win = await ctx.get_by_id(WalkForwardWindow, ctx.input["id"])
        if win is None:
            return
        wf = await ctx.get_by_id(WalkForward, win.walk_forward_id)
        if wf is None or wf.status in ("done", "failed"):
            return
        windows = await ctx.query(
            WalkForwardWindow,
            "select * from walk_forward_window where walk_forward_id = :w order by index",
            w=wf.id,
        )
        terminal = [w for w in windows if w.status in ("done", "failed")]
        if len(terminal) < wf.windows:
            ctx.save("closed", False)
            return
        failed = [w for w in windows if w.status == "failed"]
        if failed:
            wf.status, wf.error = "failed", f"{len(failed)} window(s) failed: {failed[0].error}"
            ctx.save("closed", True)
            return
        paths: list[tuple[int, str]] = []
        for w in windows:
            ir = await ctx.get_by_id(InferRun, w.infer_run_id) if w.infer_run_id else None
            if ir is None or not ir.path:
                wf.status, wf.error = "failed", f"window {w.index} has no inference table"
                ctx.save("closed", True)
                return
            paths.append((w.index, ir.path))
        rel = f"parquet/walkforward/{wf.id}.parquet"
        rows = await asyncio.to_thread(_concat, poller.data_root(), paths, rel)
        await ctx.trigger(
            PublishArtifact, {"paths": [rel], "kind": "parquet", "source": "walkforward"}
        )
        wf.predictions_path = rel
        wf.summary = summarise(rows, windows)
        wf.status = "done"
        ctx.save("closed", True)
        ctx.save("summary", wf.summary)


def _concat(root: Path, paths: list[tuple[int, str]], rel: str) -> int:
    frames = [pl.read_parquet(root / p).with_columns(pl.lit(i).alias("window")) for i, p in paths]
    out = pl.concat(frames, how="diagonal_relaxed")
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(root / rel, compression="zstd")
    return out.height


def summarise(rows: int, windows: list[WalkForwardWindow]) -> dict[str, Any]:
    """The walk-forward's summary: every window's scores, and their means over windows."""
    return {
        "rows": rows,
        "windows": [
            {
                "index": w.index,
                "test_start": w.test_start.isoformat(),
                "model_id": w.model_id,
                **(w.ic or {}),
            }  # fmt: skip
            for w in windows
        ],
        **mean_scores([w.ic for w in windows if w.ic]),
    }


class CloseModelSearch(Reaction):
    """OnChange(WalkForward, ["status"]). When every walk-forward of the search is done or failed:
    the leaderboard (per Fit, the walk-forwards ranked by the IC of the first window, the selection
    window, with every window's IC alongside) into the search row, status done."""

    async def calculate(self, ctx: Context) -> None:
        wf = await ctx.get_by_id(WalkForward, ctx.input["id"])
        if wf is None or not wf.search_id:
            return
        search = await ctx.get_by_id(ModelSearch, wf.search_id)
        if search is None or search.status in ("done", "failed"):
            return
        wfs = await ctx.query(
            WalkForward, "select * from walk_forward where search_id = :s", s=search.id
        )
        if any(w.status == "running" for w in wfs) or len(wfs) < search.walk_forwards:
            ctx.save("closed", False)
            return
        search.leaderboard = await leaderboard(ctx, search, wfs)
        search.status = "done"
        ctx.save("closed", True)
        ctx.save("leaderboard", search.leaderboard)


async def leaderboard(ctx: Context, search: ModelSearch, wfs: list[WalkForward]) -> dict[str, Any]:
    """Per Fit, the walk-forwards ranked by the Spearman IC of the first window (the selection
    window), every window's scores alongside: ``<score>_w<i>`` overall and ``<score>_w<i>_h<h>``
    per duration for the IC, hit rate and slope."""
    fits: dict[str, str] = {}
    for fid in search.fit_ids:
        fit = await ctx.get_by_id(Fit, fid)
        fits[fid] = fit.name if fit else fid
    rows: list[dict[str, Any]] = []
    for w in wfs:
        summary: dict[str, Any] = w.summary or {}
        raw_windows: list[Any] = list(summary.get("windows") or [])
        windows: list[dict[str, Any]] = (
            [dict[str, Any](x) for x in raw_windows] if w.status == "done" else []
        )
        row: dict[str, Any] = {
            "walk_forward_id": w.id,
            "fit": fits.get(w.fit_id, w.fit_id),
            "algorithm": w.algorithm,
            "params": w.params,
            "feature_set": w.feature_set,
            "status": w.status,
            "selection_ic": windows[0].get("ic") if windows else None,
        }
        for k in SCORE_KEYS[1:]:
            row[f"selection_{k}"] = windows[0].get(k) if windows else None
        for win in windows:
            for k in SCORE_KEYS:
                row[f"{k}_w{win['index']}"] = win.get(k)
            by_h: dict[str, dict[str, Any]] = dict(win.get("by_h") or {})
            for h, v in by_h.items():
                for k in ("ic", "hit", "slope"):
                    row[f"{k}_w{win['index']}_h{h}"] = v.get(k)
        rows.append(row)
    rows.sort(key=lambda r: (r["fit"], -(r["selection_ic"] or -9)))
    return {"rows": rows, "failed": sum(1 for w in wfs if w.status == "failed")}


class RescoreWalkForward(Reaction):
    """Input: {"walk_forward_id"}. Recomputes every window's scores of a done walk-forward from its
    saved predictions table and rewrites the summary. How a finished search gets a score that did
    not exist when it ran."""

    async def calculate(self, ctx: Context) -> None:
        wf = await ctx.get_by_id(WalkForward, ctx.input["walk_forward_id"])
        if wf is None:
            raise LookupError(f"no WalkForward {ctx.input['walk_forward_id']}")
        if wf.status != "done" or not wf.predictions_path:
            ctx.save("rescored", False)
            return
        fit = await ctx.get_by_id(Fit, wf.fit_id)
        if fit is None:
            raise LookupError(f"WalkForward {wf.id} without its Fit")
        windows = await ctx.query(
            WalkForwardWindow,
            "select * from walk_forward_window where walk_forward_id = :w order by index",
            w=wf.id,
        )
        path = poller.data_root() / wf.predictions_path
        scored = await asyncio.to_thread(
            _rescore, path, fit.target["columns"][0], list(fit.key), [w.index for w in windows]
        )
        for w in windows:
            w.ic = scored[w.index]
        wf.summary = summarise(int((wf.summary or {}).get("rows") or 0), windows)
        ctx.save("rescored", True)
        ctx.save("summary", wf.summary)


def _rescore(path: Path, target: str, key: list[str], indexes: list[int]) -> dict[int, Any]:
    table = pl.read_parquet(path)
    out: dict[int, Any] = {}
    for i in indexes:
        part = table.filter(pl.col("window") == i)
        tmp = path.with_suffix(f".w{i}.tmp.parquet")
        part.write_parquet(tmp)
        try:
            out[i] = window_scores(tmp, target, key)
        finally:
            tmp.unlink(missing_ok=True)
    return out


class RescoreModelSearch(Reaction):
    """Input: {"search_id"}. One deferred RescoreWalkForward per done walk-forward of the search,
    then a deferred RebuildLeaderboard. Returns in seconds; the worker does the rest."""

    async def calculate(self, ctx: Context) -> None:
        search = await ctx.get_by_id(ModelSearch, ctx.input["search_id"])
        if search is None:
            raise LookupError(f"no ModelSearch {ctx.input['search_id']}")
        wfs = await ctx.query(
            WalkForward,
            "select * from walk_forward where search_id = :s and status = 'done'",
            s=search.id,
        )
        for w in wfs:
            await ctx.trigger(
                RescoreWalkForward, {"walk_forward_id": w.id}, deferred=True, timeout=1800
            )
        await ctx.trigger(RebuildLeaderboard, {"search_id": search.id}, deferred=True, timeout=1800)
        ctx.save("walk_forwards", len(wfs))


class RebuildLeaderboard(Reaction):
    """Input: {"search_id"}. The leaderboard of a search rebuilt from its walk-forwards' current
    summaries. Deferred after the rescores, so the worker runs it once they are done."""

    async def calculate(self, ctx: Context) -> None:
        search = await ctx.get_by_id(ModelSearch, ctx.input["search_id"])
        if search is None:
            raise LookupError(f"no ModelSearch {ctx.input['search_id']}")
        wfs = await ctx.query(
            WalkForward, "select * from walk_forward where search_id = :s", s=search.id
        )
        search.leaderboard = await leaderboard(ctx, search, wfs)
        ctx.save("rows", len(search.leaderboard["rows"]))
