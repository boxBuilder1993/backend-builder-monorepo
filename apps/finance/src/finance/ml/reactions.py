"""The ML reactions. RegisterFit stores a definition; RunFit fans a FitRun out into deferred
FitModel runs; FitModel fits in a subprocess and uploads through UploadModel; CloseFitRun closes
the run when its last model lands; Infer scores a model on a query and publishes the table;
Evaluate computes
metrics over that table. LoadRows and PullStore sit under the readers. Reactions write rows; the
functions in data, fitting, artifacts, modelstore and metrics do the work and never see the
runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from backend_builder import Context, Reaction

from finance import reactions as poller
from finance import settings, store
from finance.entities import DataFile
from finance.ml import artifacts, data, fitting, metrics, modelstore, search
from finance.ml import estimators as E
from finance.ml import schema as S
from finance.ml.entities import Evaluation, Fit, FitRun, InferRun, Model
from finance.reactions import PullStore
from finance.settings import HF_REPO


def _now() -> datetime:
    return datetime.now(UTC)


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def content_hash(definition: dict[str, Any]) -> str:
    fields = {k: definition[k] for k in ("query", "params", "key", "features", "target", "search")}
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RegisterFit(Reaction):
    """Input: a Fit definition (name, description?, query, params, key, features, target, search).
    Validates the declaration and the search against the adapters' Params, then stores it; an
    identical definition (same content hash) returns the existing row. Saves fit_id and created."""

    async def calculate(self, ctx: Context) -> None:
        d = ctx.input
        declared = S.declare(list(d["key"]), list(d["features"]), dict(d["target"]))
        specs = search.expand(list(d["search"]))
        params = dict(d.get("params") or {})
        used = data.param_names(d["query"])
        undeclared = [p for p in used if p not in params]
        if undeclared:
            raise ValueError(f"query uses params the Fit does not declare: {undeclared}")
        h = content_hash({**d, "params": params})
        existing = await ctx.query(Fit, "select * from fit where content_hash = :h", h=h)
        if existing:
            ctx.save("fit_id", existing[0].id)
            ctx.save("created", False)
            return
        fit = ctx.create(
            Fit,
            name=str(d["name"]),
            description=str(d.get("description", "")),
            query=str(d["query"]),
            params=params,
            key=list(declared.key),
            features=[{"name": f.name, "kind": f.kind} for f in declared.features],
            target={"columns": list(declared.target.columns), "task": declared.target.task},
            search=list(d["search"]),
            content_hash=h,
        )
        ctx.save("fit_id", fit.id)
        ctx.save("created", True)
        ctx.save("specs", len(specs))


class LoadRows(Reaction):
    """Input: {"query", "params", "key"}. PullStore first, so the mirror is at the store's revision;
    then the query through DuckDB over the mirror, written to a scratch Parquet for the caller.
    Saves rows_path, rows and data_hash. Frames never travel through vars or the outbox."""

    async def calculate(self, ctx: Context) -> None:
        pull = await ctx.trigger(PullStore, {})
        assert pull is not None
        pulled = {k: pull.vars.get(k) for k in ("status", "revision", "files", "bytes", "error")}
        ctx.save("store", pulled)
        if pulled["status"] == "failed":
            raise RuntimeError(f"store unavailable: {pulled['error']}")
        q, params = ctx.input["query"], dict(ctx.input["params"])
        path = data.work_dir() / "rows" / f"{uuid.uuid4().hex}.parquet"
        n, h = await asyncio.to_thread(_load, poller.data_root(), q, params, path)
        ctx.save("rows_path", str(path))
        ctx.save("rows", n)
        ctx.save("data_hash", h)


def _load(root: Path, q: str, params: dict[str, Any], out: Path) -> tuple[int, str]:
    """DuckDB writes the Parquet and hashes it; no frame of the rows exists in this process."""
    con = data.connect(root)
    try:
        data.tune(con)
        n = data.write_rows(con, q, params, out)
    finally:
        con.close()
    return n, data.file_hash(out)


class RunFit(Reaction):
    """Input: {"fit_id", "params"?}. One FitRun, the rows loaded once for the record, one Model row
    per spec as queued, one deferred FitModel per spec with the algorithm's budget. Returns in
    seconds; the console shows the fan-out before a single fit starts. The run is `running` from
    here and only CloseFitRun changes it: concurrent FitModel runs never touch the FitRun row."""

    async def calculate(self, ctx: Context) -> None:
        fit = await ctx.get_by_id(Fit, ctx.input["fit_id"])
        if fit is None:
            raise LookupError(f"no Fit {ctx.input['fit_id']}")
        params = data.resolve_params(fit.params, ctx.input.get("params"))
        specs = search.expand(fit.search)
        run = ctx.create(
            FitRun,
            fit_id=fit.id,
            params=params,
            specs=len(specs),
            status="running",
            started_at=_now(),
        )
        loaded = await ctx.trigger(LoadRows, {"query": fit.query, "params": params, "key": fit.key})
        assert loaded is not None
        run.rows = int(loaded.vars.get("rows", 0))
        run.data_hash = loaded.vars.get("data_hash")
        rows_path = loaded.vars.get("rows_path")
        models: list[str] = []
        for spec in specs:
            m = ctx.create(
                Model,
                fit_run_id=run.id,
                fit_id=fit.id,
                algorithm=spec.algorithm,
                params=spec.params,
            )
            budget = E.REGISTRY[spec.algorithm].budget
            await ctx.trigger(
                FitModel,
                {"model_id": m.id, "device": spec.device, "rows_path": rows_path},
                deferred=True,
                timeout=budget,
            )
            models.append(m.id)
        ctx.save("fit_run_id", run.id)
        ctx.save("models", models)
        ctx.save("rows", run.rows)


class FitModel(Reaction):
    """Deferred, one per Model. Input: {"model_id", "device", "rows_path"?}. Loads the rows (the
    scratch file from RunFit when it is still on this host and its hash matches, else again through
    LoadRows), fits in a subprocess, writes the directory into the cache, then UploadModel.
    Statuses: fitting, uploading, fitted; a failure anywhere is the model's outcome, failed with the
    error. Writes only its own Model row, so a sweep's models never conflict on the FitRun."""

    async def calculate(self, ctx: Context) -> None:
        model = await ctx.get_by_id(Model, ctx.input["model_id"])
        if model is None:
            raise LookupError(f"no Model {ctx.input['model_id']}")
        run = await ctx.get_by_id(FitRun, model.fit_run_id)
        fit = await ctx.get_by_id(Fit, model.fit_id)
        if run is None or fit is None:
            raise LookupError(f"Model {model.id} without its run or fit")
        model.status = "fitting"
        ctx.save("model_id", model.id)
        try:
            rows_path = await self._rows(ctx, fit, run, ctx.input.get("rows_path"))
            declared = S.declare(fit.key, fit.features, fit.target)
            out_dir = data.work_dir() / "models" / model.id
            job = fitting.FitJob(
                rows_path=rows_path,
                schema=declared.to_dict(),
                algorithm=model.algorithm,
                params=model.params,
                device=str(ctx.input.get("device", "auto")),
                seed=E.DEFAULT_SEED,
                out_dir=str(out_dir),
                info={
                    "model_id": model.id,
                    "fit_run_id": run.id,
                    "fit_id": fit.id,
                    "fit_name": fit.name,
                },
            )
            done = await asyncio.to_thread(fitting.fit_in_subprocess, job)
        except Exception as exc:  # noqa: BLE001 - the model's outcome, not the sweep's
            model.status, model.error = "failed", _err(exc)
            ctx.save("status", "failed")
            ctx.save("error", model.error)
            return
        model.schema = done.schema
        model.device = done.device
        model.fit_seconds = round(done.seconds, 2)
        model.sha256, model.bytes = done.sha256, done.bytes
        model.cache_path = job.out_dir
        model.status = "uploading"
        uploaded = await ctx.trigger(UploadModel, {"model_id": model.id})
        assert uploaded is not None
        for key in ("upload", "revision", "error"):  # the upload's outcome, lifted for the caller
            if key in uploaded.vars:
                ctx.save(key, uploaded.vars[key])
        ctx.save("status", model.status)

    @staticmethod
    async def _rows(ctx: Context, fit: Fit, run: FitRun, hint: str | None) -> str:
        if hint and Path(hint).exists():
            if await asyncio.to_thread(data.file_hash, Path(hint)) == run.data_hash:
                return hint
        loaded = await ctx.trigger(
            LoadRows, {"query": fit.query, "params": run.params, "key": fit.key}
        )
        assert loaded is not None
        path = str(loaded.load("rows_path"))
        h = loaded.vars.get("data_hash")
        if run.data_hash and h != run.data_hash:
            raise RuntimeError("the data changed since the run started; start a new FitRun")
        return path


class UploadModel(Reaction):
    """Input: {"model_id"}. The cached directory to models/<id>/ in the store; records repo, path,
    revision; the model is fitted only then. A store outage leaves it uploading with its cache path,
    an outcome, and the upload can be triggered again by hand."""

    async def calculate(self, ctx: Context) -> None:
        model = await ctx.get_by_id(Model, ctx.input["model_id"])
        if model is None or not model.cache_path:
            raise LookupError(f"Model {ctx.input['model_id']} has nothing to upload")
        if not settings.PUBLISH:
            model.status = "fitted"
            model.repo, model.path = None, None
            ctx.save("upload", "disabled")
            return
        token = ctx.secret("hf_token")
        try:
            revision = await asyncio.to_thread(
                modelstore.upload, HF_REPO, token, model.id, Path(model.cache_path)
            )
        except Exception as exc:  # noqa: BLE001 - the store being down is an outcome
            model.error = _err(exc)
            ctx.save("upload", "failed")
            ctx.save("error", model.error)
            return
        model.repo, model.path, model.revision = HF_REPO, modelstore.model_path(model.id), revision
        model.status, model.error = "fitted", None
        ctx.save("upload", "ok")
        ctx.save("revision", revision)


class CloseFitRun(Reaction):
    """OnChange(Model, ["status"]). When every model of the run is fitted or failed, the run is
    done, or failed when all failed, with finished_at. Idempotent; harmless on a closed run."""

    async def calculate(self, ctx: Context) -> None:
        model = await ctx.get_by_id(Model, ctx.input["id"])
        if model is None:
            return
        run = await ctx.get_by_id(FitRun, model.fit_run_id)
        if run is None or run.status in ("done", "failed"):
            return
        models = await ctx.query(Model, "select * from model where fit_run_id = :r", r=run.id)
        terminal = [m for m in models if m.status in ("fitted", "failed")]
        if len(terminal) < run.specs:
            ctx.save("closed", False)
            return
        run.status = "failed" if all(m.status == "failed" for m in terminal) else "done"
        run.finished_at = _now()
        ctx.save("closed", True)
        ctx.save("fit_run_status", run.status)


class Infer(Reaction):
    """Input: {"model_id", "label", "query", "params"?}. The model from the cache or the store by
    (repo, path, revision); LoadRows against the model's schema; score; the table of key, truth when
    present, pred_* and metadata as Parquet in the mirror; PublishArtifact. An InferRun row
    throughout."""

    async def calculate(self, ctx: Context) -> None:
        model = await ctx.get_by_id(Model, ctx.input["model_id"])
        if model is None or model.status != "fitted" or model.schema is None:
            raise LookupError(f"Model {ctx.input['model_id']} is not a fitted model")
        schema = S.Schema.from_dict(model.schema)
        params = dict(ctx.input.get("params") or {})
        run = ctx.create(
            InferRun,
            model_id=model.id,
            label=str(ctx.input["label"]),
            query=str(ctx.input["query"]),
            params=params,
        )
        ctx.save("infer_run_id", run.id)
        try:
            directory = await asyncio.to_thread(self._model_dir, ctx.secret, model)
            loaded = await ctx.trigger(
                LoadRows, {"query": run.query, "params": params, "key": list(schema.key)}
            )
            assert loaded is not None
            rows_path = str(loaded.load("rows_path"))
            rel = f"parquet/inference/{run.id}.parquet"
            n, unknown = await asyncio.to_thread(
                _score, directory, schema, rows_path, poller.data_root() / rel, model.id, run.id
            )
        except Exception as exc:  # noqa: BLE001 - the run's outcome
            run.status, run.error = "failed", _err(exc)
            ctx.save("status", "failed")
            ctx.save("error", run.error)
            return
        run.rows, run.unknown_levels, run.path, run.status = n, unknown, rel, "done"
        await ctx.trigger(
            PublishArtifact, {"paths": [rel], "kind": "inference", "source": "inference"}
        )
        ctx.save("status", "done")
        ctx.save("rows", n)
        ctx.save("path", rel)

    @staticmethod
    def _model_dir(secret: Any, model: Model) -> Path:
        cached = data.work_dir() / "models" / model.id
        if (cached / "schema.json").exists():
            return cached
        if not (model.repo and model.path and model.revision):
            raise LookupError(f"Model {model.id} is neither cached here nor in the store")
        token = secret("hf_token")
        return modelstore.fetch(
            model.repo, token, model.path, model.revision, data.work_dir() / "fetched"
        )


def _score(
    directory: Path, schema: S.Schema, rows_path: str, out: Path, model_id: str, run_id: str
) -> tuple[int, dict[str, int]]:
    fitted, _, _ = artifacts.load(directory)
    frame = data.read_rows(Path(rows_path), schema, fitting=False)
    prepared = data.prepare(frame, schema, fitting=False)
    del frame
    preds = fitted.predict(prepared.X)
    table = prepared.keys
    if prepared.truth is not None:
        table = table.hstack(prepared.truth)
    if prepared.extra is not None:
        table = table.hstack(prepared.extra)
    for name in E.output_columns(schema):
        table = table.with_columns(pl.Series(name, preds[name]))
    table = table.with_columns(
        pl.lit(model_id).alias("model_id"),
        pl.lit(run_id).alias("infer_run_id"),
        pl.lit(_now()).alias("scored_at"),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out, compression="zstd")
    return table.height, prepared.unknown_levels


class PublishArtifact(Reaction):
    """Input: {"paths": [...], "kind", "source"}. Files already in the mirror: upload them to the
    store (unless FINANCE_PUBLISH=off) and write their DataFile rows, stamped when uploaded. A store
    outage leaves the rows unstamped for the nightly Publish."""

    async def calculate(self, ctx: Context) -> None:
        paths: list[str] = list(ctx.input["paths"])
        root = poller.data_root()
        now = _now()
        stamped: datetime | None = None
        if settings.PUBLISH:
            token = ctx.secret("hf_token")
            try:
                await asyncio.to_thread(
                    store.upload, root, paths, HF_REPO, token, "data: inference"
                )
                stamped = now
            except Exception as exc:  # noqa: BLE001
                ctx.save("error", _err(exc))
        for rel in paths:
            p = root / rel
            ctx.create(
                DataFile,
                path=rel,
                kind=str(ctx.input["kind"]),
                source=str(ctx.input["source"]),
                status="ok",
                bytes=p.stat().st_size,
                sha256=store.sha256(p),
                written_at=now,
                checked_at=now,
                published_at=stamped,
            )
        ctx.save("published", stamped is not None)


class Evaluate(Reaction):
    """Input: {"infer_run_id", "metrics": [...], "params"?}. Reads the run's table from the mirror,
    computes each metric per target column, writes an Evaluation."""

    async def calculate(self, ctx: Context) -> None:
        run = await ctx.get_by_id(InferRun, ctx.input["infer_run_id"])
        if run is None or run.status != "done" or not run.path:
            raise LookupError(f"InferRun {ctx.input['infer_run_id']} has no table")
        model = await ctx.get_by_id(Model, run.model_id)
        if model is None or model.schema is None:
            raise LookupError(f"Model {run.model_id} missing")
        names = [str(n) for n in ctx.input["metrics"]]
        params = dict(ctx.input.get("params") or {})
        ev = ctx.create(Evaluation, infer_run_id=run.id, metrics=names, params=params)
        try:
            table = await asyncio.to_thread(pl.read_parquet, poller.data_root() / run.path)
            cols = list(S.Schema.from_dict(model.schema).target.columns)
            ev.results = metrics.evaluate(table, cols, names, params)
        except Exception as exc:  # noqa: BLE001
            ev.status, ev.error = "failed", _err(exc)
            ctx.save("error", ev.error)
        ctx.save("evaluation_id", ev.id)
        ctx.save("results", ev.results)
