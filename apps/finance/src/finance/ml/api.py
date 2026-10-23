"""The ML endpoints: register a Fit, run it, read a run, score a model, evaluate a run."""

from __future__ import annotations

from typing import Any

from backend_builder import Context, Endpoint, Reaction, Reply
from pydantic import BaseModel

from finance.ml.entities import FitRun, Model
from finance.ml.reactions import Evaluate, Infer, RegisterFit, RunFit


class FitBody(BaseModel):
    name: str
    description: str = ""
    query: str
    params: dict[str, Any] = {}
    key: list[str]
    features: list[dict[str, Any]]
    target: dict[str, Any]
    search: list[dict[str, Any]]


class FitOut(BaseModel):
    fit_id: str
    created: bool


class FitPath(BaseModel):
    fit_id: str


class RunBody(BaseModel):
    params: dict[str, Any] = {}


class RunOut(BaseModel):
    fit_run_id: str
    models: list[str]
    rows: int


class RunPath(BaseModel):
    fit_run_id: str


class ModelOut(BaseModel):
    id: str
    algorithm: str
    params: dict[str, Any]
    status: str
    revision: str | None
    fit_seconds: float | None
    device: str | None
    error: str | None


class RunView(BaseModel):
    fit_run_id: str
    status: str
    specs: int
    rows: int
    models: list[ModelOut]


class ModelPath(BaseModel):
    model_id: str


class InferBody(BaseModel):
    label: str
    query: str
    params: dict[str, Any] = {}


class InferOut(BaseModel):
    infer_run_id: str
    status: str
    rows: int | None = None
    path: str | None = None
    error: str | None = None


class InferRunPath(BaseModel):
    infer_run_id: str


class EvaluateBody(BaseModel):
    metrics: list[str]
    params: dict[str, Any] = {}


class EvaluateOut(BaseModel):
    evaluation_id: str
    results: dict[str, float] | None
    error: str | None = None


class ErrorOut(BaseModel):
    error: str


class RegisterFitApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(RegisterFit, dict(ctx.input["body"]))


class RunFitApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(
            RunFit, {"fit_id": ctx.input["path"]["fit_id"], "params": ctx.input["body"]["params"]}
        )


class GetRunApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        run = await ctx.get_by_id(FitRun, ctx.input["path"]["fit_run_id"])
        if run is None:
            ctx.save("not_found", ctx.input["path"]["fit_run_id"])
            return
        models = await ctx.query(
            Model, "select * from model where fit_run_id = :r order by created_at", r=run.id
        )
        ctx.save("run", run)
        ctx.save("models", models)


class InferApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(Infer, {"model_id": ctx.input["path"]["model_id"], **ctx.input["body"]})


class EvaluateApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(
            Evaluate, {"infer_run_id": ctx.input["path"]["infer_run_id"], **ctx.input["body"]}
        )


async def fit_out(ctx: Context) -> FitOut:
    return FitOut(fit_id=ctx.load("fit_id"), created=ctx.load("created"))


async def run_out(ctx: Context) -> RunOut:
    return RunOut(
        fit_run_id=ctx.load("fit_run_id"), models=ctx.load("models"), rows=ctx.load("rows")
    )


async def run_view(ctx: Context) -> RunView | Reply:
    if "not_found" in ctx.vars:
        return Reply(status=404, body=ErrorOut(error=f"no FitRun {ctx.load('not_found')}"))
    run: FitRun = ctx.load("run")
    models: list[Model] = ctx.load("models")
    return RunView(
        fit_run_id=run.id,
        status=run.status,
        specs=run.specs,
        rows=run.rows,
        models=[
            ModelOut(
                id=m.id,
                algorithm=m.algorithm,
                params=m.params,
                status=m.status,
                revision=m.revision,
                fit_seconds=m.fit_seconds,
                device=m.device,
                error=m.error,
            )  # fmt: skip
            for m in models
        ],
    )


async def infer_out(ctx: Context) -> InferOut:
    return InferOut(
        infer_run_id=ctx.load("infer_run_id"),
        status=ctx.load("status"),
        rows=ctx.vars.get("rows"),
        path=ctx.vars.get("path"),
        error=ctx.vars.get("error"),
    )


async def evaluate_out(ctx: Context) -> EvaluateOut:
    return EvaluateOut(
        evaluation_id=ctx.load("evaluation_id"),
        results=ctx.vars.get("results"),
        error=ctx.vars.get("error"),
    )


endpoints = [
    Endpoint(
        method="POST",
        path="/ml/fits",
        reaction=RegisterFitApi,
        body=FitBody,
        response=FitOut,
        build=fit_out,
    ),
    Endpoint(
        method="POST",
        path="/ml/fits/{fit_id}/runs",
        reaction=RunFitApi,
        path_model=FitPath,
        body=RunBody,
        response=RunOut,
        build=run_out,
    ),  # fmt: skip
    Endpoint(
        method="GET",
        path="/ml/runs/{fit_run_id}",
        reaction=GetRunApi,
        path_model=RunPath,
        response=RunView,
        responses={404: ErrorOut},
        build=run_view,
    ),  # fmt: skip
    Endpoint(
        method="POST",
        path="/ml/models/{model_id}/infer",
        reaction=InferApi,
        path_model=ModelPath,
        body=InferBody,
        response=InferOut,
        build=infer_out,
        timeout=1800,
    ),  # fmt: skip
    Endpoint(
        method="POST",
        path="/ml/infer-runs/{infer_run_id}/evaluate",
        reaction=EvaluateApi,
        path_model=InferRunPath,
        body=EvaluateBody,
        response=EvaluateOut,
        build=evaluate_out,
    ),  # fmt: skip
]
