"""HTTP layer: endpoints that trigger reactions, on FastAPI.

An ``Endpoint`` binds a method and path to a reaction. The request's four
sources, path parameters, query string, body and headers, are each declared
as their own Pydantic model, each optional on any method. FastAPI validates
and documents every source as declared. The reaction input keeps them apart::

    {"path": {...}, "query": {...}, "body": {...}, "headers": {...}}

with only declared sources present. The endpoint's reaction is the API
reaction: it reads that structure and builds the inputs of whatever domain
reactions it triggers. After the run, ``build`` reads the root context and
returns the response, validated against ``response``, or a ``Reply`` with a
status of its own for outcomes the run understood, an authentication
failure, a missing resource, which are not errors: the run committed.

A generic ``POST /reactions/{name}`` route triggers any registered reaction by
qualified name with the JSON body as its input and returns an empty object.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, Header, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend_builder.context import Context
from backend_builder.engine import DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT, run
from backend_builder.log import get_logger
from backend_builder.reaction import Reaction, resolve
from backend_builder.store import Store
from backend_builder.triggers import Trigger

logger = get_logger(__name__)

REACTION_ID_HEADER = "X-Reaction-Id"

_PATH_PARAM = re.compile(r"{(\w+)}")


@dataclass(frozen=True, kw_only=True)
class Endpoint:
    method: str
    path: str
    reaction: type[Reaction]
    """The API reaction. Receives the structured input and triggers domain
    reactions with inputs it builds itself."""
    path_model: type[BaseModel] | None = None
    """One field per ``{name}`` in ``path``; types convert and validate."""
    query: type[BaseModel] | None = None
    body: type[BaseModel] | None = None
    headers: type[BaseModel] | None = None
    """Field names map to header names with underscores as hyphens, case
    insensitive: ``x_tenant_id`` reads ``X-Tenant-Id``."""
    response: type[BaseModel] | None = None
    """Validates and documents what ``build`` returns."""
    build: Callable[[Context], Awaitable[Any]] | None = None
    """Async; reads the root context after the run and returns the response
    body. Absent means an empty object."""
    timeout: float | None = None
    """Budget for this endpoint's run and its response builder, in seconds.
    None means the app default. May be tighter or looser."""
    responses: Mapping[int, type[BaseModel]] = field(default_factory=dict[int, type[BaseModel]])
    """Alternative statuses a builder may ``Reply`` with, and their body
    models, for the OpenAPI document."""

    def __post_init__(self) -> None:
        names: set[str] = set(_PATH_PARAM.findall(self.path))
        declared: set[str] = set(self.path_model.model_fields) if self.path_model else set()
        if self.path_model is not None and names != declared:
            raise ValueError(
                f"{self.method} {self.path}: path parameters {sorted(names)} "
                f"do not match path_model fields {sorted(declared)}"
            )


@dataclass(frozen=True, kw_only=True)
class Reply:
    """A response with a status other than 200, returned by a builder.

    ``body`` is a Pydantic model or a JSON-shaped value. Declared for the
    schema through ``Endpoint.responses``.
    """

    status: int
    body: Any = None


class RunFailed(Exception):
    """Wraps an exception raised by a run so the handler can report the
    reaction id alongside it."""

    def __init__(self, reaction_id: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.reaction_id = reaction_id
        self.cause = cause


def create_app(
    store: Store,
    endpoints: Sequence[Endpoint] = (),
    *,
    generic: bool = True,
    resolver: Callable[[str], type[Reaction]] = resolve,
    title: str = "backend-builder",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout: float = DEFAULT_TIMEOUT,
    triggers: Sequence[Trigger] = (),
) -> FastAPI:
    """A FastAPI app over ``store`` exposing ``endpoints`` and, if ``generic``,
    the ``POST /reactions/{name}`` route. ``resolver`` maps a qualified name to
    a reaction for that route; an ``App`` passes one scoped to its package."""
    app = FastAPI(title=title)
    app.add_exception_handler(RunFailed, _run_failed)

    async def health() -> JSONResponse:
        ok = await store.ping()
        return JSONResponse(
            status_code=200 if ok else 503,
            content={"status": "ok" if ok else "unavailable", "store": type(store).__name__},
        )

    app.add_api_route("/health", health, methods=["GET"], name="health", include_in_schema=False)

    for endpoint in endpoints:
        app.add_api_route(
            endpoint.path,
            _handler(store, endpoint, max_attempts, timeout, triggers),
            methods=[endpoint.method],
            response_model=endpoint.response,
            responses={status: {"model": model} for status, model in endpoint.responses.items()},
            name=endpoint.reaction.__name__,
        )

    if generic:

        async def trigger(name: str, request: Request, response: Response) -> Any:
            body: Any = await request.json() if await request.body() else {}
            if not isinstance(body, dict):
                return _bad_request("body must be a JSON object")
            input: dict[str, Any] = {str(k): v for k, v in body.items()}  # type: ignore[union-attr]
            try:
                reaction = resolver(name)
            except LookupError as exc:
                return JSONResponse(
                    status_code=404, content={"error": "LookupError", "message": str(exc)}
                )
            ctx = await _run(store, reaction, input, max_attempts, timeout, triggers)
            response.headers[REACTION_ID_HEADER] = ctx.reaction_id
            return {}

        app.add_api_route("/reactions/{name}", trigger, methods=["POST"], name="trigger")

    return app


def _handler(
    store: Store,
    endpoint: Endpoint,
    max_attempts: int,
    timeout: float,
    triggers: Sequence[Trigger],
) -> Callable[..., Any]:
    path_names = _PATH_PARAM.findall(endpoint.path)
    sources = {"query": endpoint.query, "body": endpoint.body, "headers": endpoint.headers}

    budget = endpoint.timeout if endpoint.timeout is not None else timeout

    async def handler(**kwargs: Any) -> Any:
        response: Response = kwargs.pop("response")
        input: dict[str, Any] = {}
        if endpoint.path_model is not None:
            raw = {name: kwargs[name] for name in path_names}
            input["path"] = endpoint.path_model.model_validate(raw).model_dump()
        elif path_names:
            input["path"] = {name: kwargs[name] for name in path_names}
        for source in sources:
            model: BaseModel | None = kwargs.get(source)
            if model is not None:
                input[source] = model.model_dump()

        reaction_id = str(uuid4())
        try:
            async with asyncio.timeout(budget):
                ctx = await run(
                    store,
                    endpoint.reaction,
                    input,
                    reaction_id=reaction_id,
                    max_attempts=max_attempts,
                    timeout=budget,
                    triggers=triggers,
                )
                body: Any = await endpoint.build(ctx) if endpoint.build is not None else {}
        except Exception as exc:
            raise RunFailed(reaction_id, exc) from exc
        if isinstance(body, Reply):
            content = (
                body.body.model_dump(mode="json") if isinstance(body.body, BaseModel) else body.body
            )
            return JSONResponse(
                status_code=body.status, content=content, headers={REACTION_ID_HEADER: reaction_id}
            )
        response.headers[REACTION_ID_HEADER] = reaction_id
        return body

    kw = inspect.Parameter.KEYWORD_ONLY
    params: list[inspect.Parameter] = []
    for name in path_names:
        annotation: Any = str
        if endpoint.path_model is not None:
            annotation = endpoint.path_model.model_fields[name].annotation or str
        params.append(inspect.Parameter(name, kw, annotation=annotation))
    if endpoint.query is not None:
        params.append(inspect.Parameter("query", kw, annotation=Annotated[endpoint.query, Query()]))
    if endpoint.body is not None:
        params.append(inspect.Parameter("body", kw, annotation=endpoint.body))
    if endpoint.headers is not None:
        params.append(
            inspect.Parameter("headers", kw, annotation=Annotated[endpoint.headers, Header()])
        )
    params.append(inspect.Parameter("response", kw, annotation=Response))
    handler.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    return handler


async def _run(
    store: Store,
    reaction: type[Reaction],
    input: dict[str, Any],
    max_attempts: int,
    timeout: float,
    triggers: Sequence[Trigger],
) -> Context:
    reaction_id = str(uuid4())
    try:
        return await run(
            store,
            reaction,
            input,
            reaction_id=reaction_id,
            max_attempts=max_attempts,
            timeout=timeout,
            triggers=triggers,
        )
    except Exception as exc:
        raise RunFailed(reaction_id, exc) from exc


def _run_failed(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RunFailed)
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc.cause).__name__,
            "message": str(exc.cause),
            "reaction_id": exc.reaction_id,
        },
        headers={REACTION_ID_HEADER: exc.reaction_id},
    )


def _bad_request(message: str) -> Any:
    return JSONResponse(status_code=400, content={"error": "BadRequest", "message": message})


def serve(app: FastAPI, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve ``app`` with uvicorn. Blocks."""
    import uvicorn

    logger.info("serving host=%s port=%d", host, port)
    uvicorn.run(app, host=host, port=port)
