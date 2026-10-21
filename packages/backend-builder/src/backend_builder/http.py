"""HTTP layer: endpoints that trigger reactions, on FastAPI.

An ``Endpoint`` binds a method and path to a reaction. The request is
validated against a Pydantic model, dumped to a dict and merged with the path
parameters to form the reaction input. After the run, a builder reads the
root context and returns the response, validated against the response model.
Reactions never see HTTP shapes; builders never write.

A generic ``POST /reactions/{name}`` route triggers any registered reaction by
qualified name with the JSON body as input and returns an empty object.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend_builder.context import Context
from backend_builder.engine import run
from backend_builder.log import get_logger
from backend_builder.reaction import Reaction, resolve
from backend_builder.store import Store

logger = get_logger(__name__)

REACTION_ID_HEADER = "X-Reaction-Id"

_BODY_METHODS = {"POST", "PUT", "PATCH"}
_PATH_PARAM = re.compile(r"{(\w+)}")


@dataclass(frozen=True, kw_only=True)
class Endpoint:
    method: str
    path: str
    reaction: type[Reaction]
    request: type[BaseModel] | None = None
    """Validates the body (POST, PUT, PATCH) or the query string (other
    methods). Its fields, plus path parameters, form the reaction input."""
    response: type[BaseModel] | None = None
    """Validates and documents what ``build`` returns."""
    build: Callable[[Context], Any] | None = None
    """Reads the root context after the run and returns the response body.
    Absent means an empty object."""


class RunFailed(Exception):
    """Wraps an exception raised by a run so the handler can report the
    reaction id alongside it."""

    def __init__(self, reaction_id: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.reaction_id = reaction_id
        self.cause = cause


def create_app(
    store: Store, endpoints: Sequence[Endpoint] = (), *, generic: bool = True
) -> FastAPI:
    """A FastAPI app over ``store`` exposing ``endpoints`` and, if ``generic``,
    the ``POST /reactions/{name}`` route."""
    app = FastAPI()
    app.add_exception_handler(RunFailed, _run_failed)

    for endpoint in endpoints:
        app.add_api_route(
            endpoint.path,
            _handler(store, endpoint),
            methods=[endpoint.method],
            response_model=endpoint.response,
            name=endpoint.reaction.__name__,
        )

    if generic:

        async def trigger(name: str, request: Request, response: Response) -> dict[str, Any]:
            body: Any = await request.json() if await request.body() else {}
            if not isinstance(body, dict):
                return _bad_request("body must be a JSON object")
            input: dict[str, Any] = {str(k): v for k, v in body.items()}  # type: ignore[union-attr]
            ctx = await _run(store, resolve(name), input)
            response.headers[REACTION_ID_HEADER] = ctx.reaction_id
            return {}

        app.add_api_route("/reactions/{name}", trigger, methods=["POST"], name="trigger")

    return app


def _handler(store: Store, endpoint: Endpoint) -> Callable[..., Any]:
    path_params = _PATH_PARAM.findall(endpoint.path)
    takes_body = endpoint.method.upper() in _BODY_METHODS

    async def handler(**kwargs: Any) -> Any:
        response: Response = kwargs.pop("response")
        model: BaseModel | None = kwargs.pop("request", None)
        input: dict[str, Any] = {name: kwargs[name] for name in path_params}
        if model is not None:
            input.update(model.model_dump())
        ctx = await _run(store, endpoint.reaction, input)
        response.headers[REACTION_ID_HEADER] = ctx.reaction_id
        return endpoint.build(ctx) if endpoint.build is not None else {}

    params = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=str)
        for name in path_params
    ]
    if endpoint.request is not None:
        annotation: Any = endpoint.request if takes_body else Annotated[endpoint.request, Query()]
        params.append(
            inspect.Parameter("request", inspect.Parameter.KEYWORD_ONLY, annotation=annotation)
        )
    params.append(
        inspect.Parameter("response", inspect.Parameter.KEYWORD_ONLY, annotation=Response)
    )
    handler.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    return handler


async def _run(store: Store, reaction: type[Reaction], input: dict[str, Any]) -> Context:
    reaction_id = str(uuid4())
    try:
        return await run(store, reaction, input, reaction_id=reaction_id)
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
