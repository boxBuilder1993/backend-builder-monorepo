"""The console's FastAPI app: routes and templates."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from backend_builder.admin import queries
from backend_builder.app import App
from backend_builder.entity import qualified_name as entity_name
from backend_builder.reaction import qualified_name as reaction_name
from backend_builder.schema import persisted_fields, table_name
from backend_builder.store import NotSupported, Store
from backend_builder.triggers import OnSchedule

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
COOKIE = "backend_builder_admin"


def create_admin_app(app: App, store: Store, *, token: str | None = None) -> FastAPI:
    """The console over ``app`` and ``store``. With ``token`` set, every page
    needs it, entered once on the login page and kept in a cookie."""
    console = FastAPI(title=f"{app.name} admin", docs_url=None, redoc_url=None, openapi_url=None)
    filters: dict[str, Any] = TEMPLATES.env.filters  # type: ignore[assignment]
    filters["reaction_short"] = _short
    filters["ms"] = _ms

    def render(request: Request, template: str, **context: Any) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, template, {"app": app, **context})

    @console.middleware("http")
    async def _gate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if token is None or request.url.path == "/login":
            return await call_next(request)
        if request.cookies.get(COOKIE) == token or request.headers.get("x-admin-token") == token:
            return await call_next(request)
        return RedirectResponse("/login", status_code=303)

    @console.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request) -> HTMLResponse:
        return render(request, "login.html", error=None)

    @console.post("/login")
    async def login(request: Request, value: str = Form(...)) -> Response:
        if token is None or value == token:
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(COOKIE, value, httponly=True, samesite="strict")
            return response
        return render(request, "login.html", error="wrong token")

    @console.get("/", response_class=HTMLResponse)
    async def overview(request: Request) -> HTMLResponse:
        try:
            summary = await queries.overview(store)
            recent = await queries.runs(store, limit=20)
        except NotSupported:
            summary, recent = None, []
        return render(request, "overview.html", summary=summary, recent=recent)

    @console.get("/graph", response_class=HTMLResponse)
    async def graph(request: Request) -> HTMLResponse:
        entities = [
            {
                "name": e.__name__,
                "qualified": entity_name(e),
                "table": table_name(e),
                "fields": [f.name for f in persisted_fields(e)],
            }
            for e in app.entities
        ]
        return render(request, "graph.html", entities=entities, reactions=_reactions(app))

    @console.get("/reactions/{name}", response_class=HTMLResponse)
    async def reaction(request: Request, name: str) -> HTMLResponse:
        info = next((r for r in _reactions(app) if r["qualified"] == name), None)
        if info is None:
            return render(request, "missing.html", what=f"reaction {name}")
        try:
            stats = await queries.reaction_stats(store, name)
            recent = await queries.runs(store, reaction=name, limit=50)
        except NotSupported:
            stats, recent = None, []
        return render(request, "reaction.html", info=info, stats=stats, recent=recent)

    @console.get("/runs", response_class=HTMLResponse)
    async def runs(
        request: Request,
        reaction: str | None = None,
        outcome: str | None = None,
        source: str | None = None,
    ) -> HTMLResponse:
        rows = await queries.runs(
            store, reaction=reaction, outcome=outcome, source=source, limit=100
        )
        return render(
            request,
            "runs.html",
            rows=rows,
            filters={"reaction": reaction or "", "outcome": outcome or "", "source": source or ""},
            reactions=_reactions(app),
        )

    @console.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(request: Request, run_id: str) -> HTMLResponse:
        record = await queries.run_detail(store, run_id)
        if record is None:
            return render(request, "missing.html", what=f"run {run_id}")
        return render(request, "run.html", record=record)

    @console.get("/outbox", response_class=HTMLResponse)
    async def outbox(request: Request, state: str = "pending") -> HTMLResponse:
        if state not in ("pending", "claimed", "dead"):
            state = "pending"
        stats = await store.outbox_stats()
        rows = await queries.outbox_rows(store, state)
        return render(request, "outbox.html", state=state, stats=stats, rows=rows)

    @console.get("/outbox/{row_id}", response_class=HTMLResponse)
    async def outbox_page(request: Request, row_id: str) -> HTMLResponse:
        row = await queries.outbox_row(store, row_id)
        if row is None:
            return render(request, "missing.html", what=f"outbox row {row_id}")
        return render(request, "outbox_row.html", row=row)

    @console.post("/outbox/{row_id}/requeue")
    async def outbox_requeue(row_id: str) -> Response:
        await queries.requeue(store, row_id)
        return RedirectResponse(f"/outbox/{row_id}", status_code=303)

    @console.post("/outbox/{row_id}/discard")
    async def outbox_discard(row_id: str) -> Response:
        await queries.discard(store, row_id)
        return RedirectResponse("/outbox?state=dead", status_code=303)

    @console.get("/schedules", response_class=HTMLResponse)
    async def schedules_page(request: Request) -> HTMLResponse:
        declared = {reaction_name(s.reaction): s for s in app.schedules}
        try:
            rows = await queries.schedules(store)
        except NotSupported:
            rows = []
        known = {str(r["reaction"]) for r in rows}
        unseeded = [
            {"reaction": name, "cron": s.cron} for name, s in declared.items() if name not in known
        ]
        return render(request, "schedules.html", rows=rows, unseeded=unseeded)

    @console.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return console


def _short(name: object) -> str:
    return str(name).rsplit(".", 1)[-1]


def _ms(value: object) -> str:
    return f"{float(str(value)):.1f} ms"


def _reactions(app: App) -> list[dict[str, Any]]:
    """Each reaction with everything that reaches it."""
    out: list[dict[str, Any]] = []
    for reaction in app.reactions:
        name = reaction_name(reaction)
        endpoints = [f"{e.method} {e.path}" for e in app.endpoints if e.reaction is reaction]
        triggers: list[str] = []
        for t in app.triggers:
            if t.reaction is not reaction:
                continue
            if isinstance(t, OnSchedule):
                triggers.append(f"OnSchedule({t.cron!r})")
            else:
                fields = getattr(t, "fields", None)
                detail = f", {list(fields)}" if fields else ""
                triggers.append(f"{type(t).__name__}({t.entity.__name__}{detail})")
        out.append(
            {
                "name": reaction.__name__,
                "qualified": name,
                "doc": (reaction.__doc__ or "").strip(),
                "endpoints": endpoints,
                "triggers": triggers,
            }
        )
    return out
