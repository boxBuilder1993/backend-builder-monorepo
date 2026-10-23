# Builder's Guide

How to write an application on Backend Builder. This document is about *how*; [design.md](design.md) is about *why*, and is the authority when the two disagree. Every example here comes from the `shop` app in `apps/shop`, which you can read alongside.

Contents: [the model](#1-the-model-in-one-page) · [an app](#2-an-app) · [entities](#3-entities) · [reactions](#4-reactions) · [the context](#5-the-context) · [reading and writing](#6-reading-and-writing) · [triggers and deferred work](#7-triggers-and-deferred-work) · [HTTP endpoints](#8-http-endpoints) · [events](#9-events) · [secrets](#10-secrets) · [timeouts and retries](#11-timeouts-and-retries) · [migrations](#12-migrations) · [testing](#13-testing) · [running](#14-running-it) · [observing](#15-observing-it) · [checklist](#16-checklist)

## 1. The model in one page

An **entity** is a dataclass. Its instances are rows.

A **reaction** is a class with one method, `async def calculate(self, ctx)`. It reads and writes entities through `ctx`, and can trigger other reactions. A reaction never sees HTTP, the database or a queue.

A **run** is one execution of a reaction as the root of a tree. The engine opens a **session** (an in-memory working set over the store), runs the reaction, and at the end **flushes**: inserts what was created, updates what changed, writes outbox rows for deferred work, and commits. All of that is one transaction. If anything raises, the session is discarded, the store is untouched, and the run is retried from scratch up to three times.

A reaction runs because something **triggered** it: an HTTP endpoint, a data change committed by another run, a cron schedule, another reaction, or the CLI. Where it ran from is invisible to the reaction; the input is the whole contract.

Three consequences shape everything else:

- **Mutating a loaded entity is the write.** There is no save call. Load it, change a field, done.
- **Nothing persists until flush.** You can create, mutate and read your own writes freely inside a run; the store sees a single commit or nothing.
- **A run must be a function of its input and the store.** It can be retried at any point before commit. Don't send emails from inside `calculate`; defer a reaction that does, and it runs exactly once, after commit.

## 2. An app

An app is a Python package in `apps/` with an `App` object at its root:

```
apps/myshop/
  pyproject.toml            # depends on backend-builder, workspace source
  src/myshop/
    __init__.py             # app = App(package="myshop", endpoints=..., triggers=...)
    entities.py
    reactions.py
    api.py                  # request models, API reactions, builders, endpoints
    triggers.py
    migrations/             # generated; versions/ holds the chain
  tests/
```

```python
# src/myshop/__init__.py
from backend_builder import App
from myshop.api import endpoints
from myshop.triggers import triggers

app = App(package="myshop", endpoints=endpoints, triggers=triggers)
```

`App` imports every module under the package and collects the entities and reactions defined there. You never register anything; defining a class in the package is the registration. Classes from a shared package are added with `include=[...]`. The file split above is a convention, not a rule: any module in the package may define entities or reactions.

`App` also takes `max_attempts` (default 3) and `timeout` (default 60 seconds) as the policy for every run of the app. See [§11](#11-timeouts-and-retries).

Add the package to the workspace in the root `pyproject.toml`, run `uv sync --all-packages`, and `uv run backend-builder graph myshop` prints what the app contains. If it prints, the app is wired.

## 3. Entities

```python
from backend_builder import Entity


class Order(Entity):
    user_id: str
    tenant: str = "default"
    total: int = 0
    status: str = "new"


class OrderLine(Entity):
    order_id: str
    price: int
    qty: int
```

Subclassing `Entity` makes the class a keyword-only dataclass and gives it three system fields:

| Field | Meaning |
| --- | --- |
| `id: str` | UUID4, generated on construction. Never None, never a sequence. |
| `created_at: datetime` | UTC, set on construction. |
| `deleted_at: datetime \| None` | Set by `ctx.delete`. Rows with it set are invisible to reads unless asked for. |

There is no `updated_at`.

**Field types** map to columns: `str`, `int`, `float`, `Decimal`, `bool`, `datetime`, `date` are scalar columns, and `X | None` makes the column nullable. Anything JSON-shaped (`dict`, `list`, a nested dataclass, a union, `Any`) becomes a `jsonb` column and round-trips through the codec, so a nested dataclass comes back as that dataclass. Anything else is a schema error at `makemigration` time.

**Table names** are the class name in snake case: `OrderLine` is `order_line`. `Order` is `"order"`, which needs quoting in SQL because it is a keyword; pick names that aren't.

**Relations** are plain id fields: `order_id: str`. There is no foreign-key type, no lazy loading and no relationship declaration. A reaction that needs the lines of an order queries for them. Many-to-many is a link entity. If a reaction wants to hang a loaded object on an instance for its own convenience, mark the field non-persisted and it stays out of the table:

```python
from dataclasses import field
from backend_builder.schema import PERSIST


class Order(Entity):
    user_id: str
    lines: list["OrderLine"] = field(default_factory=list, metadata={PERSIST: False})
```

**Constraints, indexes, uniqueness and defaults for existing rows** are not declared on the entity. They are SQL in a migration you write ([§12](#12-migrations)). An entity describes the data and nothing about storage.

**Validation** is the reaction's job. A reaction that receives a bad input raises, and the run fails. There is no validation layer on entities.

## 4. Reactions

```python
from backend_builder import Context, Reaction


class UpdateLine(Reaction):
    """Reprice or requantify a line. The total follows through OnChange."""

    async def calculate(self, ctx: Context) -> None:
        line = await ctx.get_by_id(OrderLine, ctx.input["line_id"])
        if line is None or line.order_id != ctx.input["order_id"]:
            raise LookupError(f"no line {ctx.input['line_id']} on order {ctx.input['order_id']}")
        if "price" in ctx.input:
            line.price = ctx.input["price"]
        if "qty" in ctx.input:
            line.qty = ctx.input["qty"]
        ctx.save("line", line)
```

**Input** is `ctx.input`, a `dict[str, Any]`. The reaction defines its own shape and documents it in its docstring. Inputs that arrive through the outbox must be JSON-encodable; inputs built in-process may hold anything.

**Output** is whatever the reaction wrote to the store, plus anything it `ctx.save`d. A reaction has no return value; its caller reads what it saved from the context `trigger` returns, and the response builder reads the root's.

**Errors.** Any exception aborts the run. There is no special error class to raise, and nothing catches exceptions for you. An outcome the run *understands* (the order doesn't exist, the card was declined) is not an error: record it with `ctx.save` and return normally, so the run commits and the caller decides what to do. See [§8](#8-http-endpoints) for how that becomes a 404.

**Triggering another reaction** runs it now, in the same session, depth-first:

```python
child = await ctx.trigger(OrderTotal, {"order_id": order.id})
```

The child sees everything the parent has created or changed so far, and the parent sees the child's writes when `trigger` returns. Entities are shared; nothing else is. The child's `vars` are its own, so what it saved is read from the returned context, and anything it needs from the parent arrives in its input:

```python
created = await ctx.trigger(CreateOrder, {"user_id": ..., "lines": ...})
ctx.save("order", created.load("order"))   # lift what the child saved, if the caller needs it
```

Passing information between reactions is your job, in both directions, through input and the returned context. That is what lets every invocation be recorded on its own. `deferred=True` is a different thing: see [§7](#7-triggers-and-deferred-work).

**Contracts to keep in mind.**

- Triggering inside a loop over a thousand rows runs the child a thousand times. Collect, then trigger once.
- A reaction may be retried. Keep side effects to the store; defer anything that talks to the outside world.
- Hold the context, nothing else. Don't stash it on `self`; reactions are instantiated per invocation and should be stateless.
- Reads native to one store don't exist on another: `get` with a predicate is in-memory only; `query` with SQL is Postgres only. `get_by_id`, `create`, `delete` and mutation work everywhere.

Reactions are named by their qualified name, `myshop.reactions.UpdateLine`. That name appears in logs, run records, the outbox, the admin console and the CLI.

## 5. The context

```python
ctx.input                       # dict[str, Any]
ctx.reaction_id                 # this invocation, a UUID string
ctx.parent_reaction_id          # the triggering invocation; None for the root
ctx.trace                       # the run's spans, for tests and tooling

await ctx.get_by_id(Entity, id, *, include_deleted=False) -> E | None
await ctx.get(Entity, where=lambda row: ..., *, include_deleted=False) -> list[E]   # in-memory only
await ctx.query(Entity, sql, **params) -> list[E]                                   # Postgres only
await ctx.query_raw(sql, **params) -> list[dict]                                    # Postgres only

ctx.create(Entity, **fields) -> E      # inserted at flush
ctx.delete(row)                        # soft delete: sets deleted_at

ctx.save(name, value)                  # this invocation's; a parent reads it from the returned context
ctx.load(name) -> Any                  # KeyError if absent
ctx.vars                               # the dict behind save and load, own to this invocation

ctx.secret(name) -> str                # SECRET_<NAME> from the environment

await ctx.trigger(Reaction, input, timeout=None) -> Context       # now, same session
await ctx.trigger(Reaction, input, deferred=True, timeout=None)   # after commit, by a worker
```

`vars` belong to one invocation and are not persisted as data; they are where a reaction leaves things for its caller or, on the root, for the response builder, and they are stored on the invocation's run record so the console can show them. Keep it small and JSON-friendly where you can; anything else is stored as its `repr`.

## 6. Reading and writing

**By id** works everywhere and is the common case:

```python
order = await ctx.get_by_id(Order, order_id)
```

Loaded instances live in the session's identity map: loading the same id twice returns the same object, and a child reaction sees the parent's instance, not a fresh copy. Deleted rows return `None` unless `include_deleted=True`.

**Queries** are SQL, parameterised, returning instances of the named entity merged into the session:

```python
lines = await ctx.query(
    OrderLine,
    "select * from order_line where order_id = :order_id and deleted_at is null",
    order_id=order.id,
)
```

The statement is yours: joins, `where`, `order by`, `limit`, anything Postgres accepts, as long as the result columns are the entity's columns (`select *` or an explicit list). Soft-deleted rows are *not* filtered automatically in SQL; add `deleted_at is null` yourself. For aggregates, joins returning mixed columns, or anything that isn't an entity, use `query_raw` and get dicts back.

Before any SQL runs, the session flushes its pending creates and changes to the transaction, so a query sees what the run has already done. The transaction is serializable: what you read is consistent with what you write, and a conflicting concurrent run makes one of the two retry.

The in-memory store has no SQL. `ctx.get(Entity, lambda row: ...)` filters with a Python predicate there, and raises `NotSupported` on Postgres. The two are deliberately not bridged: a reaction that needs a query picks a store and says so at the call site. Reactions that only use `get_by_id`, `create`, mutation and `delete` run on both, which is what makes in-memory unit tests cheap.

**Writes.** `ctx.create` returns the instance, already holding its id, so you can reference it immediately. Mutating any loaded or created instance's fields is the update. `ctx.delete(row)` sets `deleted_at`; the row stays in the table. The flush writes only the fields that actually changed, inserts in creation order, and never upserts: creating an id that exists fails the run with `DuplicateKeyError`.

**Hot rows.** Every run that mutates the same row serialises on it, and under contention most of them retry. Don't have a thousand runs each increment one counter. Append a row each (see [§9](#9-events)) and roll up on a schedule, or shard the counter.

## 7. Triggers and deferred work

Triggers are declared on the App, not on reactions, so one file lists everything that causes work:

```python
from backend_builder import OnChange, OnCreate, OnDelete, OnSchedule

triggers = [
    OnCreate(Order, NotifyOrder),
    OnCreate(OrderLine, OrderTotal),
    OnChange(OrderLine, ["price", "qty"], OrderTotal),
    OnDelete(OrderLine, OrderTotal),
    OnSchedule("* * * * *", ExpireUnpaidOrders),
]
```

**Data-change triggers** fire when a run commits a matching change. At flush the session compares every created, changed and deleted row against the triggers and writes one outbox row per match, in the same transaction as the data. A worker process then claims the row and runs the reaction as a run of its own. The reaction's input is the change, not the data:

```python
{
    "entity": "order_line",
    "id": "...",
    "kind": "created" | "updated" | "deleted",
    "changed": ["price"],
}
```

`changed` is present for `OnChange` and lists which of the trigger's fields changed. The reaction loads current state fresh, which is why the same `OrderTotal` serves all three triggers: whatever happened to a line, recompute the order from what's there now. Write triggered reactions so that running them twice is harmless; at-least-once delivery is the guarantee.

**Schedules** are cron expressions in UTC. The scheduler process keeps the next occurrence of each schedule in the outbox; the worker runs it when due. Input is `{"scheduled_for": "<ISO timestamp>"}`. A scheduled reaction does its own querying for what needs doing, so a missed tick catches up on the next.

**Deferring from code** is the same mechanism without a declaration:

```python
await ctx.trigger(SendReceipt, {"order_id": order.id}, deferred=True)
```

The outbox row is written at flush, so it exists only if the run commits. The input must be JSON-encodable, checked when you call it. This is how a run does something that must happen exactly once after commit, an email or an external API call: the deferred reaction does it, and if it fails it is retried by the worker, not by re-running the committing reaction.

**The worker** claims up to `OUTBOX_BATCH` rows per poll, runs them concurrently up to `OUTBOX_CONCURRENCY`, and acks each inside its run's transaction, so a crash mid-run leaves the row to be claimed again when its lease expires. A row that fails is retried with backoff up to `OUTBOX_MAX_ATTEMPTS` (default 5) and then marked dead. Dead rows are visible in the admin console, where they can be requeued or discarded, and in `backend-builder outbox myshop`.

Triggers need the outbox, which lives in Postgres. An app that declares triggers refuses to start on the in-memory store. Tests of a triggered reaction call it directly ([§13](#13-testing)); tests of the whole chain use Postgres and `Worker.run_once()`.

## 8. HTTP endpoints

An endpoint is a declaration binding a route to a reaction, with Pydantic models for the request parts and the response:

```python
from backend_builder import Endpoint, Reply

Endpoint(
    method="GET",
    path="/orders/{order_id}",
    reaction=GetOrderApi,
    path_model=OrderPath,  # one field per {name} in path
    query=GetOrderQuery,  # ?include_lines=true
    response=OrderResponse,  # documents and validates what build returns
    responses={404: ErrorOut},  # other statuses build may Reply with
    build=order_response,
)
```

`body` and `headers` are the other two request models. Header field names map to header names with underscores as hyphens, case-insensitive: `x_tenant_id` reads `X-Tenant-Id`. Validation failures are a 422 before any reaction runs.

**The API reaction** receives the validated request as structured input and translates it into domain inputs:

```python
class CreateOrderApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        body = ctx.input["body"]  # also "path", "query", "headers"
        await ctx.trigger(
            CreateOrder,
            {
                "user_id": body["user_id"],
                "lines": body["lines"],
                "tenant": ctx.input["headers"]["x_tenant_id"],
            },
        )
        ctx.save("include_lines", True)
```

Keep API reactions thin and domain reactions ignorant of HTTP. Domain reactions get reused by triggers, schedules and other apps' endpoints; API reactions exist only to shape one route.

**The builder** is an async function that runs *after* the run commits, reads the root context, and returns the response body:

```python
async def order_response(ctx: Context) -> OrderResponse | Reply:
    if "not_found" in ctx.vars:
        return Reply(status=404, body=ErrorOut(**ctx.load("not_found")))
    order: Order = ctx.load("order")
    return OrderResponse(order=OrderOut.model_validate(order), lines=[])
```

A builder may run read-only queries against the committed state. It returns a model (200) or a `Reply(status, body)` for any other status. This is how outcomes work end to end: the reaction records that the order wasn't found with `ctx.save`, the run commits normally, and the builder replies 404. Only exceptions become 500s, and a 500 carries the run's id in the `X-Reaction-Id` header so you can find it in the console.

Every app also serves `POST /reactions/{qualified.name}` with the JSON body as input, for tooling and tests, and `GET /health`. The OpenAPI document is at `/docs`.

## 9. Events

There is no event or metrics API because an entity already is one. To record that something happened, create a row:

```python
class OrderPlaced(Entity):
    order_id: str
    tenant: str
    total: int


# inside a reaction
ctx.create(OrderPlaced, order_id=order.id, tenant=order.tenant, total=order.total)
```

It commits with the run, so a retried attempt never double-records. `OnCreate(OrderPlaced, ...)` makes any number of consumers run after commit, each retried independently. SQL answers the questions, and the admin console shows which run wrote each row.

Two things are yours to decide per event type, in migrations and scheduled reactions: indexes (lead with `created_at`, or tenant then `created_at`) and retention (a scheduled reaction that deletes old rows; the runtime purges only its own `_run` tables).

For anything that needs measuring, write a reaction that measures it. The runtime's own Prometheus metrics ([§15](#15-observing-it)) cover run counts, durations, outbox depth and HTTP traffic.

## 10. Secrets

```python
key = ctx.secret("stripe_api_key")  # reads SECRET_STRIPE_API_KEY
```

Names are lowercase snake case; the variable is the name upper-cased behind `SECRET_`. Unset or empty raises `SecretError` and fails the reaction. There is no secret store in the library: compose, Kubernetes or a cloud secret manager injects the variables.

Every value read this way is masked as `[secret:stripe_api_key]` in run records and log lines, so a key that lands in `ctx.save` or a logged payload doesn't reach the console or the log aggregator. Masking is by exact value; an encoded or truncated key still leaks, and values under 8 characters are not masked at all. Treat it as a seatbelt.

In tests, `monkeypatch.setenv("SECRET_STRIPE_API_KEY", "sk_test_...")`, or construct `Context(..., secrets=Secrets({"SECRET_STRIPE_API_KEY": "..."}))` directly.

## 11. Timeouts and retries

A run gets one budget, `App.timeout` (default 60 seconds), covering every attempt. When it expires the attempt is cancelled, its transaction rolls back, and `TimeoutError` propagates without retry. Within the budget, any other exception discards the session and retries from the root, up to `App.max_attempts` (default 3), with a short jittered pause. The retried attempt gets a fresh session and the same `reaction_id`.

Per-trigger budgets override the app default for one kind of work: `OnSchedule("0 3 * * *", NightlyRollup, timeout=1800)`, `Endpoint(..., timeout=5)` (which covers the run and the builder), `ctx.trigger(..., deferred=True, timeout=300)`. An immediate `ctx.trigger(..., timeout=2)` can only tighten, since it runs inside the parent's budget.

**Long work.** A run is one transaction. Holding it for an hour conflicts with every concurrent write to what it read, and a retry starts from zero. Split bulk work: an orchestrator reaction queries ids and defers one chunk reaction per few hundred rows. Each chunk is a short transaction, retried on its own, spread across workers.

## 12. Migrations

Entities describe columns; migrations create them. The chain lives in `src/myshop/migrations/versions/` and is Alembic underneath.

```sh
uv run backend-builder makemigration myshop -m "order status"     # diff entities against the database
uv run backend-builder makemigration myshop -m "line index" --empty   # a blank file for hand-written SQL
uv run backend-builder migrate myshop                               # apply
uv run backend-builder migrate myshop --check                       # fail if entities and schema differ
```

`makemigration` needs a database to compare against (`BACKEND_BUILDER_DATABASE_URL`, usually the compose Postgres). It generates table and column changes for your entities, and index and constraint changes only for the runtime's own `_`-prefixed tables. Your indexes, uniqueness, foreign keys and check constraints go in `--empty` migrations as `op.execute("create index ...")`.

Read every generated file before committing it. A new non-nullable column on a populated table needs a `server_default` added by hand; a renamed field generates a drop and an add, which loses data, and should be rewritten as a rename. Migrations are reviewed in the pull request like any other code. Keep `migrations.check(app, migrated_database_url)` as a test ([§13](#13-testing)): it applies the chain to a fresh database and fails if the entities have drifted from it, so CI catches a forgotten migration.

The runtime's own tables (`_outbox`, `_schedule`, `_run`) are part of your app's chain, generated the same way when the runtime adds them. When a library upgrade adds a system table, `makemigration` picks it up.

## 13. Testing

**Unit tests run in memory and need nothing running:**

```python
from backend_builder import InMemoryStore, run


async def test_create_order_computes_total() -> None:
    store = InMemoryStore()
    ctx = await run(store, CreateOrder, {"user_id": "u7", "lines": [{"price": 10, "qty": 2}]})

    order = ctx.load("order")
    saved = await store.get_by_id(Order, order.id)
    assert saved.total == 20
```

`run(store, Reaction, input)` is the engine itself, so the test exercises the real flush, retry and trace. The returned context is the root's; `ctx.load` reads what the run saved and `ctx.trace` has its spans. Reactions that use `ctx.query` need Postgres; everything else runs here. The suite is `asyncio_mode=auto`, so `async def test_*` just works.

**Postgres tests** use the fixtures from `backend-builder-testing`. Define one fixture naming your app and the plugin gives you a fresh, migrated database per test:

```python
# tests/conftest.py
@pytest.fixture(scope="session")
def backend_builder_app() -> App:
    return app


@pytest.fixture
async def store(migrated_database_url: str) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(migrated_database_url, app.entities)
    yield store
    await store.close()
```

Drive the HTTP layer with `httpx2.AsyncClient(transport=ASGITransport(app=app.http(store=store)))`, and deliver triggered work with `Worker(app, store, Settings()).run_once()`, which claims and runs what's in the outbox and returns the count. The fixtures skip when no Postgres is reachable at `BACKEND_BUILDER_TEST_DATABASE_URL` (default: the compose one), except in CI where they fail. `docker compose up -d` starts it. `test_migrations_match_entities` is one line, `migrations.check(app, migrated_database_url)`, and every app should have it.

**End-to-end tests** build the image and bring up the whole compose stack; `apps/shop/tests/e2e` is the template. They drive the public API, read the database directly and read container logs. Mark them `@pytest.mark.e2e`; the default `pytest` excludes them and `pytest -m e2e <path>` runs them.

Run everything the way CI does: `uv run ruff format --check . && uv run ruff check . && uv run pyright && uv run pytest`.

## 14. Running it

Every process is the one CLI with a different command, and the same container image:

| Command | What it does |
| --- | --- |
| `backend-builder serve myshop [--reload]` | the API on 8000, metrics on 9100 |
| `backend-builder worker myshop [--once]` | deliver outbox rows |
| `backend-builder scheduler myshop [--once]` | seed schedules, purge old run records |
| `backend-builder admin myshop` | the admin console on 9200 |
| `backend-builder migrate myshop [--check]` | apply or verify the migration chain |
| `backend-builder makemigration myshop -m msg [--empty]` | generate a migration |
| `backend-builder trigger myshop myshop.reactions.X --input '{...}'` | run one reaction from the shell |
| `backend-builder graph myshop` | print entities, reactions, endpoints, triggers |
| `backend-builder outbox myshop` | pending, claimed and dead row counts |

Configuration is environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RUNTIME_ENV` | `dev` | `test`, `dev` or `prod`; `prod` refuses to start without a database |
| `BACKEND_BUILDER_DATABASE_URL` | unset | unset means in memory; set means Postgres at that URL |
| `BACKEND_BUILDER_LOG_LEVEL` | `INFO` | |
| `BACKEND_BUILDER_OUTBOX_POLL_SECONDS` | `1` | worker poll interval |
| `BACKEND_BUILDER_OUTBOX_BATCH` | `10` | rows claimed per poll |
| `BACKEND_BUILDER_OUTBOX_CONCURRENCY` | `4` | rows run concurrently per worker |
| `BACKEND_BUILDER_OUTBOX_MAX_ATTEMPTS` | `5` | deliveries before a row is dead |
| `BACKEND_BUILDER_OUTBOX_LEASE_GRACE_SECONDS` | `10` | added to a row's timeout to form its lease |
| `BACKEND_BUILDER_SCHEDULER_TICK_SECONDS` | `15` | must stay under a minute |
| `BACKEND_BUILDER_RUN_RECORDS` | `full` | `full`, `summary` (no input or vars), `off` |
| `BACKEND_BUILDER_RUN_RETENTION_DAYS` | `7` | run records older than this are purged |
| `BACKEND_BUILDER_METRICS_PORT` | `9100` | `/metrics` and `/health` on every process |
| `BACKEND_BUILDER_ADMIN_PORT` | `9200` | |
| `BACKEND_BUILDER_ADMIN_TOKEN` | unset | when set, the console asks for it once |
| `SECRET_<NAME>` | | app secrets, read by `ctx.secret` |

Locally, `compose.yaml` runs all of it for the shop under the `shop` profile; copy that block for a new app, changing the `APP` build argument. The `Dockerfile` builds any app with `--build-arg APP=myshop`.

## 15. Observing it

**Run records.** Every reaction invocation, the seed of a run or a child triggered from another reaction, leaves its own row in `_run`: the reaction, its input, what it saved, what it created or deleted, what it deferred, its duration and outcome, and its parent, so the tree of a run is a query on `root_id`. The seed's row adds what only exists at commit: the attempt count, the flush time, every field change the run wrote, and whether it committed; a child's row says whether the run it belonged to did. The seed's row is written as `running` when the run starts and replaced, with every child's row, when it ends, so the console shows what is executing right now. A `running` row whose deadline has passed is marked `lost` by the scheduler: the process died mid-run, and Postgres rolled its transaction back. The retention window is purged by the scheduler, whole runs at a time.

**The admin console** (`backend-builder admin myshop`, port 9200) reads those tables: an overview of recent runs and outbox depth, the app's graph, a page per reaction with its stats and recent runs, a page per run with its span tree, inputs, vars and changes, the outbox with requeue and discard for dead rows, and the schedules. Set `BACKEND_BUILDER_ADMIN_TOKEN` anywhere it isn't on a private network.

**Metrics.** Every process serves Prometheus metrics on its metrics port: runs by reaction, source and outcome; run and span durations; HTTP requests by route and status; outbox deliveries and depth; schedule seeds and purges. Labels are bounded; there is no per-id metric. `/health` on the same port pings the store.

**Logs** go to stderr, one line per lifecycle event at INFO with identifiers only (reaction name, reaction id, parent id, counts, durations) and payloads only at DEBUG. Every line of a run carries its `reaction_id`, which is also the `X-Reaction-Id` response header and the run record's id, so an error report, a log search and a console page meet at one string.

## 16. Checklist

Before opening a pull request that adds or changes an app:

- Entities are data. Indexes, constraints and defaults are in a migration you read before committing.
- Domain reactions never read `ctx.input["body"]`; API reactions never touch entities beyond handing off.
- Nothing crosses between reactions except input down and the returned context up. No reaction relies on a value another one saved.
- Anything that talks to the outside world is a deferred reaction, and is safe to run twice.
- Triggered reactions load current state and recompute; they don't trust the change record beyond the id.
- Outcomes are saved and replied; only failures raise.
- Secrets come from `ctx.secret`, never `os.environ`.
- Every reaction has an in-memory test; every query has a Postgres test; `migrations.check` is in the suite.
- `backend-builder graph myshop` reads like a description of the feature.
