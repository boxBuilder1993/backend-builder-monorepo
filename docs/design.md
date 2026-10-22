# Backend Builder Design

Status: current as of 2026-10-22. This is the rationale and the record of decisions; [guide.md](guide.md) is the how-to for building an app. Describes what is built on `main` and what is decided but not yet built. Each section says which. The earlier draft, with the reconcile loop, consumer config and WAL worker, is superseded; those ideas are listed under "Later" with what has to be true before they return.

## Intent

The goal is for an entire company's business logic to live in a single repository, and for that repository to manage its own infrastructure. Backend Builder is the runtime that makes this practical: a single backend in which the way logic executes, synchronously in a request, in a pool of workers after commit, or on a schedule, is a deployment choice expressed in config, not a boundary expressed in code.

Three separations make that possible. Reactions are the unit of logic, and are functions of an input and a session. The engine is the unit of execution, and runs any reaction in any mode. The app is the unit of scope, and declares what reactions, entities and endpoints exist. Because the three never leak into each other, the same reaction can be moved between execution modes, pooled across workers, or exposed on a new endpoint without a line of its code changing.

What the company gets from this:

- **One map of the business.** `backend-builder graph <app>` prints every entity, reaction and endpoint. Entities and reactions are the whole system; a new engineer reads them and knows it.
- **Whole-operation transactions.** A request that touches orders, ledgers and stock is one flush and one commit. No sagas, no compensation logic.
- **Uniform testing.** Every reaction is tested by building a context over a store, calling it, and inspecting the store. No HTTP, no service mocks.
- **Infra as flags, not rewrites.** Start everything synchronous and in memory. Move to Postgres by setting a URL. Defer the slow parts later by changing how they are triggered.
- **A small deployment surface.** One image, one CLI, one process type per role.
- **Maintainable by AI as well as people.** Every dependency is a declaration, every reaction's whole world is its input and the session, and every run is deterministic given the store. An assistant, like a new engineer, can read the entities, the reactions and the endpoints and have the whole system.

The database is an abstraction the engine owns. The in-memory store is native and complete for development and tests. Postgres is the first persistent store. Others may follow, and the interface is written so that a store's native capabilities stay native rather than being flattened into a lowest common denominator.

## 1. Purpose and scope

Backend Builder is a Python library for building a business backend as a set of entities and the reactions that operate on them. A developer declares entity classes as plain dataclasses, writes reactions as async functions of a context, declares endpoints that map HTTP requests to reactions, and collects them in an `App`. The engine owns everything else: the session, the flush, the transaction, logging, tracing, the HTTP surface and the CLI.

**Built** (sections 2 through 11): entities, reactions, context, session, the in-memory store, the run lifecycle, the app, the CLI, the HTTP layer, logging and tracing.

**Decided, not built** (sections 12 through 14): `query`, the Postgres store, schema and migrations, settings.

**Later** (section 15): deferred execution, data-change triggers, scheduling, Docker and Kubernetes.

Design principles that settled the hard calls:

- **One mechanism.** Every reaction runs through `invoke`. Validation, actions, creates and reads are all reactions. Delete is a field write. GET and POST are the same path.
- **The developer owns the contract.** The engine does not police what a reaction reads, infer dependencies, or referee overlapping writes. It makes behaviour deterministic and visible.
- **Entities are data.** A dataclass describes fields and nothing about storage. Constraints, indexes and references are the app's, written as migrations.
- **Stores stay native.** The in-memory store does what Python does. Postgres does what SQL does. A reaction that uses a store-specific read says so at the call site.
- **Nothing persists until flush.** A run mutates a session. The store is written once, at the end, or not at all.
- **Any error aborts.** An exception of any kind inside a run discards the session. There is no special error class.

## 2. Core model

**Entity.** A keyword-only dataclass. Subclasses declare fields as annotations. Instances are rows. Every entity has `id`, `created_at` and `deleted_at` from the base class. Registered by qualified name on definition.

**Reaction.** A class with one method, `async def calculate(self, ctx)`. It reads from the context, mutates instances, creates instances, saves values, triggers other reactions, or raises. Registered by qualified name on definition. Code refers to reactions by class; the registry exists for boundaries where a class must be a string.

**Context.** What one reaction invocation sees: `input`, `reaction_id`, `parent_reaction_id`, the session, and the trace. One per invocation. All invocations in a run share the session and the trace.

**Session.** The working set of one run. An in-memory store that loads from a backing store on a miss, holds every instance the run created or loaded, carries free-form `vars`, and writes everything back on `flush`. Discarding it discards the run.

**Store.** Where rows live between runs. The in-memory store and, when built, the Postgres store. A store provides `get_by_id`, `create`, `update`, a lifetime-of-run scope, and a health check, plus the reads that are native to it.

**Trigger.** `await ctx.trigger(reaction, input)` runs a reaction now, with a child context on the same session. With `deferred=True` it records an outbox row instead, written with the run's flush and delivered after commit by a worker.

**App.** The declaration of an application's scope: a Python package, the reactions and entities defined under it, its endpoints, and its store factory. The CLI derives every process from it.

## 3. Entities

Built.

```python
class OrderLine(Entity):
    order_id: str
    price: int
    qty: int
    order: Order | None = field(default=None, metadata={"persist": False})
```

**Dataclasses, automatically.** `Entity.__init_subclass__` applies `dataclass(kw_only=True)` to every subclass, and the base is decorated with `dataclass_transform` so type checkers know. Fields without defaults may follow fields with defaults because everything is keyword-only.

**System fields**, declared on the base and inherited:

| Field | Type | Managed how |
| --- | --- | --- |
| `id` | `str` | A version 4 UUID string generated by the field default. The primary key. May be supplied explicitly. |
| `created_at` | `datetime` | UTC time the instance was constructed. |
| `deleted_at` | `datetime | None` | `None` while live; set by `ctx.delete`. |

There is no `updated_at`. Mutation is by reference and nothing observes it, so the engine cannot stamp it honestly.

**Ids are client-generated.** An instance has its final id the moment it exists, so references between new instances are plain strings and nothing is pending at flush. The probability of a collision is far below hardware error rates, and if one occurred the store's primary key would reject the insert rather than overwrite the row, because created rows are always inserted and never upserted.

**Non-persisted fields.** `field(metadata={"persist": False})`, or a helper that sets it. Present on instances, filled by reactions, never written to the store. This is how an order carries its lines for the rest of a run: a reaction loads them and assigns them. There is no automatic loading of related rows.

**Values, not references.** A field typed as another entity, a nested dataclass, a dict or a list is a value. On Postgres it is stored as `jsonb`, a snapshot taken at flush. The engine never infers a relation from a type. Relations are string id fields, and the constraint that makes them foreign keys is declared in the app's migrations, not on the entity.

**Soft delete.** `ctx.delete(row)` stamps `deleted_at`. Reads hide the row unless asked with `include_deleted=True`. Nothing is ever removed.

**Unmappable types** fail when the schema is generated, not at runtime.

## 4. Reactions

Built.

```python
class OrderTotal(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = await ctx.get_by_id(Order, ctx.input["order_id"])
        if order is None:
            raise LookupError(ctx.input["order_id"])
        lines = await ctx.get(OrderLine, lambda line: line.order_id == order.id)
        order.total = sum(line.price * line.qty for line in lines)
```

**Input** is a `dict[str, Any]`. The reaction knows its own shape. API reactions receive the structured request described in section 10 and build domain inputs themselves.

**Invocation** goes through one function, `invoke(reaction, ctx)`, whether the reaction is the seed of a run or was triggered. It logs start, end and failure, and records a span. That is where timeouts and tracing attach.

**Errors.** Any exception aborts the run. The session is discarded, the store is untouched, the caller receives the error.

**Contracts the developer must know.**

- Triggers are synchronous and depth-first. Triggering inside a loop over a thousand instances runs the sub-reaction a thousand times. Collect, then trigger once.
- A sub-reaction shares the session. Everything it mutates or creates is visible to the caller when `trigger` returns.
- Rows are by reference. Mutating a loaded instance is the write. There is no save call.
- Reads that are native to one store are not available on another. `get` with a lambda is in-memory only; `query` with SQL is Postgres only. `get_by_id`, `create`, `delete` work everywhere.

## 5. Context API

Built.

```python
ctx.input                                   # dict[str, Any], the invocation's input
ctx.reaction_id                             # this invocation
ctx.parent_reaction_id                      # the triggering invocation, None for the seed

await ctx.get_by_id(Entity, id, *, include_deleted=False) -> E | None
await ctx.get(Entity, where=lambda row: ..., *, include_deleted=False) -> list[E]   # in-memory only
await ctx.query(Entity, statement, **params) -> list[E]                             # Postgres only, decided
await ctx.query_raw(statement, **params) -> list[dict[str, Any]]                    # Postgres only, decided

ctx.create(Entity, **fields) -> E           # registered in the session, inserted at flush
ctx.delete(row)                             # deleted_at = now()

ctx.save(name, value)                       # free-form, held by the session for the run
ctx.load(name) -> Any                       # KeyError if absent
ctx.vars                                    # the dict behind save and load, for `name in ctx.vars`

ctx.secret(name) -> str                     # SECRET_<NAME> from the environment; SecretError if unset

await ctx.trigger(Reaction, input, timeout=None) -> Context        # runs now; returns the child context
await ctx.trigger(Reaction, input, deferred=True, timeout=None)   # outbox row at flush; runs after commit
```

`get` and `get_by_id` are awaited because the Postgres store does I/O and the interface has one shape. Today the in-memory implementation returns immediately.

`query` runs a native statement on the backing store and returns instances of the named entity built from the rows, merged into the session's identity map. `query_raw` returns untracked dicts for aggregates and joins. Both raise `NotSupported` on the in-memory store. Section 13 specifies the row-to-instance conversion.

The session itself is not exposed. A reaction holds a context and nothing else.

## 6. Session and flush

Built.

The session is an `InMemoryStore` with a backing store and a flush.

**Loading.** `get_by_id` checks the working set, then the backing store, and copies a found row in. `get` loads the whole entity from the backing store once per run, then filters locally; this is why it is in-memory only. `query` and `query_raw` first flush the session's pending writes to the backing store, inside the run's transaction, so the statement sees everything the run has done so far, then `query` merges the rows the statement returns. In every case, an instance already in the working set is kept and the freshly loaded copy discarded, so earlier mutations in the run are never overwritten.

**Creating.** `create` constructs the instance and registers it as created. It never touches the backing store.

**Vars.** `session.vars` is a plain dict. `ctx.save` and `ctx.load` read and write it. Shared by every reaction in the run, readable after the run by whatever consumes it, never persisted.

**Flush.** Called once by `run` after the seed reaction returns. Created rows are inserted. For each loaded row the session compares the instance to a snapshot of its field values taken at load, and passes only the changed fields to the store's `update`; an unchanged row produces no write, and a value changed and changed back within the run is unchanged. Nested mutations are caught because the comparison is of the whole value. The in-memory store applies the changes in place; Postgres issues an `UPDATE` setting only those columns. Flush never upserts: a created row whose id exists is a primary key violation, not an overwrite.

**Discard.** An exception anywhere in the run propagates out of `run` before `flush`. The session is garbage and the store is as it was.

## 7. Stores

Both stores are built.

**The interface** is what the session needs and nothing more:

| Method | Purpose |
| --- | --- |
| `get_by_id(entity, id, *, include_deleted)` | one row by primary key |
| `get(entity, where, *, include_deleted)` | rows matching a Python predicate; native to the in-memory store, raises elsewhere |
| `query(entity, statement, **params)` | rows matching a native statement; native to Postgres, raises elsewhere |
| `create(entity, **fields)` | insert; `DuplicateKeyError` if the id exists |
| `update(entity, id, changes)` | apply changed fields by id; `MissingRowError` if absent |
| `run_scope()` | async context held for a whole run |
| `ping()` | reachable, for the health endpoint |

Each store keeps its native reads native. The interface does not define a portable query language, and a reaction that needs one store's read says so at the call site.

**In-memory store.** `entity -> id -> instance`. Rows returned by reference. `run_scope` holds an `asyncio.Lock`, so runs on one store execute one at a time; that is what makes concurrent requests safe without transactions, and it is the throughput ceiling of this backend. Enforces the primary key and nothing else. Does not serialize values. State lives in the process and dies with it.

**Postgres store.** Built. SQLAlchemy Core with asyncpg. `run_scope` opens a transaction at `SERIALIZABLE` and holds its connection in a context variable; every store call inside the run uses it; it commits on exit. Calls outside a run use their own short transaction. A serialization failure, a deadlock, or a primary key violation on `id` raise `TransientError`, which the engine's retry handles like any other exception. `get` raises `NotSupported`. `query` and `query_raw` run statements with bound parameters and return exactly what the statement returns, with no `deleted_at` filter added; `uuid` values from raw statements are normalised to strings. One engine, so one connection pool, per running event loop, because asyncpg connections are loop-bound. Reactions writing raw SQL quote `"order"` themselves.

## 8. Run lifecycle

Built.

```python
ctx = await run(store, reaction, input, reaction_id=None)
```

1. Open a session over the store.
2. Build the root context with the input, a fresh or supplied reaction id, and no parent.
3. Enter `store.run_scope()`.
4. Open the `run` span. `invoke` the seed reaction; every trigger inside it runs depth-first and returns before the seed does.
5. Open the `flush` span. Flush the session.
6. Leave the scope. Log the trace, success or failure. Return the root context.

An exception at step 4 or 5 propagates; steps 5 and 6 still log; nothing is written.

**Retry.** Any exception discards the session and the run starts again from the seed with a fresh session, up to the app's `max_attempts`, three by default, with a short jittered pause. One policy for every reaction of the app, over HTTP and the CLI. A run is a function of its input and the store, so a retry is safe; a reaction that does something outside the session would repeat it. The last attempt's exception propagates.

**Concurrency.** Within a run there is one task. Across runs on the in-memory store, the lock serialises them. Across runs on Postgres, serializable isolation and retry handle interaction.

## 9. App and CLI

Built.

**App** is the single declaration of scope:

```python
app = App(package="shop", endpoints=endpoints, triggers=triggers, store=None, include=(), max_attempts=3)
```

On construction it imports every module under its package with `pkgutil.walk_packages`, so all reactions and entities are registered, then derives `reactions` and `entities` as those whose qualified name starts with the package. `include` adds classes from other packages explicitly, because a shared entity means a table in this app's database and that should never be implicit. `resolve(name)` maps a qualified name to one of the app's reactions and nothing else. `http(store)` builds the FastAPI app. `describe()` lists everything.

An app is any importable package. The monorepo keeps apps under `apps/`, but the runtime depends only on the package name, so the library works the same when installed from PyPI into someone else's repository. Entities and reactions may live in any module at any depth under the package.

**The CLI**, `backend-builder`, is the one launcher. Each command imports an app by package name and reads its `app`:

| Command | Does |
| --- | --- |
| `serve <package> [--host --port --reload --log-level]` | the HTTP process |
| `trigger <package> <reaction> --input <json>` | one reaction against a fresh store from the app's factory |
| `graph <package>` | prints entities, reactions, endpoints, triggers |
| `worker <package> [--once]` | delivers outbox rows |
| `scheduler <package> [--once]` | keeps each schedule's next occurrence in the outbox; purges old run records |
| `admin <package> [--host --port]` | the admin console |
| `outbox <package>` | prints pending, claimed and dead counts |
| `makemigration <package> [-m] [--empty] [--check] [--merge]` | decided, section 13 |
| `migrate <package>` | decided, section 13 |

## 10. HTTP layer

Built.

An `Endpoint` binds a method and path to an API reaction, with one Pydantic model per request source:

```python
Endpoint(
    method="POST", path="/orders/{order_id}/lines",
    reaction=AddLineApi,
    path_model=OrderPath,       # one field per {name}; types convert
    query=PageQuery,
    body=AddLineBody,
    headers=TenantHeaders,      # x_tenant_id reads X-Tenant-Id
    response=LineResponse,
    build=line_response,
    timeout=None,               # seconds; None is the app default
)
```

Each source is optional on any method. FastAPI validates and documents each one as declared; a 422 names its source. The reaction input keeps them apart, `{"path": ..., "query": ..., "body": ..., "headers": ...}`, with only declared sources present. A path model whose fields do not match the placeholders is rejected at declaration.

**API reactions** translate the structured input into domain inputs and trigger domain reactions. Domain reactions never see HTTP shapes. The translation has its own reaction id and span.

**Builders** are async functions that run after the run, read the root context, and return the response, validated against the response model. They read `ctx.load`, `ctx.vars`, `ctx.get_by_id` and the rest; they do not write. A reaction that produced something for the response saves it under a name; the builder loads it.

**Outcomes versus failures.** A run either commits or is a 500. Everything a reaction understands, an authentication failure, a missing resource, a rejected input, is an outcome, not a failure: the reaction records it with `ctx.save` and returns, the run commits having written nothing, and the builder returns `Reply(status=404, body=ErrorOut(...))` instead of the response model. `Endpoint(responses={404: ErrorOut})` documents the alternatives in the OpenAPI document. Authentication and authorization are written this way, as ordinary reactions an app defines once and its API reactions trigger first; the engine has no auth of its own and no 401 of its own.

**Generic route.** `POST /reactions/{qualified_name}` triggers any of the app's reactions with the JSON body as input and returns `{}`. A name that is not one of the app's reactions is a 404. Off with `generic=False`.

**Responses** carry `X-Reaction-Id`. A run failure is a 500 with the exception type, message and reaction id; the store is untouched. Every other status is a `Reply` from a builder; the engine does no mapping of its own. Nothing about the run's internals is returned; that goes to logs.

**`GET /health`** returns the store's name and `ok`, or 503. Hidden from the schema.

## 11. Logging and tracing

Built.

The package logs under `backend_builder` with a `NullHandler`; applications call `configure_logging(level)`. INFO carries identifiers and counts, never payloads. DEBUG carries inputs and rows.

| Event | Level |
| --- | --- |
| reaction started, finished, failed | INFO, with name, reaction id, parent, duration |
| reaction input | DEBUG |
| flush started, finished | INFO, with counts |
| flush insert, update | DEBUG, with the row |
| run finished | INFO, with total ms, span count, failed span names |
| span tree | DEBUG, one line per span indented by depth |

A `Trace` per run holds a `Span` per invocation plus `flush` and `run`: name, reaction id, parent id, `started_at`, duration, ok. `trace.total_ms()` is the `run` span. The trace is logged at the end of every run and is otherwise internal.

## 12. Settings

Built.

`RUNTIME_ENV` is `test`, `dev` or `prod`; unset means `dev`; anything else is an error. `BACKEND_BUILDER_DATABASE_URL` unset means the in-memory store; set means Postgres at that URL; `prod` refuses to start without it. `BACKEND_BUILDER_LOG_LEVEL` defaults to `INFO`. `Settings.from_env()` reads them; `App.create_store()` applies them unless the app was given an explicit store factory; every CLI command reads the log level from them unless a flag overrides. Further layering, TOML files and per-app settings, waits until there is more than one setting worth layering.

## 13. Schema, migrations and the Postgres mapping

Decided.

**Tables and columns come from entities.** Table name is the class name in snake case. Columns are the persisted fields:

| Python | Postgres |
| --- | --- |
| `id` | `uuid primary key` |
| `str` | `text` |
| `int` | `bigint` |
| `float` | `double precision` |
| `Decimal` | `numeric` |
| `bool` | `boolean` |
| `datetime` | `timestamptz` |
| `date` | `date` |
| `dict`, `list`, nested dataclass, entity as value, unions of these | `jsonb` |
| `X | None` | nullable |

Anything else fails at `App` construction.

**Everything else comes from the app, as SQL.** Foreign keys, unique constraints, indexes, triggers, extensions, check constraints. Autogenerate is configured to compare only tables and columns, so it never drops what it did not declare.

**One migration chain.** `apps/<app>/migrations/versions/` holds every migration, generated or hand-written, each pointing at its parent. `makemigration <app> -m "..."` autogenerates from entity changes. `makemigration <app> --empty -m "..."` scaffolds a file for SQL. `migrate <app>` applies pending migrations in order. `makemigration --check` fails if entities differ from what the migrations produce; CI runs it, so an entity change cannot merge without its migration, and a migration cannot merge without its entity change. `--merge` joins two heads from parallel branches.

Autogenerate sees a renamed field as drop plus add. The generated file is reviewed and edited before commit; that is where a rename becomes `alter_column`.

**Pipeline.** Developers generate migrations locally and commit them. CI applies the chain to a fresh database, runs the drift check, and runs the test suite against that database. Deploy applies the chain to the target database, then rolls the services. A migration must therefore be compatible with the version still running: add before reading, drop one release after writing stops.

**Row to instance.** The store builds instances from rows by the entity's fields: for each field, read the column of the same name and convert the backend value to the field type, `uuid` to `str`, `timestamptz` to aware `datetime`, `numeric` to `Decimal`, `jsonb` by the field's annotation, recursively, with nested dataclasses reconstructed and ISO strings parsed back into datetimes where a nested field says so. A missing column is an error unless the field has a default. Extra columns are ignored, so a joined query with aliases works. Then `Entity(**values)`.

**Instance to row.** The reverse, on flush. Dicts and lists recurse, dataclasses become dicts, datetimes and dates become ISO strings, `Decimal` becomes a string. Anything else raises at flush naming the entity, field and type.

**Local Postgres for tests.** `docker compose up -d` starts Postgres 17 from `compose.yaml`. The test harness creates one empty database per test on it and drops it afterwards; tests that need it skip when the server is unreachable, except in CI, where a Postgres service container must be present.

**One database per app** by convention; the URL is the user's.

## 14. Order of work

1. Async reads and `query` on the interface, in-memory raising on `query`; shop and fixture apps updated.
2. Snapshot-and-compare flush. Done.
3. Field-to-column mapping and Alembic: `makemigration`, `migrate --check`, `--empty`; the shop app's first migration committed. Done.
4. `PostgresStore` with `run_scope`, retry, `query`; the `backend_builder_testing` pytest plugin with per-test databases; the shop suite on Postgres. Done.
5. Settings and the store factory. Done.

## 15. Later

Each of these was designed in the earlier draft and is parked, with what has to exist first.

- **Deferred execution.** Built; see section 16.
- **Data-change triggers and schedules.** Built; see section 16.
- **Kubernetes.** Per [infrastructure.md](infrastructure.md). Docker is built; see section 17.

## 16. Deferred execution

Built.

**The outbox** is a system table, `_outbox`, in every app's database, added to the metadata automatically and so to the app's migration chain. A row is `(id, reaction, input jsonb, created_at, available_at, attempts, last_error, claimed_by, claimed_at, dead_at)` with a partial index on `available_at` where `dead_at is null`. Rows are written by the run's flush, in the run's transaction, so a row exists if and only if the run committed.

**Producing.** `await ctx.trigger(R, input, deferred=True)` records an entry in the session; the input is checked to be JSON-encodable at that line, not at flush. Flush writes the rows after the entity writes. Postgres only: the in-memory store raises `NotSupported`.

**Delivering.** `backend-builder worker <app>` polls. Each poll claims up to `BACKEND_BUILDER_OUTBOX_BATCH` due rows with `FOR UPDATE SKIP LOCKED`, stamping `claimed_by` and `claimed_at`; a row whose lease, `BACKEND_BUILDER_OUTBOX_LEASE_SECONDS`, has lapsed is claimable again, which is how a crashed worker's rows return. Any number of workers may run against one database; they divide rows without coordination. Within a worker, a batch is delivered with `BACKEND_BUILDER_OUTBOX_CONCURRENCY` deliveries in flight. A delivery is an ordinary `run` with the app's retry policy; the outbox row is deleted inside that run's transaction after the flush, so the run's writes and the row's removal commit together, and the database effects of a deferred reaction happen exactly once. Anything a reaction does outside the database is at least once.

**Failing.** A failed delivery, after the run's own retries, is released: `attempts` increments, `last_error` is recorded, the claim clears, and `available_at` moves out by an exponential jittered backoff capped at five minutes. At `BACKEND_BUILDER_OUTBOX_MAX_ATTEMPTS` the row is marked `dead_at` and never claimed again; it stays for inspection and is requeued by clearing `dead_at`. A row naming a reaction the app does not have is released the same way.

**Chains.** A deferred reaction may trigger immediately, defer further work, and when data-change triggers exist, cause more rows; it is a run like any other.

**Observing.** `backend-builder outbox <app>` prints pending, claimed and dead counts and the oldest pending time. The poll interval is `BACKEND_BUILDER_OUTBOX_POLL_SECONDS`.

**Data-change triggers.** Declared on the `App`, passed through to every run, and matched at flush against what the session wrote:

```python
triggers = [
    OnCreate(OrderLine, OrderTotal),
    OnChange(OrderLine, ["price", "qty"], OrderTotal),
    OnDelete(OrderLine, OrderTotal),
]
```

`OnCreate` matches every row inserted by the run. `OnChange` matches a loaded row when at least one listed field differs from its load snapshot; a change reverted within the run does not count, and a row whose `deleted_at` went from null to set is a delete, not an update. `OnDelete` matches that transition. Each match is one outbox row, so a run that creates two lines enqueues the reaction twice. The input names what happened and nothing more; the reaction loads current state itself:

```python
{"entity": "order_line", "id": "...", "kind": "created"}
{"entity": "order_line", "id": "...", "kind": "updated", "changed": ["qty"]}
{"entity": "order_line", "id": "...", "kind": "deleted"}
```

Declarations are validated when the `App` is built: fields must exist on the entity, and the entity and reaction must belong to the app. An app that declares triggers refuses the in-memory store, since it has no outbox and they would silently never fire.

**Schedules.** `OnSchedule("0 3 * * *", NightlyRollup)` is declared alongside the others. The outbox carries time, so a scheduled occurrence is an outbox row with `available_at` and `scheduled_for` set to that time, and the worker delivers it like any row; the worker knows nothing about schedules. A separate process, `backend-builder scheduler <app>`, keeps the rows coming: on every tick, `BACKEND_BUILDER_SCHEDULER_TICK_SECONDS` (default 15, which must stay under cron's one-minute granularity), it reads each schedule's `last_scheduled_for` from the `_schedule` system table, keyed by reaction and cron so a reaction may run on several schedules, under a row lock and, if that time has passed, enqueues the next occurrence after now and advances the record. An edited cron is a new row; the old one goes idle. Declaring the same cron twice for one reaction is refused when the `App` is built. The unique index on `(reaction, scheduled_for)` makes this safe for any number of schedulers and across restarts. Occurrences that fall while no scheduler runs are skipped, as cron does. The reaction's input is `{"scheduled_for": "<ISO timestamp>"}`. A failed scheduled run follows the ordinary release and dead path and does not stop the schedule.

**The eight ways a reaction runs.** Endpoint, generic endpoint or CLI, immediate trigger from a reaction, deferred trigger from a reaction, `OnCreate`, `OnChange`, `OnDelete`, `OnSchedule`. Every one ends in the same `run` with the app's retry policy and the app's triggers, so a deferred or scheduled reaction can trigger immediately, defer further work, or cause more data-change rows exactly as a request-driven one can. The reaction never knows which started it.

## 17. Timeouts

Built.

**One deadline per run.** `App(timeout=60)` is the budget for a whole run: the seed reaction, everything it triggers immediately, the flush, and every retry attempt together, so three attempts cannot take three times the budget. `run` wraps the attempt loop in `asyncio.timeout`. When it expires the current attempt is cancelled, its transaction rolls back, and `TimeoutError` propagates: a 500 over HTTP, a released outbox row in the worker, an error from the CLI. A timeout is never retried. On Postgres the cancelled statement is abandoned and the connection discarded; the store is usable again immediately.

**Per-endpoint budgets.** `Endpoint(..., timeout=5)` bounds that endpoint's run and its response builder together, tighter or looser than the app default, since each request is its own run. The generic route and the CLI use the app default. A client that gives up first does not cancel the run; it continues to its own deadline and commits if it finishes. The builder runs after the commit, so a budget that expires during the builder returns a 500 for a run that did commit; builders are expected to be trivial.

**Per-trigger budgets.** `await ctx.trigger(R, input, timeout=5)` bounds an immediate sub-reaction within the run's remaining budget; it can only tighten, because a transaction cannot be extended from inside, and its expiry aborts the whole run, because a cancelled transaction cannot be resumed. `await ctx.trigger(R, input, deferred=True, timeout=3600)` sets the budget of that delivery's own run and may be any value. `OnCreate`, `OnChange`, `OnDelete` and `OnSchedule` take `timeout=` for the rows they produce. `None` everywhere means the app default.

**The outbox carries the budget.** Every row has `timeout_seconds`, filled at enqueue. The worker runs each delivery under its row's budget, and a worker's lease on a row is that budget plus `BACKEND_BUILDER_OUTBOX_LEASE_GRACE_SECONDS` (default 10). A row is therefore never claimed twice while it is legitimately running, however long that is, and a short job's row is reclaimable seconds after a worker dies. There is no global lease.

**Long work.** A run is one serializable transaction. Held for an hour it conflicts with every concurrent write to what it read, and a retry starts from zero; that is acceptable when the hour is spent waiting on an external system and wrong for mass writes. The shape for bulk work is an orchestrator reaction that queries ids and defers one chunk reaction per few hundred rows: each chunk is a short transaction, retried on its own, spread across the workers. Fan-in, running something after all chunks complete, is not built; it would be a completion counter entity with an `OnChange` on it, and is a later discussion.

**Triggers are sequential.** A run has one connection and one connection executes one statement at a time, so sub-reactions run one after another. The earlier idea of gathering several triggers concurrently is dropped.

## 18. Run records

Built.

Every run leaves a record: what it was, where it came from, what came in, what the reactions saved, what was written, what was deferred, how it ended, and every attempt's spans. Two system tables hold them, `_run` and `_span`. A `running` record is written before the first attempt and replaced, by id, after the run finishes, each in a short transaction of its own, never the run's, so a failed or timed-out run is recorded with its error, a committed run with its changes, and a run in progress is visible while it runs. A `running` record whose deadline has passed means the process died before it could report; the scheduler marks it `lost` on every tick, after a short grace period. Recording never fails a run: an error while writing a record is logged and swallowed, and a store without records, the in-memory one keeps a plain list, is simply skipped.

A `_run` row: the run's reaction id as key, the reaction, `source` (`endpoint POST /orders`, `generic`, `cli`, `outbox`, `schedule`, `direct`), `outcome` (`running`, `ok`, `error`, `timeout`, `lost`), the error, attempt count, start time, duration, `input`, `vars`, `changes` (entity, id, kind, changed fields per row written) and `enqueued` (the outbox rows produced). A `_span` row per span of every attempt, tagged with the attempt number, so a retried run shows why its first attempt failed. Vars are encoded like `jsonb`; a value the codec rejects is stored as its `repr`.

`BACKEND_BUILDER_RUN_RECORDS` is `full` (default), `summary` (no input or vars) or `off`. `BACKEND_BUILDER_RUN_RETENTION_DAYS` (default 7) bounds the tables; the scheduler purges older records once an hour. These tables are what the admin console reads; they are never returned by the API.

## 19. Metrics

Built.

Every process serves Prometheus-format metrics and a health check on its own port, `BACKEND_BUILDER_METRICS_PORT` (default 9100), separate from the API's listener so metrics never sit on a public port. Metrics live in the process's memory and are updated where things happen; `/metrics` renders them for a scraper to poll; nothing is pushed. Labels are bounded, reaction names, sources, outcomes, route templates and statuses, never a reaction id.

| Metric | Type | Updated |
| --- | --- | --- |
| `bb_runs_total{reaction, source, outcome}` | counter | end of every run |
| `bb_run_duration_seconds{reaction}`, `bb_run_attempts{reaction}` | histograms | end of every run |
| `bb_span_duration_seconds{name}` | histogram | every span of every attempt |
| `bb_http_requests_total{route, method, status}`, `bb_http_request_duration_seconds{route, method}` | counter, histogram | API middleware, by route template |
| `bb_outbox_deliveries_total{reaction, outcome}` | counter | worker, per delivery: `ok`, `error`, `unknown` |
| `bb_outbox_pending`, `bb_outbox_claimed`, `bb_outbox_dead`, `bb_outbox_oldest_pending_seconds` | gauges | worker, refreshed on every poll |
| `bb_schedule_seeds_total`, `bb_run_records_purged_total`, `bb_runs_lost_total` | counters | scheduler |

**No application metrics.** These are the runtime's metrics and the only ones. A reaction that wants to record that something happened creates an entity: an event is a row, committed with the run, so a retried attempt never double-counts, consumers hang off `OnCreate`, SQL answers the questions, and a scheduled reaction rolls it up or purges it. A reaction that wants to measure something is itself a reaction, scheduled or triggered. There is no `ctx.metric`, no free-form counters, and no registry of app metrics to review; decided, see §23.

Outbox depth and the oldest row's age are the worker autoscaling signals; runs by outcome is the error-rate alert; dead rows above zero is a page. The compose stack publishes each process's metrics port, and the e2e suite scrapes all three.

## 20. The admin console

Built.

`backend-builder admin <app>` serves the console on `BACKEND_BUILDER_ADMIN_PORT` (default 9200), a separate process on its own port, never the API's. It is server-rendered HTML with no build step, lives in the core, and reads the run records, the outbox and the schedule table alongside the app's own declarations. `BACKEND_BUILDER_ADMIN_TOKEN`, when set, is asked for once on a login page and kept in a cookie; it is also accepted as an `X-Admin-Token` header. The console belongs behind network policy regardless.

| Page | Shows |
| --- | --- |
| Overview | counts of reactions, entities, endpoints and triggers; runs by outcome in the last 24 hours; outbox depth; recent runs |
| Graph | entities with their tables and fields; reactions with what reaches them, endpoints, triggers, schedules, and their docstrings; endpoints with their budgets |
| Reaction | one reaction: what reaches it, runs by outcome with p50 and p95 over the last 24 hours, outbox rows waiting for it, recent runs |
| Runs | filter by reaction, outcome and source prefix |
| Run | outcome, source, timing, attempts, the error; the span tree per attempt with sub-triggers nested; the input; the saved vars; the changes written; the deferred rows and their state |
| Outbox | pending, claimed and dead rows; a row's input, error and budget; requeue and discard |
| Schedules | each schedule's cron, last seeded occurrence and next due; declared schedules not yet seeded |

Requeue clears `dead_at` and the attempt count and makes the row due now; discard deletes it. Those are the only writes the console makes. On the in-memory store the console serves the graph and says the store keeps no records.

## 21. The image, the stack and the end-to-end suite

Built.

**One image per app.** The `Dockerfile` builds `backend-builder-<app>` from the lock file with `uv`, third-party dependencies as a cached layer, the workspace sources on top, a non-root user, and the `backend-builder` CLI as the entrypoint. The process is the command: `serve`, `worker`, `scheduler`, `migrate`.

**The stack.** `compose.yaml` has a `shop` profile with `migrate` as a one-shot service, then `api`, `worker` and `scheduler` from the same image, each depending on `migrate` having completed, all pointed at the `postgres` service with `RUNTIME_ENV=prod`. `docker compose --profile shop up --build` is the whole system.

**The end-to-end suite** in `apps/shop/tests/e2e` is the repository's standard for what "working" means. Its session fixture builds the image, brings the stack up under its own compose project and ports with fast poll and tick settings, and tears it down afterwards. The tests drive the published API with an HTTP client and observe the published database and the container logs. They cover: every service up and healthy, migrations at head; the OpenAPI document; an order created with lines, committed, then totalled by the worker through `OnCreate`; a line repriced and deleted, retotalled through `OnChange` and `OnDelete`; ten concurrent orders all totalled; validation errors naming their source; a missing order answered with a 404 reply from a run that committed; a failing run returning a 500 with its correlation id, writing nothing, and logging the failure; the generic trigger route scoped to the app; the scheduler seeding the schedule, the worker delivering it and expiring a stale order, and the scheduler re-seeding; an undeliverable row retried with backoff and marked dead, visible through the `outbox` command; a secret reaching a deferred reaction and appearing in its run record and the container logs only as its placeholder; and the worker stopped, work accumulating, and the worker restarted and catching up. `pytest -m e2e apps/shop/tests/e2e` runs it; the regular `pytest` excludes it; CI runs both.

## 22. Secrets

Built.

A secret is a value a reaction needs and the repo must never hold: a payment provider's key, a mail service token. Reactions read one with `ctx.secret("stripe_api_key")`, which returns the environment variable `SECRET_STRIPE_API_KEY`: the name lowercase snake case, the variable the same name upper-cased behind the `SECRET_` prefix. The prefix is deliberately not `BACKEND_BUILDER_`: these are the app's secrets, not the runtime's settings, and it reads cleanly in compose files and manifests. A name that is unset or empty raises `SecretError`, which fails the reaction like any other exception.

The library has no secret store. The platform running the process owns storage, rotation and audit, and every platform we would deploy to can inject environment variables from its own store: compose `secrets:` or an `.env` file, Kubernetes Secrets, a cloud secret manager through an external-secrets operator. Variables are read on access, not at startup, so the runtime does not need to know which secrets an app uses. A file-mounted or cloud-backed provider behind the same `Secrets` object is the obvious extension and needs no change to reactions.

**Redaction** is the part that needs library support, because run records store inputs and vars and the admin console shows them. Every value read through `ctx.secret` is remembered by the process, and wherever text leaves the process it is masked as `[secret:name]`: the run record's input, vars and error text, and every line through the configured log handler. Redaction is by exact value, so an encoded or truncated secret still leaks; it is a seatbelt, not a vault. Values shorter than eight characters are not masked, since replacing every "test" in the logs is worse than the leak; reading one logs a warning.

**Not built, discussed.** Per-reaction grants (a reaction declares which secrets it may read) and input-scoped secrets (a tenant's own key, chosen by a field of the input, never by the reaction). Both fit as declarations on the reaction class with the provider taking a scope, and would be least privilege at the framework level and a review surface rather than a sandbox, since every reaction shares one process and one environment. Deferred until an app needs them.

## 23. Decisions

Settled in discussion, listed so they are not reopened by accident.

- Ids are client-generated UUID strings. No sequences, no pending keys.
- `updated_at` is not a system field; mutation is unobserved.
- Entities carry no storage configuration. Constraints and indexes are SQL migrations owned by the app.
- Many-to-many is a link entity. Navigation is a non-persisted field a reaction fills. No automatic relation loading.
- The in-memory store is native: Python predicates, references, no serialization, a run lock instead of transactions.
- `get` with a predicate is in-memory only; `query` with SQL is Postgres only. No portable query language.
- Flush inserts created rows and updates loaded rows. Never upserts.
- Any exception aborts the run. Every exception is a 500 over HTTP. Any other status is an outcome a reaction recorded and a builder replied with; there is no exception-to-status mapping.
- Nothing about a run's internals is returned over HTTP. Traces go to logs.
- Context input is a dict. API reactions build domain inputs from the structured request.
- Reactions and entities belong to the app whose package defines them. Shared classes are included explicitly.
- The CLI is in the core library and is the only process launcher.
- Migrations are written by developers and reviewed in pull requests. The pipeline verifies, tests, applies and deploys; it never generates.
- Secrets are environment variables under `SECRET_`, read on access. The library redacts what it has read from records and logs; it stores nothing.
- No application metrics API. Anything worth recording is an entity; anything worth measuring is a reaction. Prometheus carries the runtime's metrics only.
