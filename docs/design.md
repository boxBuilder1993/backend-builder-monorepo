# Backend Builder Design

Status: current as of 2026-10-21. Describes what is built on `main` and what is decided but not yet built. Each section says which. The earlier draft, with the reconcile loop, consumer config and WAL worker, is superseded; those ideas are listed under "Later" with what has to be true before they return.

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

**Trigger.** `await ctx.trigger(reaction, input)` runs a reaction now, with a child context on the same session. The only way one reaction causes another to run.

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

Built, except `query`, which is decided.

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

await ctx.trigger(Reaction, input) -> Context   # runs now; returns the child context
```

`get` and `get_by_id` are awaited because the Postgres store does I/O and the interface has one shape. Today the in-memory implementation returns immediately.

`query` runs a native statement on the backing store and returns instances of the named entity built from the rows, merged into the session's identity map. `query_raw` returns untracked dicts for aggregates and joins. Both raise `NotSupported` on the in-memory store. Section 13 specifies the row-to-instance conversion.

The session itself is not exposed. A reaction holds a context and nothing else.

## 6. Session and flush

Built; the changed-columns comparison is decided.

The session is an `InMemoryStore` with a backing store and a flush.

**Loading.** `get_by_id` checks the working set, then the backing store, and copies a found row in. `get` loads the whole entity from the backing store once per run, then filters locally; this is why it is in-memory only. `query`, when built, merges the rows the statement returns. In every case, an instance already in the working set is kept and the freshly loaded copy discarded, so earlier mutations in the run are never overwritten.

**Creating.** `create` constructs the instance and registers it as created. It never touches the backing store.

**Vars.** `session.vars` is a plain dict. `ctx.save` and `ctx.load` read and write it. Shared by every reaction in the run, readable after the run by whatever consumes it, never persisted.

**Flush.** Called once by `run` after the seed reaction returns. Created rows are inserted. Loaded rows are updated. On the in-memory store the update replaces the row. On Postgres the session compares each loaded instance to a snapshot taken at load and issues `UPDATE` only for rows that changed, setting only the changed columns. Flush never upserts: a created row whose id exists is a primary key violation, not an overwrite.

**Discard.** An exception anywhere in the run propagates out of `run` before `flush`. The session is garbage and the store is as it was.

## 7. Stores

The in-memory store is built. The Postgres store is decided.

**The interface** is what the session needs and nothing more:

| Method | Purpose |
| --- | --- |
| `get_by_id(entity, id, *, include_deleted)` | one row by primary key |
| `get(entity, where, *, include_deleted)` | rows matching a Python predicate; native to the in-memory store, raises elsewhere |
| `query(entity, statement, **params)` | rows matching a native statement; native to Postgres, raises elsewhere |
| `create(entity, **fields)` | insert; `DuplicateKeyError` if the id exists |
| `update(entity, row)` | replace by id; `MissingRowError` if absent |
| `run_scope()` | async context held for a whole run |
| `ping()` | reachable, for the health endpoint |

Each store keeps its native reads native. The interface does not define a portable query language, and a reaction that needs one store's read says so at the call site.

**In-memory store.** `entity -> id -> instance`. Rows returned by reference. `run_scope` holds an `asyncio.Lock`, so runs on one store execute one at a time; that is what makes concurrent requests safe without transactions, and it is the throughput ceiling of this backend. Enforces the primary key and nothing else. Does not serialize values. State lives in the process and dies with it.

**Postgres store.** SQLAlchemy Core with asyncpg. `run_scope` opens a transaction at `SERIALIZABLE`; the session's loads and the flush use its connection; it commits on exit. A serialization failure, or a unique violation on `id`, aborts the run and the engine retries it from the seed with a fresh session, three attempts by default with jittered backoff. `get` raises `NotSupported`. `query` and `query_raw` run statements with bound parameters. Identifiers are always quoted, because `order` is a table name here.

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

**Concurrency.** Within a run there is one task. Across runs on the in-memory store, the lock serialises them. Across runs on Postgres, serializable isolation and retry handle interaction.

## 9. App and CLI

Built.

**App** is the single declaration of scope:

```python
app = App(package="shop", endpoints=endpoints, store=InMemoryStore, include=())
```

On construction it imports every module under its package with `pkgutil.walk_packages`, so all reactions and entities are registered, then derives `reactions` and `entities` as those whose qualified name starts with the package. `include` adds classes from other packages explicitly, because a shared entity means a table in this app's database and that should never be implicit. `resolve(name)` maps a qualified name to one of the app's reactions and nothing else. `http(store)` builds the FastAPI app. `describe()` lists everything.

An app is any importable package. The monorepo keeps apps under `apps/`, but the runtime depends only on the package name, so the library works the same when installed from PyPI into someone else's repository. Entities and reactions may live in any module at any depth under the package.

**The CLI**, `backend-builder`, is the one launcher. Each command imports an app by package name and reads its `app`:

| Command | Does |
| --- | --- |
| `serve <package> [--host --port --reload --log-level]` | the HTTP process |
| `trigger <package> <reaction> --input <json>` | one reaction against a fresh store from the app's factory |
| `graph <package>` | prints entities, reactions, endpoints |
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
)
```

Each source is optional on any method. FastAPI validates and documents each one as declared; a 422 names its source. The reaction input keeps them apart, `{"path": ..., "query": ..., "body": ..., "headers": ...}`, with only declared sources present. A path model whose fields do not match the placeholders is rejected at declaration.

**API reactions** translate the structured input into domain inputs and trigger domain reactions. Domain reactions never see HTTP shapes. The translation has its own reaction id and span.

**Builders** are async functions that run after the run, read the root context, and return the response, validated against the response model. They read `ctx.load`, `ctx.get_by_id` and the rest; they do not write. A reaction that produced something for the response saves it under a name; the builder loads it.

**Generic route.** `POST /reactions/{qualified_name}` triggers any of the app's reactions with the JSON body as input and returns `{}`. Off with `generic=False`.

**Responses** carry `X-Reaction-Id`. A run failure is a 500 with the exception type, message and reaction id; the store is untouched. Nothing about the run's internals is returned; that goes to logs.

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

Decided.

`RUNTIME_ENV` is `test`, `dev` or `prod`. `BACKEND_BUILDER_DATABASE_URL` unset means the in-memory store; set means Postgres at that URL. The app's default store factory reads these. Further layering, TOML files and per-app settings, waits until there is more than one setting worth layering.

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

**Local Postgres for tests.** The `pgserver` package bundles Postgres and starts it from `pip install`, with no Docker and no sudo. One server per test session; one database per test, created from a template and dropped. CI uses the same package.

**One database per app** by convention; the URL is the user's.

## 14. Order of work

1. Async reads and `query` on the interface, in-memory raising on `query`; shop and fixture apps updated.
2. Snapshot-and-compare flush.
3. Field-to-column mapping and Alembic: `makemigration`, `migrate`, `--check`, `--empty`; the shop app's first migration committed.
4. `PostgresStore` with `run_scope`, retry, `query`; the `pgserver` test harness; the shop suite running on both stores.
5. Settings and the store factory.

## 15. Later

Each of these was designed in the earlier draft and is parked, with what has to exist first.

- **Deferred execution.** `trigger(..., deferred=True)` writes a row to an outbox table in the same flush, and a worker process polls the table, runs each row as an ordinary run, and deletes it on success or releases it with backoff on failure. Needs the Postgres store, because an in-memory outbox cannot cross processes. Brings `backend-builder worker <app>` and the second container.
- **Data-change triggers.** Reactions that run because a field changed, declared in a config rather than called. The earlier draft matched them against per-pass deltas before commit and against WAL-decoded changes after. The session can now detect changes by snapshot comparison at flush, which is the natural input for the synchronous half. The design will be redone on that basis once deferred execution exists.
- **Scheduling.** Cron entries that enqueue an outbox row when due.
- **Timeouts and concurrent triggers.** A task group per context and a per-trigger timeout.
- **Docker and Kubernetes.** Per [infrastructure.md](infrastructure.md), after settings.

## 16. Decisions

Settled in discussion, listed so they are not reopened by accident.

- Ids are client-generated UUID strings. No sequences, no pending keys.
- `updated_at` is not a system field; mutation is unobserved.
- Entities carry no storage configuration. Constraints and indexes are SQL migrations owned by the app.
- Many-to-many is a link entity. Navigation is a non-persisted field a reaction fills. No automatic relation loading.
- The in-memory store is native: Python predicates, references, no serialization, a run lock instead of transactions.
- `get` with a predicate is in-memory only; `query` with SQL is Postgres only. No portable query language.
- Flush inserts created rows and updates loaded rows. Never upserts.
- Any exception aborts the run. Every exception is a 500 over HTTP. A REST layer above may map codes.
- Nothing about a run's internals is returned over HTTP. Traces go to logs.
- Context input is a dict. API reactions build domain inputs from the structured request.
- Reactions and entities belong to the app whose package defines them. Shared classes are included explicitly.
- The CLI is in the core library and is the only process launcher.
- Migrations are written by developers and reviewed in pull requests. The pipeline verifies, tests, applies and deploys; it never generates.
