# Backend Builder Design

Status: draft for review, 2026-10-17.

## Intent

The goal is for an entire company's business logic to live in a single repository, and for that repository to manage its own infrastructure. Backend Builder is the runtime that makes this practical: a single backend in which the way logic executes, synchronously in a request, in a pool of workers after commit, or on a schedule, is a deployment choice expressed in config, not a boundary expressed in code.

Three separations make that possible. Reactions are the unit of logic, and are pure functions of an input and a store. The engine is the unit of execution, and runs any reaction in any mode. The consumer config is the unit of wiring, and says which data changes cause which reactions to run and whether they run before or after commit. Because the three never leak into each other, the same reaction can be moved between execution modes, pooled across workers, or re-wired to new producers without a line of its code changing.

What the company gets from this:

- **One map of the business.** The config is a complete, printable answer to "what happens when this field changes". Entities and reactions are the whole system; a new engineer reads them and knows it.
- **Consistency by construction.** Derived data is maintained by subscriptions, not by every writer remembering. A new code path writes a field and the subscriptions do the rest.
- **Infra as flags, not rewrites.** Start everything synchronous. Measure. Defer the slow parts and pool the hot ones by changing config. Nothing is rewritten when the shape of the deployment changes.
- **Whole-operation transactions.** A request that touches orders, ledgers and stock is one commit. No sagas, no compensation logic.
- **Uniform testing.** Every reaction is tested by building a context, calling it, and inspecting the store. No HTTP, no service mocks, no queue fakes.
- **A small deployment surface.** One API process type, one deferred worker type, one WAL worker, one database. Scaling is adding workers and assigning them reaction subsets.
- **Incremental analytics with the same machinery.** Analytics tables are entities. Their maintenance is deferred reactions subscribed to source fields. The WAL worker delivers exactly the rows that changed, with before and after, so a metric is updated by removing the old contribution and adding the new one, never by rescanning. The config is the lineage. Backfill is replaying synthetic changes through the same reaction. This replaces a separate change-data-capture pipeline and warehouse transformation layer with configuration.

- **Event-driven precomputation is native.** Systems that listen to a change log and precompute views, caches or denormalised read models are usually built from a change-data-capture tool, a message broker such as Kafka, and a fleet of consumers each managing offsets, retries and dead letters. Here the WAL worker is the capture, the outbox is the broker, the config is the subscription list, and the deferred worker is the consumer group. Adding a consumer is a config line. Latency is the WAL worker's lag plus the poll interval, sub-second with a short poll, which is as near to real time as any commit-then-react system can be. A broker remains useful only for fan-out to systems outside the repo or for long-retention replay by consumers that did not exist when the event happened. If either is needed, a reaction that publishes to a topic is one more consumer, not a change of architecture.

- **Maintainable by AI as well as people.** What makes a system hard to change is the part that cannot be read: wiring that lives in topic names, deploy manifests and other repositories. Here every dependency is a config line, every reaction's whole world is its input and the store, and every run is deterministic given the database. Impact analysis is a grep of the config. A change to one reaction cannot affect another except through declared fields. Tests need a context and a database, nothing else. Reprocessing history after a fix is a CLI call. An AI assistant, like a new engineer, can read the entities, the config and the reactions, and have the whole system.

Two contracts apply to analytics reactions wherever they run. First, only aggregates that decompose are incremental: sums, counts, averages as sum and count, min and max with tombstones. Distinct counts, medians, percentiles and funnels do not decompose and need sketches such as HyperLogLog or t-digest, or periodic full recomputation. Second, a change arriving late must be bucketed by the event's own timestamp, not the time it was processed.

Separating analytics onto its own database, with a columnar or timeseries extension, is a later concern. It would need only an entity-level database tag and a lookup when the deferred worker opens a transaction, with eager reactions remaining single-database. Nothing in a reaction changes when that happens.

The database is an abstraction the engine owns. v1 assumes PostgreSQL and uses it fully. Scaling past a single database, or evaluating alternatives, is a later concern, and the engine's coupling to Postgres is documented in section 12 so that work is informed when it comes. What must be protected as the system grows is the purity of the reaction contract, because every benefit above depends on reactions remaining functions of input and store and nothing else.

## 1. Purpose and scope

Backend Builder is a Python library for building a business backend as a set of entities and the reactions that keep them consistent, on top of PostgreSQL. A developer declares entity classes with typed fields, writes reactions as async functions of a context, and connects them two ways: a reaction can trigger another explicitly, and a config can subscribe a reaction to changes in an entity's fields. The engine owns everything else: the schema, the transaction, lazy loading, change tracking, synchronous propagation before commit, deferred work after commit, and the HTTP surface.

v1 covers: entity classes and schema generation, the context, explicit and config-driven triggers, the reconcile loop, deferred execution through an outbox fed by a WAL worker, serializable transactions with retry, API entities over HTTP, and scheduled triggers. It assumes PostgreSQL and nothing else.

Design principles that settled the hard calls:

- **One mechanism.** Every reaction runs through `trigger(reaction, input)`. Validation, actions, creates and reads are all reactions. Delete is a field write. GET and POST are the same path.
- **The developer owns the contract.** The engine does not police undeclared reads, infer dependencies, prevent slow access patterns, or referee overlapping writes. It makes behaviour deterministic and visible.
- **Pure where it matters.** What is persisted depends only on each instance's initial and latest state. The context carries no knowledge of why a reaction ran beyond the input it was handed.
- **Both directions of wiring.** A producer can name its consumers with an explicit trigger. A consumer can subscribe to a producer in config. Neither is privileged.
- **Any error aborts.** An exception of any kind inside a run rolls back the transaction. There is no special error class.

## 2. Core model

**Entity.** A Python class with typed fields. Persisted entities own a table. API entities have no table; their instances are created by HTTP requests.

**Field.** A typed attribute of an entity. Persisted by default, or marked non-persisted. No input-versus-computed distinction exists; a field is computed only in the sense that some reaction writes it.

**Key.** The identity of an instance, the `id` field on the `Entity` base class. A library object with a pending flag. Shared by reference between an instance and every field that points at it, so resolution at flush needs no bookkeeping.

**Reaction.** A class with one method, `async def calculate(ctx)`. It reads `ctx.input`, pulls what it needs from the store, writes fields, creates instances, triggers other reactions, or raises.

**Trigger.** The one way a reaction runs. `trigger(reaction, input, deferred=False)`. Synchronous triggers run now, sharing the caller's store and transaction. Deferred triggers write an outbox row and run after commit in a fresh context.

**Consumer config.** A module that subscribes reactions to field changes on entities, eager or deferred. Eager subscriptions are matched by the engine against the store before commit. Deferred subscriptions are matched by the WAL worker against committed changes.

**Context.** The unit of work. An input, a store, a transaction. The store is an ORM session in the unit-of-work sense, lazily populated as a reaction pulls entities and props, tracking each instance's initial and latest state.

## 3. Entities and fields

An entity is a class deriving from `Entity`. Its fields are class annotations. The class name, snake-cased, is the table name. The field name is the column name. Entities carry no trigger declarations.

```python
class OrderLine(Entity):
    order_id: int = Field(foreign_key=Order)
    price: Decimal = Numeric(12, 2)
    qty: int
```

**Persisted and API entities.** `Entity` subclasses are persisted by default. `Entity` with `api=True` is an API entity: no table, instances created by HTTP requests, fields carry request data in and response data out. Both kinds live in the store identically. API instances are never flushed.

**System fields.** Declared on the `Entity` base class and inherited.

| Field | Type | Managed how |
| --- | --- | --- |
| `id` | `Key` | Drawn from the entity's sequence at flush. The single primary key. |
| `created_at` | `datetime` | Set at insert from the transaction's `now()`. |
| `updated_at` | `datetime` | Set at insert and on every update from the transaction's `now()`. |
| `deleted_at` | `datetime | None` | Null until `instance.delete()` sets it. |

**Field types.** `int` to `bigint`, `str` to `text`, `bool`, `Decimal`, `datetime`, `date`, `dict` and `list` to `jsonb`. A library type object as the default refines the column: `Numeric(12, 2)`, `Varchar(80)`. Optional fields use `| None`. `Field(...)` carries options: `persist`, `foreign_key`, `default`, `default_factory`, `index`.

**Entity-typed fields are values, not relations.** A field typed as another entity class holds a snapshot of that instance's fields, including `id`, stored as `jsonb`. It is a copy taken at write time. If it should track the source, a consumer subscribed to the source rewrites it. The engine never infers a foreign key from a type. Relations are `int` id columns, with a constraint only if `foreign_key` is set. Reading a snapshot yields an untracked blob: a plain object built from the jsonb, not in the store, with no versions. Mutating it changes only the copy, and the change persists only if the field is reassigned. The developer must know what they are doing.

**Non-persisted fields.** `Field(persist=False)`. Lives in the store for the run, populated by reactions. How API entities assemble responses. `list[Order]` is legal only here.

**Soft delete.** `instance.delete()` sets `deleted_at`. The engine never removes rows. The engine creates a view per entity that filters deleted rows, and reaction SQL targets the view by default. The raw table is reachable under a separate name.

**Writes are buffered.** All field writes are held in the store and flushed before any SQL runs and at the end of the run.

## 4. Keys

Every instance has an `id` of type `Key`: a resolved integer or a pending marker.

- Loaded instances hold a resolved Key.
- `ctx.new(...)` creates an instance with a pending Key owned by the context.
- API instances hold a pending Key that is never resolved.

**Shared by reference.** Assigning `order.id` to `line.order_id` stores the same Key object. `line.order_id is order.id` holds while pending. Resolution happens once, in place.

**Resolution at flush.** Before any SQL runs, the engine groups pending Keys by entity, draws that many sequence values in one statement per entity, and assigns them. Insert order does not matter. Declared foreign keys are `DEFERRABLE INITIALLY DEFERRED`.

**Behaviour.** Hashable and comparable by identity while pending, by value when resolved. Reading the integer of a pending Key raises. The database adapter and the JSON serializer refuse a pending Key. Tracing shows `pending:<n>`.

## 5. Reactions

```python
class order_total(Reaction):
    async def calculate(ctx):
        changes = ctx.input                                   # list of Change(entity, id, before, after)
        order_ids = {c.after["order_id"] for c in changes}
        orders = await ctx[Order].many(order_ids)
        await ctx.load("select 'order_line', id, '*' from order_line where order_id = any(:ids)", ids=order_ids)
        lines = ctx[OrderLine].loaded()
        for order in orders:
            order.total = sum(l.price * l.qty for l in lines.filter(order_id=order.id))
```

**Input.** `ctx.input` is whatever the trigger site passed. The reaction knows its own input shape. Reactions reached from consumer config receive a list of `Change` records, `(entity, id, before, after)`, with `before` and `after` as field dicts. The same shape arrives whether the match happened in the reconcile loop or in the WAL worker, so a reaction does not know or care which.

**Reading.** All reads go through the store, lazily and in bulk. Section 6.1 lists the operations. Reads that hit the database are awaited.

**Writing.** Assigning a field records a write if the value differs from the current one. `ctx.new(Entity, **fields)` creates an instance. Cascades are the developer's concern.

**Triggering.** `ctx.trigger(reaction, input, deferred=False, timeout=60)` returns an awaitable.

- `await ctx.trigger(r, x)` runs `r` immediately with a child context sharing this store and transaction, and returns when it is done. The caller then sees everything `r` wrote.
- Triggers started without an immediate await begin in the order they were triggered. Several may be awaited together with `asyncio.gather`. They interleave at their await points on one connection. This overlaps waits; it is not parallelism, and the store is never touched by two coroutines in the same instant, but a read-then-write across an await is not atomic with respect to a sibling.
- The context owns a task group. Every triggered coroutine is registered in it, and the engine awaits the group when `calculate` returns. A sub-reaction cannot outlive its caller and nothing escapes the transaction. An exception in any task cancels the rest and aborts the run.
- `timeout` bounds the triggered reaction and everything it triggers synchronously. Default 60 seconds, set per trigger. The seed trigger's timeout therefore bounds the whole run, including reconcile. Exceeding it raises and aborts.
- `deferred=True` writes an outbox row `(reaction, input)` in the current transaction. The input must be serializable. Pending Keys in it are resolved before the row is written. The reaction runs after commit in a fresh context with the given timeout.

**Validation and errors.** A reaction that finds invalid state raises. Any exception aborts the run: rollback, outbox rows vanish with it, the caller receives the error. No exception class is special.

**Contracts the developer must know.**

- Explicit triggers are depth-first. Triggering inside a loop over a thousand instances runs the sub-reaction a thousand times. Collect, then trigger once with a set-shaped input.
- Consumers from config run after this reaction returns, in the reconcile loop, in no guaranteed order. Writing a field and then reading something a consumer derives from it sees the old value.
- A sub-reaction or a consumer may overwrite a field this reaction wrote or read. Last write wins. The engine does not referee.
- A loop calling `get` per item issues one query per item. Set-shaped operations exist so that the fast path is the obvious one.

## 6. Context

A context is three things.

**Input.** `ctx.input`, read-only, set at creation. The engine never interprets it.

**Store.** An identity map of instances keyed by `(entity, id)`, or by the pending Key object for new and API instances. Populated lazily as reactions pull. Every instance records `initial`, its state when first loaded or absent if created in this run, and `latest`, its current state. Writes update `latest`. Loaded rows merge into the map; an instance already present is never overwritten by a reload. Child contexts from synchronous triggers share the parent's store.

**Transaction.** Attached when the context runs. Every store load flushes pending Keys and writes first and executes inside it. In-memory reads never flush. `query` and `load` run in the transaction and never commit. Commit belongs to the engine.

**Marks.** For the reconcile loop, each instance also records a mark: the state last seen by consumer matching. Marks are engine-internal and not readable by reactions.

The store is built on SQLAlchemy Core for table metadata, the expression language, connections and transactions, with an async driver, and Alembic for migrations. The session layer is ours because it differs from a conventional ORM session in three places: initial state that survives mid-run flushes, transient API instances, and cell-level loading.

### 6.1 Context API

```python
ctx.input                                       # free-form, read-only

# Entity views: by id, lazy, awaited when they hit the database
ctx[Order]                                      # EntityView
await ctx[Order].get(id) -> Order | None
await ctx[Order].many(ids) -> Collection
ctx[Order].loaded() -> Collection               # what is in the store, no query

# SQL inside the transaction; flush first, never commit
await ctx.query(sql, **params) -> list[Row]     # plain rows, untracked
await ctx.load(sql, **params) -> None           # rows (entity, id, prop) into the store; '*' = all persisted fields

# Creation and deletion
ctx.new(Order, **fields) -> Order
order.delete()

# Triggering
ctx.trigger(reaction, input, deferred=False, timeout=60) -> Awaitable[None]

# Instance
order.id                                        # Key
order.initial                                   # read-only view, None if created this run
order.latest                                    # writable view; the instance proxies to it
order.changed(Order.total) -> bool
order.is_new -> bool

# Collection
coll.ids -> list[Key]
coll.price -> list                              # vectorized read
coll.filter(**eq) -> Collection                 # in-memory filter, no query
for line in coll: ...
len(coll), coll[id]
```

## 7. Consumer config

A module in the repo that subscribes reactions to field changes. Entities do not know their consumers.

```python
consumers = [
    OnCreate(OrderLine, order_total),
    OnChange(OrderLine, [price, qty, order_id, deleted_at], order_total),
    OnChange(Order, [total], notify_total_changed, deferred=True),
    OnSchedule("0 3 * * *", nightly_rollup),
]
```

| Entry | Matches |
| --- | --- |
| `OnCreate(Entity, reaction, deferred=False)` | an instance of `Entity` was inserted |
| `OnChange(Entity, fields, reaction, deferred=False)` | any listed field of an instance changed |
| `OnSchedule(cron, reaction)` | the scheduler fired; always deferred |

Entries are structural: entity and field membership only. Value conditions live in the reaction. The engine can print the whole map without running anything.

**Eager entries** are matched by the API process in the reconcile loop, before commit, against the store.

**Deferred entries** are matched by the WAL worker, after commit, against the committed row images. The reconcile loop ignores them, so nothing is matched twice.

A matched reaction receives a list of `Change` records as input, grouped so that a reaction matched for six instances runs once with six changes.

## 8. Run lifecycle

A run is one context, one transaction, one commit. It starts with a seed trigger from the API process or the deferred worker.

1. **Seed.** Build a context with the seed input. For HTTP, the input is the API request instance, also placed in the store. Attach a transaction.
2. **Run the seed reaction** with `await trigger(reaction, input)`. Explicit triggers inside it run depth-first and complete before it returns. The task group is awaited.
3. **Reconcile.** Breadth-first over consumer config:
    1. Diff the store against the marks. The frontier is every instance with a field whose `latest` differs from its mark, plus instances created since the mark.
    2. Match eager config entries against the frontier. Group by reaction. Build each reaction's input as `Change` records with `before` from the mark and `after` from `latest`.
    3. Run every matched reaction with `await trigger(reaction, changes)`, in no guaranteed order. Each may trigger explicitly, depth-first, and write to the store.
    4. Advance every mark to the current state.
    5. If the frontier was non-empty, repeat from step 1. The run's timeout bounds oscillation.
4. **Flush.** Resolve pending Keys. Insert new instances with `created_at` and `updated_at`. Update instances whose `latest` differs from `initial`, writing only the differing columns plus `updated_at`. Write deferred outbox rows from explicit `deferred=True` triggers.
5. **Commit.**
6. **Respond.** For HTTP, render the API instance from the store.

Any exception at any step rolls back and aborts.

**Why marks.** A field changed once is matched once, because after the mark advances it no longer differs. A field changed and written back between two marks has no delta. A field changed, matched, then changed again produces a second delta, which is correct because consumers must see the new value. Flush uses `initial` versus `latest` regardless, so a field that ended where it started is not written even if consumers ran on it in between.

**Termination.** Equal writes produce no delta. A system at a fixed point produces an empty frontier. Genuine oscillation between consumers runs until the seed trigger's timeout, then aborts with a timeout error that names the reactions in the last level.

## 9. Deferred execution

Deferred reactions run after commit, each in its own context and transaction.

**Outbox.** One table, one row shape: `(reaction, input jsonb, attempts, next_attempt_at, claimed_by)`. Two producers write it.

**Producer 1, explicit deferred triggers.** `ctx.trigger(r, input, deferred=True)` writes a row at flush, inside the originating transaction. A rollback discards it.

**Producer 2, the WAL worker.** Consumes the logical replication stream from a publication over all entity tables using the built-in `pgoutput` decoder. For each committed transaction it receives the changed rows with before and after images. It matches them against deferred config entries, groups by reaction, and writes one outbox row per reaction with the `Change` list as input. It persists the last processed LSN in the same transaction as the outbox inserts, then confirms the slot, so a crash replays from the last confirmed LSN and already-written rows are skipped. Requirements: `wal_level = logical`, a replication slot with `max_slot_wal_keep_size` set, `REPLICA IDENTITY FULL` on entity tables for before-images.

**Deferred worker.** Polls the outbox on a configurable interval. Claims a batch with `SELECT ... FOR UPDATE SKIP LOCKED`. For each row, builds `Context(input)`, attaches a transaction, runs `await trigger(reaction, input)` followed by the same reconcile, flush and commit as any run, then deletes the row. On failure, releases the row with an incremented attempt count and a backoff. Attempt cap configurable. It runs as its own process, as does the WAL worker. Neither runs inside the HTTP server.

**Coalescing.** Rows for the same reaction whose inputs are `Change` lists may be merged by `(entity, id)`: earliest `before`, latest `after`. Rows with free-form input are not merged.

**Scheduler.** `OnSchedule` entries are evaluated by a scheduler that writes an outbox row with the fire time as input when a cron expression is due. No separate mechanism.

**Backfill and replay.** A backfill is a trigger started from outside a run. Two entry points, both of which write ordinary outbox rows so the deferred worker's batching, retries and pooling apply:

- `bb trigger <reaction> --input <json>` writes one row with the given input. A reaction written as a backfiller reads a range from its input, loads rows, and triggers other reactions with whatever inputs they expect. Explicit triggers fan out as they would in any run.
- `bb backfill <reaction> --entity <Entity> --sql <select of ids>` reads matching rows in batches and writes rows whose input is a list of `Change` records with `before` as `None` and `after` as the current row. The reaction cannot distinguish these from WAL-produced changes, so any config-driven reaction is replayable without modification.

A backfill of millions of rows is therefore a queue, not a long-lived process, and runs at whatever rate the deferred workers provide.

## 10. Transactions and concurrency

**Isolation.** `SERIALIZABLE` by default, configurable.

**Retry.** A serialization failure aborts the run. The engine retries from the seed, up to a configurable count with backoff. Safe because a run is a function of its seed and database state, and outbox rows exist only after commit.

**Within a run.** One context, one connection, one thread. Synchronous triggers are function calls. Gathered triggers interleave at await points on the one connection; they do not run in parallel. Nothing in a run is concurrent with itself.

**Across runs.** Separate contexts, separate transactions. Serializable handles their interaction. An aggregate computed through `ctx.query` widens the read set to everything it scanned, so concurrent writes to those rows conflict. Hot counters and wide aggregates belong in deferred reactions.

## 11. HTTP layer

**API entities are endpoints.** `Entity` with `api=True` is reachable at a path derived from its name, overridable on the class. A request creates one instance from the body or query string, places it in the store, and seeds the run with `trigger(reaction, instance)`, where `reaction` is declared on the API entity:

```python
class GetOrder(Entity, api=True):
    order_id: int
    order: Order | None = Field(persist=False, default=None)
    lines: list[OrderLine] = Field(persist=False, default_factory=list)

    reaction = get_order
```

**Method.** GET and POST both create the instance; GET from the query string, POST from the body. Nothing differs afterwards.

**Request shape.** Fields are type-checked and required fields must be present. Beyond that, validation is a reaction raising.

**Response.** After commit the API instance's `latest` is rendered. Entity-typed fields and lists render inline from what is in the store, recursively, with no loading during rendering. An instance not in the store renders as its bare id.

**Errors.** Any exception rolls back and renders as a 500 with the exception type, message and a correlation id. The engine does no other status mapping. A REST layer built above API entities may inspect the context and the exception to choose its own codes.

## 12. Persistence and schema

Generated from entity classes through SQLAlchemy Core metadata. Alembic produces migrations from the diff.

| Object | Shape |
| --- | --- |
| Table `<entity>_t` | `id bigint primary key default nextval(...)`, `created_at`, `updated_at`, `deleted_at`, one column per persisted field; `REPLICA IDENTITY FULL` |
| Sequence | backing `id` |
| View `<entity>` | `select * from <entity>_t where deleted_at is null` |
| Publication | all `<entity>_t` tables, for the WAL worker |
| `_outbox` | `reaction, input jsonb, attempts, next_attempt_at, claimed_by` |
| `_wal_position` | last confirmed LSN |
| `_schedule` | cron registrations and last-fired timestamps |

Foreign keys only where declared, always `DEFERRABLE INITIALLY DEFERRED`. Indexes on `deleted_at`, declared foreign keys, and fields marked `index=True`.

**Postgres coupling.** The context, reactions, triggers, config and reconcile loop are database-agnostic. The engine relies on Postgres for six things, listed so that any future alternative knows what it must provide: bulk sequence draws for Key resolution; serializable isolation and its abort semantics; a JSON column type; views; `FOR UPDATE SKIP LOCKED` for outbox claims; and logical replication with `pgoutput` for the WAL worker. The last is the most Postgres-specific. Its fallback is the engine writing change rows to the outbox at flush, which needs nothing from the database.

## 13. Worked example

```python
class User(Entity):
    email: str

class Order(Entity):
    user_id: int
    total: Decimal = Numeric(12, 2)

class OrderLine(Entity):
    order_id: int = Field(foreign_key=Order)
    price: Decimal = Numeric(12, 2)
    qty: int

class CreateOrder(Entity, api=True):
    user_id: int
    lines: list[dict]
    order: Order | None = Field(persist=False, default=None)
    reaction = create_order

class GetOrder(Entity, api=True):
    order_id: int
    order: Order | None = Field(persist=False, default=None)
    lines: list[OrderLine] = Field(persist=False, default_factory=list)
    reaction = get_order

consumers = [
    OnCreate(User, email_unique),
    OnChange(User, [email], email_unique),
    OnCreate(OrderLine, order_total),
    OnChange(OrderLine, [price, qty, order_id, deleted_at], order_total),
    OnChange(Order, [total], notify_total_changed, deferred=True),
]
```

```python
class create_order(Reaction):
    async def calculate(ctx):
        req = ctx.input
        if await ctx[User].get(req.user_id) is None:
            raise ValueError(f"unknown user {req.user_id}")
        order = ctx.new(Order, user_id=req.user_id, total=0)
        for l in req.lines:
            ctx.new(OrderLine, order_id=order.id, price=l["price"], qty=l["qty"])
        req.order = order

class order_total(Reaction):
    async def calculate(ctx):
        order_ids = {c.after["order_id"] for c in ctx.input}
        orders = await ctx[Order].many(order_ids)
        await ctx.load("select 'order_line', id, '*' from order_line where order_id = any(:ids)", ids=order_ids)
        lines = ctx[OrderLine].loaded()
        for order in orders:
            order.total = sum(l.price * l.qty for l in lines.filter(order_id=order.id))

class email_unique(Reaction):
    async def calculate(ctx):
        ids = [c.id for c in ctx.input]
        rows = await ctx.query("""
            select u.email from "user" u
            join "user" o on o.email = u.email and o.id <> u.id
            where u.id = any(:ids)
        """, ids=ids)
        if rows:
            raise ValueError(f"email {rows[0].email} already in use")

class notify_total_changed(Reaction):
    async def calculate(ctx):
        for c in ctx.input:
            order = await ctx[Order].get(c.id)
            send_email(order.user_id, before=c.before["total"], after=c.after["total"])

class get_order(Reaction):
    async def calculate(ctx):
        req = ctx.input
        req.order = await ctx[Order].get(req.order_id)
        await ctx.load("select 'order_line', id, '*' from order_line where order_id = :id", id=req.order_id)
        req.lines = ctx[OrderLine].loaded()
```

**Trace of `POST /create_order` with two lines.**

1. Seed: `CreateOrder` instance built, placed in the store, `trigger(create_order, req)`.
2. `create_order` loads the user, creates one `Order` and two `OrderLine` instances with pending Keys, sets `req.order`. Returns.
3. Reconcile level 1: frontier is three created instances. `OnCreate(OrderLine)` matches, `order_total` runs once with two `Change` records. `many` and `load` flush first, drawing three Keys and inserting three rows, then load. `order.total` set. Marks advance.
4. Reconcile level 2: frontier is `Order.total`. No eager entry lists it. Marks advance. Frontier was non-empty, so one more diff: empty. Done.
5. Flush: update the order's `total` and `updated_at`. Commit.
6. Response: `req` rendered with `order` inline, id resolved.
7. WAL worker sees the committed transaction: inserts on `order_t` and `order_line_t`, update on `order_t.total`. `OnChange(Order, [total])` is deferred and matches. One outbox row for `notify_total_changed` with one `Change`.
8. Deferred worker claims it, builds a context, runs the reaction, commits, deletes the row.

## 14. Out of scope for v1

- Permissions and authentication.
- REST resource mapping over API entities.
- Files, caches, search indexes, any data source other than PostgreSQL.
- Cross-request caching of loaded data.
- Persisted audit history of field changes.
- Static dependency graph, cycle detection before runtime, lineage documentation.
- Enforcement of undeclared reads or slow access patterns.
- Alternative outbox transports such as SQS or Kafka. The table is the contract; transports can be added behind it.

## 15. Decisions and open questions

Settled in review:

- **Ordering.** Explicit triggers start in trigger order; an awaited trigger runs immediately. Consumer config entries within a reconcile level run in no guaranteed order.
- **Timeouts replace depth caps.** Every reaction has a 60 second timeout by default, set per trigger. The seed trigger's timeout bounds the run.
- **Status codes.** Every exception is a 500. A REST layer above may do its own mapping.
- **Workers.** The deferred worker and the WAL worker are separate processes.
- **Packaging.** Starlette on uvicorn for ASGI, SQLAlchemy async with `asyncpg`, configuration through environment variables with a TOML file for defaults. Public surface is `backend_builder.{Entity, Field, Reaction, Context, OnCreate, OnChange, OnSchedule, run, workers}`.
- **Testing.** Against a local Postgres with `wal_level = logical`, so the WAL worker is exercised end to end.
- **`ctx.load` wildcard.** `'*'` as the prop column loads all persisted fields.
- **No `where`.** Reads are by id or through `ctx.load` with SQL. In-memory filtering over loaded instances is `Collection.filter`.
- **Snapshots are untracked blobs.** Reading an entity-typed field gives a plain object outside the store. Mutating it is the developer's business.
- **`ctx.query` stays read-only.** A value-writing variant is not needed for v1.

No open questions remain.
