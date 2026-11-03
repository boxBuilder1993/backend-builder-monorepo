# Scaling: compute targets, keyed stores and throughput

Status: proposal, 2026-11-03. Written from discussion; to argue with before any code. Extends
[design.md](design.md) and [pools.md](pools.md); where they disagree, this document is newer, and the one
decision of design.md it revises is named in §2. Sections 3 to 5 are the three scopes, compute, stores and
throughput; §6 is the order of work; §7 lists what was considered and set aside; §8 the open questions.

## 1. Intent

Three things will need to grow, and the box we run on today is the box we have:

- **Compute.** Where reactions execute: today processes on one host; later several hosts, a Kubernetes
  or ECS cluster, request-scaled containers, functions, and external job engines such as Spark.
- **Stores.** Where entities live: today one Postgres; later a second kind of entity, computed and keyed,
  that does not need a transaction and may live in DynamoDB.
- **Throughput.** How many source changes a second the system absorbs and how quickly derived data follows,
  on this box first.

None of these may cost a rewrite when it arrives. The design therefore builds the **seams** now and the
**engines** when they are needed, and every engine is a setting of the environment, never a change to a
reaction.

**The reference box.** 12 cores, 31 GB of memory of which the finance work uses most, a SATA SSD
(Samsung 850 EVO, no power-loss protection) holding Postgres 17 in Docker, and a spinning disk. Every
throughput figure in this document is an estimate for this box until the benchmark in §5.1 replaces it.

## 2. Principles

1. **Semantics in code, engines in the environment.** What a reaction means, its transaction mode, what it
   may write and how consistent its writes are, is declared on the reaction or the entity and reviewed like
   logic. Where it runs and which product stores its rows is declared on the `Environment` (pools.md §11),
   which may not change logic. This is pools.md's rule for placement, extended to stores.
2. **A seam is only real if both sides of it are exercised.** Every contract that will one day have a
   second engine has two engines from the day it is built: an in-memory one for tests and a Postgres one
   for this box. A contract with one engine quietly grows assumptions only that engine satisfies.
3. **Refuse, never substitute.** An engine that cannot honour a declaration refuses it when the App is
   built, naming the declaration: a reaction whose timeout exceeds what functions allow, a keyed reaction
   that writes a transactional entity. Nothing degrades silently.
4. **Measure before moving.** Every engine beyond this box has a stated condition for being built, in
   numbers the console or the benchmark produces.

**A revised decision.** design.md §23 says entities carry no storage configuration. That stands for
storage: no entity names a product, a table option or an index. It is narrowed by one thing an entity does
declare: its **consistency contract**, transactional or keyed (§4). The contract is logic, since it decides
what reactions may do with the entity; the product behind it is the environment's.

## 3. Compute

### 3.1 The model

A pool (pools.md §4) is a set of identical replicas that run work. A pool's declaration says shape:
replicas, concurrency, resources. This document adds where its replicas run, per environment:

```python
Environment(
    runs_on={"api": Host(), "default": Host(), "ml": Host()},                           # this box
)
Environment(
    runs_on={"api": Kubernetes(), "default": Kubernetes(), "ml": Kubernetes(gpu_nodes=True)},
)
Environment(
    runs_on={"api": Function(), "default": Ecs(fargate=True), "ml": Ecs(instance="g5.xlarge")},
    database="rds", connection_proxy="rds-proxy",
)
```

The default target for every pool is the environment's `default_target`, `Host()` for local
environments, so an environment that names nothing behaves as today. A pool has one target per
environment; two targets for one pool would be two pools.

What never changes with the target: the reaction's code, its transaction mode, the trigger modes and their
refusals (pools.md §2), profiles, records, the outbox and its exactly-once database effects, and the console.
What a target may refuse: work it cannot hold, listed in the matrix.

### 3.2 Targets

| | Host | Kubernetes | ECS (EC2 or Fargate) | Request-scaled containers (Cloud Run) | Functions (Lambda) | External jobs |
|---|---|---|---|---|---|---|
| How a replica starts | systemd unit, `worker --pool p` | Deployment | Service with a task definition | Service, scaled by requests | Invocation | Job submission |
| Pulls from pollers | yes | yes | yes | only with `min >= 1` and CPU always allocated | no: a dispatcher pushes (§3.4) | no: launched by a reaction (§3.5) |
| Long runs | yes | yes | yes | up to the platform's request limit | **no: 15 minutes** | yes, hours |
| Receives calls (pools.md §8) | yes, by host and port | yes, by Service | yes, by service discovery | yes, by URL | yes, synchronous invoke | no |
| Scale to zero | no | later (KEDA) | by desired count, not on demand | `api` only | yes | yes |
| GPU | yes | yes | EC2 only; Fargate has none | limited | no | engine's |
| Metrics | scraped | scraped | scraped by a collector sidecar | pushed (OpenTelemetry) | embedded-metric log lines | engine's, summarised into the run record |
| Database connections | the process pool | PgBouncer | PgBouncer or RDS Proxy | PgBouncer or proxy | **RDS Proxy or PgBouncer required** | none: the launching reaction holds them |

Two groups emerge. **Long-running workers**, host to request-scaled containers, run the same
`worker --pool p` command and pull work exactly as today; they differ only in how they are deployed. A new
target in this group is a renderer: given the App and the environment, produce the units, manifests or task
definitions and apply them. **Push-invoked compute**, functions and jobs, needs the runtime to hand it work,
and each has a section below.

### 3.3 Long-running targets

**Host, one box (now).** Every pool becomes its own unit, `bb@<app>:worker-<pool>`, so each pool has its
own systemd resource limits. A unit takes `MemoryMax` and `CPUQuota` from the pool's `memory` and `cpu`. On
one box this is the most useful part of pools today: a fitting pool cannot starve the API of memory or CPU.
`replicas` on a host is the number of units for the pool; `concurrency` is per unit.

**Hosts, several boxes.** `Host("box2")` names the machine. `backend-builder deploy <app> --env <name>`
renders each host's unit set and env file and applies them over SSH: pull, sync, restart that host's units.
Migrations run once, from the first host, before any unit restarts. No coordination between hosts is needed
for pollers, since `SKIP LOCKED` divides rows between any number of workers and the scheduler's unique index
makes a second scheduler harmless. Calls across hosts use the pool's address list from the environment; the
caller picks one replica in turn, and a refusal at `concurrency` is retried on the next within the caller's
budget (pools.md §8). Postgres stays on one host; §5.6 describes when it stops being enough.

**Kubernetes.** As designed in pools.md §11 to §13.

**ECS.** A second renderer of the same declarations: a task definition and service per pool, the API
behind a load balancer, the scheduler and admin as single-task services, the migrate task run first and
waited on. Calls use ECS Service Connect names. Secrets mount from Secrets Manager into the same file path
pools.md §10 reads. Scaling between `min` and `max` follows the scheduler's waiting-depth gauge, published
through the collector sidecar as a CloudWatch metric. Fargate and EC2 differ only in the capacity provider
and in GPUs.

**Request-scaled containers.** The `api` pool fits naturally: the platform scales it on requests and to
zero. A worker pool runs only with `min >= 1` and CPU allocated outside requests, because pollers work in
the background; nothing wakes a pool at zero, so `min = 0` on this target is refused for any pool that owns
a poller.

### 3.4 Functions

A function pool runs each group in its own invocation. Three things make it work.

**A dispatcher instead of a poll loop.** Functions cannot pull. A dispatcher, which runs inside the
scheduler process (already a singleton) or in any long-running pool, does for a function pool what a worker
does for its own: it claims rows for that pool's pollers with the ordinary lease (budget plus grace),
groups them, and invokes the function asynchronously with `{poller, key, items, lease}`. The function runs
the ordinary `run()` for the group and acknowledges the rows **inside the run's transaction**, exactly as a
worker does, so a deferred reaction's database effects still happen exactly once. The platform's own retry
of asynchronous invocations is switched off; a failed or lost invocation leaves rows whose lease lapses, and
the dispatcher claims them again. The dispatcher caps invocations in flight at the pool's
`max × concurrency`, and the function's reserved concurrency is set to the same number.

**Refusals at build time.** A reaction that can resolve to a function pool with a budget above the
platform limit minus a margin is refused, naming the reaction and its timeout. Pollers with
`transactional = False` (Kafka, SQS) are refused on function pools in the first cut, since the platform's
own event-source mappings would be the right path for them and are not designed here. The scheduler and the
admin console never run as functions.

**Connections and metrics.** Many short-lived instances each opening connections would exhaust Postgres,
so an environment with any function pool must name a connection proxy, and the store runs with
`BACKEND_BUILDER_DATABASE_POOLED=true` (pools.md §11). A function cannot be scraped, so at the end of each
invocation the runtime writes its metric deltas as one embedded-metric log line; the series and labels are
those of design.md §19. The image is the same image with the platform's runtime interface client as entry
point.

**HTTP on functions.** The `api` pool on a function target runs the same ASGI application behind an
adapter. Endpoints keep their budgets; a budget above the platform's request limit is refused.

### 3.5 External jobs

Some work is not a reaction at all but a program run by another engine: a Spark job over a lake, a
training job on a GPU queue, a container on a batch service. A run is one Postgres transaction and must not
be held for hours or spread over executors, so such work is **launched, watched and finalised**:

```python
class RollupTrades(Job):
    entry = "jobs/rollup.py"                 # the program; same code on every engine
    pool = "jobs"                            # where launch and finalise run

class FinaliseRollup(Reaction): ...          # verifies the output, writes the rows that reference it

triggers = [OnJobDone(RollupTrades, FinaliseRollup)]
```

- **Launch** is a reaction that creates a `JobRun` row (`queued`) and submits the job with a submission key
  and an output path derived from the `JobRun` id, so a retried launch finds the job it already submitted
  rather than starting a second.
- **Watch** is a poller whose source is the engine's job status: it polls `submitted` and `running` job
  runs, and a terminal status becomes a message to the finalising reaction.
- **Finalise** is an ordinary run. It checks the output exists at its path, writes the entities that
  reference it, and sets the `JobRun` to `done` or `failed`. Until it commits, nothing in the business can
  see the output. This is the lifecycle the finance `Model` already follows by hand (queued, fitting,
  uploading, fitted), made general.

The engine is the environment's: `JobEngine.local()` (a subprocess, DuckDB for SQL) for tests and this box,
Kubernetes Jobs, and later Spark (EMR Serverless, Databricks, Dataproc, Spark on Kubernetes) or a batch
queue. The console shows a job run with its engine's id, status, timings and log link beside the launch and
finalise runs.

**Condition for building beyond local:** a workload that does not fit the box, or a deployment whose
company already runs one of these engines.

## 4. Stores: keyed entities and DynamoDB

### 4.1 Two consistency contracts

Every entity today is **transactional**: it lives in Postgres, it is read and written inside a serializable
transaction, any SQL may join it, and its writes commit with the run or not at all. That stays the default
and the right choice for anything a business decision rests on.

A **keyed** entity is a value computed from other data and addressed by a key: a customer's last-30-day
totals, a product's current score, a day's revenue for a region. It needs no transaction, because it can
always be computed again; it needs to be fast to write, cheap to read by key, and never wrong for long. The
developer chooses, per entity, by its base class:

```python
class Order(Entity):                       # transactional
    customer_id: str
    amount: int

class CustomerStats(KeyedEntity):          # keyed
    key: ClassVar = ("customer_id",)
    customer_id: str
    orders_30d: int
    revenue_30d: int
```

The environment chooses the backend for keyed entities: `Environment(keyed=PostgresKeyed())`, the default,
or `Environment(keyed=DynamoDb(table_prefix="shop-"))`. Tests use the in-memory backend. All three honour
exactly the same contract, so moving an entity to DynamoDB is one setting and a backfill (§4.7).

### 4.2 The contract

- **Addressed by key.** `key` names one or more fields; an optional `sort` names fields ordered within a
  key's partition. The `id` is derived, `uuid5(entity, canonical key)`, so `get_by_id` still works and
  records still name rows by id.
- **Read by key only.** `await ctx.get_by_key(E, **key)`, `await ctx.get_many(E, keys)`, and, with a sort
  key, `await ctx.range(E, start=..., end=..., **partition)`. `ctx.query` on a keyed entity raises. No join
  across a keyed entity exists on any backend.
- **Written whole.** `ctx.put(E, **fields)` writes the whole value; a loaded keyed row is read-only, and
  assigning to it raises. There is no increment and no partial update, because a read-modify-write without
  a transaction loses updates. A keyed value is always recomputed, never adjusted.
- **Versioned.** Every put carries the version of the run that computed it (§4.3) and is applied only if
  that version is newer than the stored one. A put that loses is counted, not raised.
- **Deleted by tombstone.** `ctx.delete(row)` on a keyed row writes a tombstone with the run's version, so
  an older computation still in flight cannot bring the value back.
- **Converges.** The guarantee is precise: if every change to a keyed value's sources causes a recompute of
  that key, delivered after the change commits, then once the last such recompute has run, the stored value
  reflects every change. Between those moments the value may be stale, briefly and visibly. It is never
  silently stale for good, except when a recompute dead-letters, which the outbox shows.

### 4.3 Versions

A version must order computations by what they saw. The runtime uses the Postgres log position: a keyed
run's first statement opens its read snapshot and reads `pg_current_wal_lsn()` in the same statement; on a
replica, `pg_last_wal_replay_lsn()` (§5.5). Snapshots only ever see more as time passes and the log position
only advances, so a computation that started later saw at least as much and carries a larger version.

Two computations that start within microseconds of a concurrent commit can receive versions in the
opposite order to what they saw. That is harmless under the contract: the commit that separated them has
its own triggered recompute, which starts after both, sees the commit and carries a larger version than
either. The version therefore needs to be monotonic with time, not exact, and the log position is the
cheapest monotonic number the database already has. Keyed values computed from other keyed values use the
same number; an upstream value's version is always older than a downstream run that read it.

### 4.4 Keyed reactions and their run

The kind of run is the developer's statement, on the reaction:

```python
class RecomputeCustomerStats(Reaction):
    consistency = "keyed"

    async def calculate(self, ctx: Context) -> None:
        ids = [item["input"]["customer_id"] for item in ctx.input["items"]]
        rows = await ctx.query_raw(
            "select customer_id, count(*) n, sum(amount) total from \"order\" "
            "where customer_id = any(:ids) and created_at > now() - interval '30 days' group by 1",
            ids=ids,
        )
        for r in rows:
            ctx.put(CustomerStats, customer_id=r["customer_id"], orders_30d=r["n"], revenue_30d=r["total"])

triggers = [OnCreate(Order, customer_stats)]   # customer_stats: an OutboxPoller grouping by customer_id
```

A keyed run has four steps, and its retry is the whole sequence:

1. **Read.** A read-only `REPEATABLE READ` transaction: one snapshot, the version taken with it, no
   serialization failures, no commit cost. Transactional entities are readable; keyed entities are read
   from their backend.
2. **Compute.** The reaction's code.
3. **Put.** The session's puts and tombstones go to the keyed backend, each conditional on its version.
4. **Acknowledge.** A short transaction deletes the poller's rows, writes any deferred triggers and the
   keyed entities' trigger matches (§4.5), and commits with `synchronous_commit = off` (§5.3).

Failure anywhere before step 4 commits leaves the rows claimed; their lease lapses and the group runs again
with a newer version, overwriting whatever a failed attempt managed to put. A crash that loses the
acknowledgement does the same. So a keyed run needs no transaction across Postgres and the backend, and
has none.

**Rules, checked where they can be.** A keyed reaction may not create, change or delete a transactional
entity; `ctx.create` or a mutation of a transactional row raises at the call site and on flush. A
transactional reaction may not `put`. A keyed reaction may join only keyed reactions; it may `call` or defer
anything. A reaction that needs both kinds of write is two reactions, the transactional one deferring the
keyed one, which is the honest shape of what it does.

### 4.5 Triggers on keyed entities

`OnPut(CustomerStats, Reaction or poller)` fires on every put that was applied, whether or not the value
changed, and not on a put that lost to a newer version, since that newer run fires its own. Matches are
written as outbox rows in the acknowledgement transaction of step 4. This makes models of models work on
both backends through the outbox alone.

Why not only on change: a run that put a value and then lost its acknowledgement is retried, puts the
same value again, and would see no change; firing on change would drop the downstream recompute for good.
Firing on every applied put costs some recomputes that find nothing new, which grouped pollers and the
recompute-by-key shape make cheap.

### 4.6 Backends

**Postgres (default).** A keyed entity is a table in the `keyed` schema with its key as primary key and a
`version pg_lsn` column. A put is one statement for the session's whole batch:
`insert into keyed.<table> ... on conflict (<key>) do update set ... where <table>.version < excluded.version`, in its own
`READ COMMITTED` transaction with asynchronous commit. Keyed tables come from entities through
`makemigration` like any other; indexes beyond the key are not declared, since the contract has no reads
that would use them. On this box this backend is what makes derived data cheap: no serializable conflicts,
no synchronous commit, one statement per group.

**DynamoDB.** A table per keyed entity, partition key from `key`, sort key from `sort`, an attribute
`_v` for the version. A put is `PutItem` with the condition `attribute_not_exists(_v) OR _v < :v`; a failed
condition is a lost put, counted. Puts in one run are issued concurrently, since batch writes take no
condition. Reads use `ConsistentRead`, so a downstream run reads what an upstream run applied; ranges
are `Query` on the partition. Values map through the codec: numbers as numbers, decimals as decimals,
datetimes as ISO strings, nested values as maps; the 400 KB item limit is checked at put, naming the
entity. Tables are created by `backend-builder keyed-tables <app> --env <name>`, which prints the plan, and
with `--apply` creates what is missing; it never changes or deletes a table. Locally, DynamoDB Local runs as
a compose profile for anyone who wants to try the real API; the in-memory backend is what tests use.

**In memory.** A dict per entity with the same conditional put, the same tombstones and the same refusals.
Because it has no SQL, any reaction that tries to join a keyed table through `query_raw` fails in tests even
though it would work against the Postgres backend's tables. That is the seam of §2.2 doing its job: the
contract is tested against the backend with the fewest capabilities.

### 4.7 Changing a keyed model

Logic changes, so values must be rebuildable without downtime. A keyed reaction may declare
`all_keys = "select distinct customer_id from \"order\""`. `backend-builder rebuild <app> <Reaction>` enqueues
groups for every key that query returns, addressed to an implicit rebuild poller of the same reaction, so
live rows never queue behind a rebuild; its `instances` and pool are the environment's to turn up or down
while it runs. The rebuild and live traffic run side by side; versions make the newest computation win
wherever they meet. A change to an entity's key is a new entity, filled by a rebuild and switched to by the reactions that
read it, after which the old one is dropped.

A **reconciliation** reaction can be scheduled for any keyed model: it recomputes a sample of keys from
scratch, compares, and records the differences as rows the console shows. It is the check that tells a team
the incremental path agrees with the full computation, and it is how a pilot replaces a batch model.

### 4.8 Observability

Run records list a keyed run's puts with key, version and outcome. New metrics:
`bb_keyed_puts_total{entity, outcome}` with `applied`, `stale` and `error`, `bb_keyed_throttles_total{entity}`,
and `bb_keyed_read_seconds{entity}`. A high `stale` share is concurrent recomputes of one key; throttles and
`bb_run_attempts` concentrated on one reaction are a hot key. Hot keys are a data modelling decision and the
developer's: a coarser or finer grain, grouping, or a sharded value summed when read. The runtime's part is
to make them visible, and the guide says what to do about them. The console's graph shows each entity's
contract and backend, so which data is transactional and which is eventual is visible at a glance.

## 5. Throughput

### 5.1 Measure first

A benchmark app, `apps/bench`, drives the system on this box and produces the numbers every later decision
uses. A generator inserts `Event(customer_id, amount)` rows at a rising rate through the API and directly;
an `OnCreate` trigger recomputes a per-customer value, once as a transactional reaction and once as a keyed
one. It reports, per configuration and rate: source commits per second; end-to-end lag from an event's commit
to the value that includes it, p50 and p99; runs per event; serialization retries; Postgres CPU, WAL bytes and
fsyncs per second; outbox depth over time. The matrix: run records `full`, `summary`, `off`; synchronous and
asynchronous commit; transactional and keyed; key cardinality from one hot key to a million; worker units and
concurrency. `make bench` runs it and writes the result table into `docs/bench/` with the commit it ran on.

### 5.2 Where the commits go today

Every synchronous commit waits for the WAL to reach the SSD. On this disk that wait, not CPU, is the first
ceiling; concurrent commits share one flush (group commit), so the ceiling rises with concurrency until the
disk saturates. Counting commits per source event with one data-change trigger:

| Step | Synchronous commits today | Notes |
|---|---|---|
| Source run (endpoint) | 3 | run record `running`, the run, run record final |
| Claim | 1 per batch | amortised across `OUTBOX_BATCH` rows |
| Deferred delivery | 3 | run record `running`, the run with its row's delete, run record final |
| **Total** | **about 6** | records are 4 of the 6 |

Run records are two-thirds of the commits. That is the first thing to change.

### 5.3 Levers on one box, cheapest first

1. **Record levels per pool and per reaction.** `BACKEND_BUILDER_RUN_RECORDS` today is global. It becomes
   a default overridable on a pool or a reaction: `full`, `summary`, `errors` (only failed runs, and no
   `running` row), `sample(0.01)`, `off`. Without a `running` row a crashed run is not marked `lost`; its
   work is still recovered by the outbox lease, which is what matters for a high-volume pool.
2. **Asynchronous commit where loss is recoverable.** `SET LOCAL synchronous_commit = off` per transaction
   for run record writes, for keyed acknowledgements (§4.4) and for any transactional reaction declaring
   `durability = "relaxed"`. A crash loses at most the last few hundred milliseconds of such commits and
   never corrupts. A lost relaxed commit takes its outbox delete with it, so the row is delivered again;
   and because the WAL is sequential, any later synchronous commit makes every earlier relaxed commit
   durable too, so nothing durable can depend on a relaxed commit that was lost. Business transactions stay
   synchronous by default.
3. **Batched record writes.** Finished records are buffered per process and written every 200 ms in one
   transaction, asynchronously. Records are already best-effort (design.md §18); a crash loses a buffer of
   them, never a run.
4. **Grouped pollers** (pools.md §6). Fifty events for one customer become one recompute. With records off
   or batched, the commits per event fall towards one: the source run's own.
5. **Table hygiene.** The outbox is all inserts, claim updates and deletes: per-table autovacuum with no scale
   factor and a fixed threshold, no cost delay, and a fill factor that leaves room for claim updates to stay
   on their page. `_run` becomes partitioned by day, and the hourly purge drops whole partitions instead of
   deleting rows, which removes the purge's bloat and its vacuum debt.
6. **Postgres configured for the box.** Starting values in `compose.yaml`, each to be confirmed by the
   benchmark: `shared_buffers` 2 GB (the finance work needs the rest), `effective_cache_size` 8 GB,
   `max_wal_size` 8 GB and `checkpoint_timeout` 15 minutes so checkpoints are rare and spread,
   `wal_compression = lz4`, `random_page_cost = 1.1` and `effective_io_concurrency = 200` for the SSD.
   `fsync` and `full_page_writes` are never turned off: the drive has no power-loss protection, and they are
   what keeps a power cut from corrupting the database.
7. **Connections.** Each process holds one engine with SQLAlchemy's default pool of 5 plus 10 overflow. The
   pool size follows the process's concurrency, and the sum over every unit on the box stays under
   `max_connections`, checked by `make up`. PgBouncer is not needed on one box.

**Rough expectation, to be replaced by §5.1.** Today's six synchronous commits per event put the box in the
hundreds to low thousands of events a second. With levers 1 to 4, a keyed model costs about one synchronous
commit per event, and the same disk should carry several thousand to tens of thousands of events a second,
with lag in the low seconds. Postgres reading the keyed model's sources is cheap by comparison when the
recompute query is indexed on its key; Postgres indexes (B-tree on key and time, covering, partial,
expression) are what make recompute-by-key fast, and each index costs writes, so a model's indexes match its
recompute query and nothing more.

### 5.4 Streaming consumers

The outbox is the stream on this box. Two other sources fit the `Poller` interface of pools.md §6 and have
stated conditions:

- **Logical decoding of the WAL** (`WalPoller`). Data-change triggers read from the log Postgres writes
  anyway, so a source write carries no outbox row. Costs that kept the earlier WAL worker parked (design.md
  §15) still apply and are the conditions: `OnChange` needs old values, so its tables need
  `REPLICA IDENTITY FULL`, which enlarges the WAL; a replication slot holds WAL while its reader is down, so
  `max_slot_wal_keep_size` must bound it; one slot is one reader, so the reader dispatches groups to worker
  loops and acknowledges the log position only below the oldest group still running. Deferred triggers and
  schedules stay on the outbox. **Built when** the benchmark shows outbox writes are the limit after the
  levers of §5.3.
- **Kafka, Redpanda, NATS, SQS** (pools.md §6, `KafkaPoller`, `SqsPoller`). Another service using memory
  and disk on the box, so **built when** a system outside the app produces or consumes the events.

### 5.5 Read replicas

Postgres does not route queries; the client chooses. The environment may name a `read_url`. Keyed runs,
which read at `REPEATABLE READ` and write nothing transactional, may read there, with the version taken from
`pg_last_wal_replay_lsn()`; the contract of §4.2 holds unchanged, and replica lag adds to staleness only.
Transactional runs always use the primary: a hot standby cannot run serializable transactions, and a run
that just wrote must read its own writes. Endpoints use the primary. Long reads on a replica can be cancelled
by replay; the replica's `max_standby_streaming_delay` is set for the longest keyed read, and a cancelled read
is a retried run. **Built when** there is a second machine: a replica on the same box shares its CPU and
disk and adds a copy of every write.

### 5.6 Beyond one box

In order, each step only when the previous one is measured full: a larger box or a faster disk (an NVMe
drive with power-loss protection raises the commit ceiling more than any setting); keyed entities moved to
DynamoDB, which takes their writes off Postgres entirely; a replica for keyed reads; apps split into their
own databases, which they already are by convention; then sharding sources by tenant or domain, or a
Postgres-compatible distributed database (Citus, YugabyteDB, CockroachDB, Aurora Limitless) for an app whose
transactional writes outgrow one primary.

## 6. Order of work

pools.md §14 stays the order for pools, pollers, calls, profiles and the cluster. This document's work
interleaves with it; each step is one PR and leaves the host setup working unchanged.

1. **Benchmark.** `apps/bench`, the generator, the report, `make bench`, and today's numbers committed. No
   runtime change.
2. **Throughput levers.** Record levels per pool and reaction, batched and asynchronous record writes,
   `durability = "relaxed"`, outbox table settings, `_run` partitioned by day with partition drops, the
   compose Postgres settings, the connection check in `make up`. The benchmark rerun and committed.
3. **Pools on the host** (pools.md step 1 already includes the runtime): a unit per pool with its systemd
   limits from the pool's resources.
4. **Keyed entities.** `KeyedEntity`, `consistency = "keyed"`, the keyed run, `get_by_key`, `get_many`,
   `range`, `put`, tombstones, versions, `OnPut`, the in-memory and Postgres backends, the rules and their
   errors, records and metrics, the console's contract column, `rebuild`, the reconciliation reaction. The
   benchmark's keyed variant.
5. **DynamoDB backend.** The backend, `keyed-tables`, the DynamoDB Local compose profile, the contract suite
   run against all three backends.
6. **Compute renderers.** Hosts over SSH; Kubernetes as pools.md step 4; ECS; request-scaled containers;
   functions with the dispatcher. Each when an environment needs it.
7. **Jobs.** `Job`, `JobRun`, launch, the status poller, finalise, `OnJobDone`, the local engine; the finance
   fits as the first user. Other engines when a workload or a deployment needs them.
8. **Later, by condition.** `WalPoller` (§5.4), read replicas (§5.5), Kafka and SQS pollers.

## 7. Considered and set aside

- **Keyed writes after the Postgres commit.** Discussed first. A failed write after commit would need the
  whole run retried, repeating its transactional writes. Putting before the acknowledgement, and forbidding
  transactional writes in keyed runs, makes the retry harmless.
- **A transaction across Postgres and DynamoDB.** None exists; the keyed run's order of steps and versions
  make one unnecessary.
- **Increments and partial updates on keyed entities.** Lose updates without a transaction. Values are
  recomputed whole.
- **Firing keyed triggers only on change.** Drops the downstream recompute when an acknowledgement is lost
  after a put (§4.5).
- **DynamoDB Streams as the trigger source for keyed entities.** Works only for one backend; the outbox
  serves both. Comes back if outbox writes for keyed triggers are measured as a limit.
- **A store named on the entity.** The product is the environment's; the entity declares only its contract.
- **SQL over keyed tables on the Postgres backend.** Would make the move to DynamoDB a rewrite; tests use the
  in-memory backend so such SQL fails early.
- **Per-query routing to replicas.** Routing follows the run's contract, not individual statements.
- **A replica, Kafka or a distributed database on this box.** Each adds load to the one machine they would
  relieve.
- **Disabling `fsync` or `full_page_writes`.** Corruption on a power cut. Asynchronous commit gets most of the
  speed with none of that risk.
- **Hot-key mitigation in the runtime.** Sharded counters and the like are data modelling; the runtime shows
  where the heat is.
- **Functions for the scheduler or the console, and scale-to-zero worker pools on request-scaled
  containers.** Neither has anything to wake it.
- **A fourth kind of pool for Spark.** Spark runs over executors and files, not inside a transaction; it is a
  job (§3.5).

## 8. Open questions

1. Should `durability = "relaxed"` exist for transactional reactions at all, or only for records and keyed
   acknowledgements? It is the largest lever for transactional derived data and the easiest to misuse.
2. Secondary indexes on keyed entities (DynamoDB global secondary indexes, eventually consistent even under
   `ConsistentRead`). Left out of the first contract; needed as soon as a model is read by something other
   than its key.
3. The dispatcher's home for functions in an environment with no long-running process at all: a scheduled
   function ticking every minute would cost the scheduler's 15-second granularity.
4. Whether `rebuild` should rate-limit itself against live lag, or rely on its separate poller alone.
5. Retention for keyed tombstones: they must outlive any computation that could still be in flight, which is
   the longest keyed budget plus the lease grace; a scheduled purge after that is safe.
