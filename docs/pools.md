# Pools, Pollers and Profiles

Status: design, 2026-10-31. Decided in discussion, not built. Companion to [design.md](design.md), which it extends; where the two disagree, this document is newer. Sections 1 to 13 are the design; section 14 is the order of work; section 15 lists what was considered and set aside.

## Intent

Run a company's backend as one repository that also describes, completely, how it is deployed, scaled and secured. Three ideas carry that:

- **Pools** are where work runs. Any reaction's work can be placed in any pool by several means, down to one pool per reaction, so each kind of work scales on its own. Placement is a scaling decision and never changes what a reaction's writes mean.
- **Pollers** are how work enters. Everything the system consumes, outbox rows from data changes, schedules and deferrals, and later queues such as Kafka or SQS, arrives through a `Poller`: poll, group, run one reaction per group, acknowledge. Fifty changes to one order become one run, not fifty.
- **Profiles** are who work runs as. A reaction has one profile, an execution identity with a positive allowlist of entities, secrets and other profiles it may invoke, enforced by Postgres roles and by the runtime. Joined children run as their parent; calls and deferrals run as themselves.

Everything is declared in code on the `App`, environments included. Kubernetes objects are rendered from the declaration and applied; nothing is configured in Kubernetes terms by hand. Secrets are the one thing whose values live outside the repo, and the console manages them.

The transactional relationship between a reaction and a child it triggers stays an explicit choice: join the parent's transaction, call as another service would be called, or defer until after commit. Placement and profiles never alter that choice; they only make some choices unavailable, and say so at the trigger.

## 1. The transaction decides what is possible

Everything here follows from one fact of Postgres: a transaction belongs to one connection. A second connection cannot join it, see its uncommitted writes, or be rolled back with it. Savepoints are scoped to one transaction; snapshot export shares a read view, not a transaction; two-phase commit makes two transactions agree on an outcome without making them one.

Therefore a child that shares its parent's transaction must run in the parent's process, and a child in another pool is another transaction, by construction.

## 2. Three trigger modes

```python
await ctx.trigger(Child, input)                          # join: same transaction, session, process and profile
await ctx.trigger(Child, input, mode="call")             # call: own run, own transaction, own profile; returns a result
await ctx.trigger(Child, input, mode="call", key="...")  # call with an explicit idempotency key
await ctx.trigger(Child, input, deferred=True)           # defer: outbox row at flush; runs after commit, as itself
```

`mode` is `"join"` (default), `"call"` or `"defer"`; `deferred=True` remains as the spelling of `mode="defer"`. The mode is the developer's statement of what the child's writes mean to the parent's.

**join.** The child runs in the parent's session, transaction, process and profile, depth-first, and returns its context. It sees the parent's uncommitted entities and mutates them in place; its writes commit with the run or not at all. A joined child that raises aborts the run unless the parent catches the exception, in which case the child's writes stay in the session; no savepoint is added (decided). Join requires the child to resolve to the parent's pool (section 5); a join whose child resolves elsewhere raises `TriggerError` before anything runs: *join is not supported across pools; use mode="call" or deferred=True*. Nothing is converted silently. Join needs no permission, because the child can reach nothing the parent could not (section 9).

**call.** The service call. The child runs now as a run of its own, through `run()`, in its own serializable transaction, under its own profile, in whichever pool it resolves to, the parent's included, and commits. It sees committed state only, never the parent's uncommitted writes, so everything it needs arrives in its input. The parent's transaction is untouched; the parent blocks until the child returns, within the parent's remaining budget. The parent receives a `CallResult`: the child's saved vars as JSON, its `reaction_id`, and `replayed` (section 3). If the child raises, the parent gets the exception. If the parent later fails, **the child's commit stands**; undoing it is a compensating reaction the developer triggers from the failure path, which is a saga, and the guide says so. A call behaves identically whether the child's pool shares the parent's process or sits across the cluster: input and result always cross as JSON, and a non-JSON value in either, an entity object for one, fails at the call site. The caller's profile must be allowed to invoke the child's (section 9).

**defer.** An outbox row written at flush, addressed to the child's poller (section 6), consumed after the parent commits, run under the child's profile in its pool, retried and dead-lettered by the poller's rules. The parent gets no result; a result travels the event-driven way, through an entity the child writes and a trigger on it. The caller's profile must be allowed to invoke the child's.

Across every mode: input is the only thing a parent gives a child; vars belong to one invocation; a child's saved vars reach the parent only through the returned context or the `CallResult`. These rules from [design.md](design.md) §4 are unchanged.

## 3. Call idempotency: the ledger

A run is retried from the seed on any exception, so a parent that calls a child and then fails would call it again. Service calls need an idempotency key; the runtime derives one so the common case needs no thought and the uncommon case has an escape hatch.

**The key.** `uuid5(parent_reaction_id, reaction + canonical_json(input) + n)`, where the parent's `reaction_id` is stable across its attempts, `canonical_json` is the input with sorted keys, and `n` counts earlier calls in this attempt with the same reaction and input (almost always 0). The same call with the same input on a retry gets the same id. Adding, removing or reordering *other* calls between deploys does not change it; an ordinal-only key would have replayed the wrong child after a deploy that reordered calls while a retry was pending. When the input carries something that varies between attempts, the developer passes `key="..."` and owns stability.

**The record is the ledger.** For a call, the run record is written **inside the child's own transaction**, so "the child committed" and "the ledger says the child committed" are one fact. The record carries `source="call"`, the caller's `root_id` and `parent_reaction_id`, the derived id, the profile, and the outcome.

**Replay.** Before running a call, the runtime looks up the id in `_run`. A committed record means the child already ran: its saved vars are returned with `replayed = true` and nothing executes. A failed record, or none, means run it. A `running` record means an earlier attempt's child is still executing or died mid-way; the caller waits up to its budget for it to resolve and treats a stale one as failed (the scheduler's lost sweep applies).

Consequences: a child's commit that causes the parent's serialization failure does not run the child twice on retry; a retried parent sees the result the first attempt saw; a call's effects happen exactly once per committed call, or not at all. Not bought: undoing the child if the parent gives up for good.

## 4. Pools

```python
pools = [
    Pool("api", min=2, max=10, cpu="500m", memory="512Mi"),
    Pool("default", min=1, max=8, concurrency=16),
    Pool("pricing", min=1, max=20, concurrency=8),
    Pool("ml", min=0, max=4, concurrency=1, cpu="4", memory="8Gi", gpu=1),
]
```

A pool is a set of identical replicas, usually spread across machines, running `backend-builder worker <app> --pool <name>`. A `Pool` declares shape only: `name`, `min` and `max` replicas, `concurrency` (runs in flight per replica, shared by every poller and call the replica serves), and `cpu`, `memory`, `gpu` as defaults an environment may override. It lists no reactions; reactions, triggers, endpoints and pollers name their pool (section 5). `default` and `api` exist even if undeclared. `api` runs `serve`: endpoint seeds and whatever they join, and claims nothing. `default` is the floor for everything unplaced.

A pool's capacity is `replicas × concurrency`. Rules checked when the App is built: every pool named anywhere is declared; `concurrency >= 1`, `0 <= min <= max`, `max >= 1`. Guidance, not rules: a pool that receives calls should keep `min >= 1`, since a cold start inside a caller's held transaction is latency the caller pays; the console shows cold starts so the choice is informed. A pool that only consumes pollers can sit at `min = 0`. A GPU pool is typically `concurrency = 1`.

What a pool does not declare: image, tag, database endpoint, secret values, namespace, node selectors, tolerations, environment-specific replica counts. Those belong to the environment (section 11). A pool's database login role and its mounted secrets are derived from the profiles that can run there (section 9).

`backend-builder graph` prints each pool with the reactions and pollers that can resolve to it; the console shows them (section 12); `describe()` includes them.

## 5. Placement

A pool is where work executes. Placement is resolved before a run or an invocation starts, from these sources, lowest precedence first:

1. **Default.** The `default` pool.
2. **Inherited.** A child runs in its parent's pool. A seed runs where the thing that started it says: an endpoint's seed in `api` unless the endpoint names a pool; a poller's runs in the poller's pool.
3. **Trigger-site.** `Endpoint(..., pool="reports")`; `OnSchedule("0 3 * * *", Rollup, pool="nightly")`; `Poller(..., pool="sinks")`. Where a seed runs is a property of what starts it; a heavy synchronous endpoint does not belong in `api`.
4. **Class pin.** `pool = "ml"` on the reaction class. The reaction's runs go there whoever triggers them, for a resource the reaction needs: a GPU, memory, a licence. A pin is a constraint: a call site or poller naming a different pool for a pinned reaction is a `TriggerError`.
5. **Call-site.** `ctx.trigger(X, input, mode="call", pool="batch")` or `deferred=True, pool=...`. The same reaction can be cheap in one context and heavy in another, and the caller may know what the class cannot. Not valid with `join`, which has no choice of pool.
6. **By input.** Wherever a pool name is accepted, a function of the input is accepted too: `pool=lambda input: "gpu" if input["rows"] > 1_000_000 else "cpu"`, or one pool per tenant. Resolved at the trigger.

The resolved pool and the source that chose it are recorded on every invocation's row.

Patterns left out: overflow from a saturated pool to another, which hides saturation instead of showing it, the poller's waiting count being the honest signal; and placement by data locality, since nothing in the runtime is node-local.

## 6. Pollers

Everything the system consumes arrives through a poller. A poller is the loop that takes messages from a source, groups them, runs one reaction per group with the whole group as input, and acknowledges them. The runtime owns the loop; a `Poller` subclass owns the source.

```python
class Poller:
    name: str
    reaction: type[Reaction]
    pool: str = "default"
    every: float = 5.0            # idle interval; a full poll re-polls at once
    timeout: float | None = None  # budget per group run; default the app's
    max_rows: int = 500           # messages per poll
    max_groups: int = 20          # runs per poll
    instances: int = 1            # poll loops per replica
    transactional: bool = False   # ack can run inside the group's transaction

    async def poll(self, limit: int) -> list[Message]: ...
    async def ack(self, ids: Sequence[str]) -> None: ...
    async def release(self, ids: Sequence[str], error: str) -> None: ...
    async def execute(self, batch: list[Message], run: Runner) -> None:
        for key, items in group_by_key(batch, self.max_groups):
            await run(self.reaction, key, items)
```

`Message` is `id`, `input`, `key` (grouping key or None), `received_at`, `attempts`. `poll`, `ack` and `release` are the source's and fully the subclass's. `execute` is overridable, for a poller that wants one run per item or one run for everything, but it may only start reactions through `run`, never by calling `calculate` itself, so records, metrics, acknowledgement and failure handling are identical for every poller anyone writes.

**What `run` does, for every poller:** the normal `run()` path with the app's retries and the poller's timeout, under the reaction's profile, in the poller's pool; a run record with `source = "poller:<name>"`; metrics; the ack of the group's messages inside the run's transaction when `transactional` is true and after commit otherwise; `release` with the error on failure. On repeated failure of a group, `run` bisects it: halves are retried separately until a poison message is alone, which then goes to the dead-letter path by itself while the rest commit.

**The reaction's input** has one shape for every source:

```python
{"poller": "order_totals", "key": "<group key or null>",
 "items": [{"id": ..., "input": {...}, "received_at": ..., "attempts": n}, ...]}
```

There is no translate step: shaping items into whatever the reaction needs is the reaction's job, as it is for API reactions.

**Groups are notifications, not state.** A grouped reaction treats its items as "these keys changed", loads the current state for each key, and recomputes. It never applies items incrementally. Under that contract a key whose rows are split across two concurrent polls is harmless: both runs load the same current state and write the same result, and serializable isolation makes one of them a cheap retry if they overlap. No key-level locking exists anywhere; rows are exclusive always, keys never need to be.

**`OutboxPoller`** is the first subclass and the one the core ships:

```python
order_totals = OutboxPoller(
    name="order_totals",
    reaction=RecomputeOrders,
    query="select id, input->>'order_id' as group_key from rows order by created_at",
    pool="default", every=2.0, max_rows=500, max_groups=50,
)
```

`rows` is a runtime-provided relation: this poller's due, unclaimed outbox rows, with `id`, `reaction`, `input` (jsonb), `created_at`, `available_at`, `attempts`. The query returns `id` and optionally `group_key`; the runtime wraps it with the lease claim, `for update skip locked`, and `limit`. Default query: `select id from rows order by created_at`, no grouping. `poll` claims and leases, `ack` deletes, `release` clears the claim and bumps `attempts`, `transactional = True`. Rows are **addressed, never selected across pollers**: a row carries `poller = <name>` and nothing else ever claims it, so no row can match two consumers and the query only filters within the poller's own rows.

**Implicit pollers make today's delivery an instance of this.** Every reaction has an implicit `OutboxPoller` with the query `select id as group_key, id from rows`, one row per group, `max_rows` from `OUTBOX_BATCH` and `every` from `OUTBOX_POLL_SECONDS`, in the reaction's pool. `OnCreate(E, Reaction)` addresses rows to it; so do `ctx.trigger(..., deferred=True)` and the scheduler's occurrences. `OnCreate(E, poller)` addresses rows to an explicit poller instead. The worker is therefore a poller runner and nothing else: for each poller whose pool it serves, `instances` loops of poll, execute, ack.

**Replicas and instances.** No two polls ever receive the same row: `SKIP LOCKED` partitions rows between concurrent polls on any number of replicas and loops, and a replica that dies leaves rows whose leases lapse and are reclaimed. `instances` adds poll loops per replica for sources with slow or small polls; `concurrency` caps the group runs a replica has in flight across everything it serves. A pool's poll loops are `replicas × instances`; its runs in flight `replicas × concurrency`.

**Other sources** are subclasses in their own packages: `KafkaPoller(topic, group)` with the message key as `key`, offsets committed in `ack` after commit, `transactional = False`; `SqsPoller(queue_url)` likewise with delete-message as `ack`. A non-transactional source is at-least-once, so its reaction must be idempotent; the guide says so, and the source's own dead-letter (a topic, a DLQ) receives what bisection isolates. Triggers do not address rows to these pollers; their messages come from outside. Everything else, pool, instances, overrides, console, is identical.

**Declaration.** `App(pollers=[order_totals, warehouse, ...])`. The App collects the explicit pollers plus the implicit ones, checks names are unique, each reaction is the app's, each pool is declared, and each `OutboxPoller` query prepares against the outbox schema and returns `id`; it then resolves addressing so every outbox row names exactly one poller. The console shows explicit and implicit pollers alike, the implicit ones under their reaction's name.

## 7. Scheduler

Unchanged, one replica, restarted by its Deployment if it dies; nothing is lost while it is down, since seeding and the lost sweep catch up on the next tick. It seeds schedule occurrences as outbox rows addressed to the scheduled reaction's poller, and it exports waiting depth per poller and per pool, since a pool at zero replicas has no worker to export anything.

## 8. Calls across processes

A call whose child resolves to a pool served by another process is a request to that pool. Each pool's worker serves `POST /calls` on its metrics port, body `{"run_id", "reaction", "input", "root_id", "parent_reaction_id", "profile", "deadline"}`, response the child's vars and outcome. The worker checks the caller's profile may invoke the child's, consults the ledger, executes through the same `run()` as a poller delivery under the child's profile, counting it against `concurrency` and refusing when full, which the caller treats as unreachable and retries within its budget. The caller connects to the pool's Kubernetes Service and passes its remaining budget as the deadline; the callee refuses work whose deadline has passed. A non-2xx or a dropped connection is an exception in the caller, so the run fails or retries like any other error, and the retry consults the ledger. Pods trust one another inside the app's namespace; a NetworkPolicy limits `/calls` to the app's pods, and the caller's profile is attested by the calling pod's identity, which can only carry profiles its pool is a member of (section 9).

In-process calls, when caller and callee share a process, skip the network and execute `run()` on a second connection with JSON in and out. Semantics are identical; compose and tests exercise exactly the same paths without a cluster. Joined children never cross the network.

## 9. Profiles

A profile is an execution identity: what a reaction may read and write, which secrets it may read, and which other profiles it may invoke. Positive allowlists, nothing marked on entities.

```python
profiles = [
    Profile("api",      entities=[],                 secrets=[],                 invokes=["orders", "payments"]),
    Profile("orders",   entities=[Order, OrderLine], secrets=[],                 invokes=["payments", "pricing"]),
    Profile("payments", entities=[Card, Payment],    secrets=["stripe_api_key"], invokes=[]),
    Profile("operator", entities="*", secrets="*", invokes="*"),
]

class ChargeCard(Reaction):
    profile = "payments"
```

A reaction has exactly one profile, declared on the class; unlabelled reactions are `default`. Before an app declares any profiles, `default` is unrestricted, so today's apps run unchanged; declaring profiles and labelling reactions is how an app tightens, and the App then refuses, at build time where it can see it and at the trigger where it cannot, any entity access or trigger edge the declarations do not cover.

**Where a profile is established.** Only at the root of a run: by the endpoint's reaction, by the poller's reaction, by a call's callee, by the operator in the console or CLI. A joined child runs as its parent, always; the child's own profile is irrelevant in a join. Nothing is gained by joining: the child can only touch what the parent already could, and if it tries more, Postgres refuses under the parent's role and the run fails. A reaction written to run as `payments` that would misbehave when joined under `orders` is the developer's to know; the failure is loud and the permissions hold. An optional `joinable = False` on a reaction makes any join to it fail at the trigger with *run this as a call*, for developers who want the intent explicit.

**Permissions on triggers.** `call` and `defer` start a new run under the child's profile; the caller's profile needs the child's profile in its `invokes`, checked at the trigger before anything runs. The callee runs as itself whoever called it; the call site cannot pick a profile for it. What crosses back is what the callee chose to save, as JSON; the callee is responsible for not returning what the caller should not see, as an API is responsible for its response. Join needs no permission.

**Enforcement.** Each profile is a Postgres role with grants on exactly its entities' tables, generated as a migration from the declaration the way indexes and constraints already are. Each pool's login role is granted membership in the profiles of the reactions that can resolve to it, derived from placement. A run begins with `SET LOCAL ROLE <profile>`, once, transaction-scoped and so safe under PgBouncer's transaction pooling; a joined child changes nothing. In memory and under compose the session checks the run's profile on every store call and raises `AccessError` naming the entity, so tests catch violations before any cluster does. `ctx.secret` checks the profile's secret allowlist, and the mounted Secret for a pool contains only the union of its profiles' keys, so a bypass of the runtime check finds nothing to read. For cloud resources a profile may name an IAM role the pod assumes for that profile's runs; same identity, one more backend.

**HTTP and the console.** HTTP is not a profile; authentication is a reaction, as decided. An endpoint's seed runs under its reaction's profile, usually a small `api` profile that touches little and invokes the domain profiles. The console and the CLI run as `operator`, which can invoke everything, so a human can trigger anything and the records say it was the operator. Every invocation's run row records its profile, so "who read this" is a query.

`invokes` lives on the caller, IAM style, because looking at a profile and asking "what can this do" is the common question. A callee-side restriction, "only these may reach me whatever they claim", is a possible later addition and is not needed for the permissions to hold.

## 10. Secrets

**Declared in code.** `App(secrets=["notify_api_key", "stripe_api_key"])` names what the app needs; `.env.example` is generated from it. Startup refuses a missing name; `make run-all` checks `.env` against it; profiles allowlist from it.

**Values live outside the repo and reach pods as files.** `ctx.secret(name)` reads `/run/secrets/backend-builder/SECRET_<NAME>` if the file exists, else the environment variable, on access, so a rotated value is picked up without a restart. Locally the repo's gitignored `.env` becomes the app's Kubernetes Secret via dry-run and apply, never written to YAML or git. Elsewhere the environment names a vault, and an external secrets operator fills the same mount. Compose, tests and host setups keep using variables and notice nothing. Redaction is unchanged.

**The console manages them.** A secrets page lists each declared name with set, when and by whom, never a value. Set or rotate writes through the environment's provider: locally the app's Kubernetes Secret, the one object the console may write, with RBAC scoped to it alone; elsewhere the vault, and Kubernetes is not touched. Each write adds a `_secret_change` row (name, time, actor, never the value) shown beside the run records so a rotation can be lined up against what it broke or fixed. The admin token gates it, a confirm step guards it, and a setting disables the writer for a read-only console.

## 11. Environments and deploy: nothing configured in Kubernetes

The App is the only place anything is configured, environments included:

```python
app = App(
    package="shop",
    endpoints=endpoints, triggers=triggers, pollers=pollers, pools=pools,
    profiles=profiles, secrets=["notify_api_key", "stripe_api_key"],
    environments={
        "local": Environment(cluster="minikube", replicas={"ml": 1}),
        "prod": Environment(
            cluster="eks-prod", domain="shop.example.com",
            replicas={"api": 3, "default": 2, "ml": 0},
            nodes={"ml": NodeGroup(selector={"gpu": "true"}, tolerations=["gpu"])},
            images={"ml": "shop-gpu"},
            pollers={"warehouse": PollerOverride(every=10.0, instances=4)},
            database="rds", secrets="aws-secrets-manager",
        ),
    },
)
```

Typed objects, validated when the App is built, reviewed in the same PR as the reactions they run. An environment may override, per pool, replicas, resources, node placement and image; per poller, cadence, sizing and instances; and name the database and secret sources and the api's domain. It may not change queries, reactions, profiles or modes: those are logic.

**Rendering and applying.** `backend-builder manifests <app> --env <name>` renders Kubernetes YAML from the App; `backend-builder deploy <app> --env <name>` renders and applies it with `kubectl apply --prune -l app=<name> -n <namespace>`. Rendered, in a namespace derived from the app name:

- one Deployment per pool running `worker <app> --pool <name>`, with the pool's resources, `nvidia.com/gpu` when `gpu` is set, `replicas` from the environment or `min`, the pool's database login role and its scoped secret mount, and a Service;
- the `api` Deployment running `serve`, its Service, an Ingress with the environment's domain and TLS, and an HPA on CPU between `min` and `max`;
- one scheduler Deployment; one admin Deployment with its Service and an Ingress behind the admin token;
- a migrate Job, applied first and waited on, which also applies the generated role and grant migrations;
- a PodDisruptionBudget for every pool with `min >= 2`; a NetworkPolicy for `/calls`;
- PgBouncer in transaction pooling mode, and Prometheus scraping every process's metrics port for Grafana;
- Alloy as a DaemonSet and Loki with a volume, from [logs.md](logs.md) §2, with the one-day retention.

The YAML is a build artifact, not configuration: CI renders it and attaches the diff to the PR so a change to `replicas["ml"]` shows the Deployment it changes, but nobody edits it and it is not committed. The runtime never talks to the Kubernetes API except the console's read-only view and its single Secret write; nothing provisions itself at startup.

**What stays outside code, and why.** Secret values (section 10). The image tag, which the build produces and the deploy command takes from the commit. Cluster credentials: the kube context named in `Environment.cluster` must exist where the deploy runs, and that is operator identity, not app configuration.

**Rolling deploys and schema.** The migrate Job runs before new pods roll, so old pods briefly run against the new schema. Migrations are therefore expand-then-contract: add in one release, drop in a later one. The guide carries this rule, and `makemigration`'s autogenerated drop operations are the thing to hold back.

**Postgres through PgBouncer.** Connections grow with pods and the runtime's transactions are short and per run, so PgBouncer pools them. asyncpg's prepared-statement cache does not survive transaction pooling; the store sets `statement_cache_size=0` when `BACKEND_BUILDER_DATABASE_POOLED=true`, which the manifests set. `SET LOCAL ROLE` is transaction-scoped and unaffected.

**Scaling** between `min` and `max` for worker pools is set per environment in the first cut, with the scheduler's waiting-depth gauge per pool as the evidence the console shows. Automation on that signal (KEDA) comes back when there is a case for it (section 15).

## 12. Console

- **Pools:** for each pool, the reactions and pollers that can resolve to it and how, desired and ready replicas (from the Kubernetes API, read-only, shown in-cluster), poll loops, runs in flight, runs in the last 24 hours by outcome, p95, calls received with outcomes, cold starts. Replica counts against waiting depth over time, so `min` and `max` are numbers with evidence.
- **Pollers:** explicit and implicit, with waiting rows and their age, groups and runs, failures and dead letters, the implicit ones under their reaction's name. A poller whose pool has no replicas shows a growing feed, which is the honest signal.
- **Profiles:** the access matrix, profiles by entities, secrets and invoke edges, derived from code; every run row shows its profile.
- **Secrets:** declared names with set, when, by whom; set and rotate (section 10).
- **Runs:** the pool, placement source and profile on every invocation row; a call child as its own linked run with `replayed` marked; a poller run with its items.
- A heuristic flag for the grouped-reaction smell: a poller's reaction that wrote rows it did not read in the same run.

The console never writes to Kubernetes except the app's Secret, and never runs work except the operator's explicit triggers.

## 13. Local cluster

`make run-all` brings the whole system up on this machine, every process a pod:

1. **minikube** with the Docker driver, started with `--gpus=all` and the `nvidia-device-plugin` addon, so a pool with `gpu=1` gets the machine's GPU. kind cannot pass a GPU through and is not used. `make cluster` checks for minikube, kubectl and the NVIDIA container toolkit and prints the install commands when missing; it does not install system packages. Prometheus is installed into the cluster by the same target.
2. Build the app image, and any per-pool image the environment names, and load them into the cluster; there is no local registry.
3. Postgres as a StatefulSet in the cluster, one database per app, behind PgBouncer as in any environment.
4. Secrets from the repo's `.env`, checked against the App's declared names, into the app's Kubernetes Secret (section 10). The Makefile refuses this path when the kube context is not minikube.
5. `backend-builder deploy <app> --env local`; the migrate Job first.
6. Port-forward api, admin and Grafana and print the URLs.

Re-running `make run-all` is a rolling redeploy. `make cluster-down` deletes the cluster. The existing host and systemd targets stay as they are. `make e2e-k8s` runs the end-to-end suite against the cluster; compose remains the fast loop.

One physical GPU means `replicas={"ml": 1}` locally. The GTX 1080 is Pascal (compute capability 6.1): current CUDA 12, XGBoost and LightGBM GPU builds support it; recent PyTorch builds may not, in which case the fitting code's `device="auto"` falls back to CPU.

## 14. Order of work

1. **Runtime: placement, pools and pollers.** `Pool`; `pool` on reactions, triggers, endpoints and call sites including the input function; resolution with precedence and the pin constraint; the join-across-pools refusal. `Poller` with `poll`, `ack`, `release`, `execute`; `OutboxPoller` with the `rows` relation, the lease claim and bisection; implicit pollers replacing the worker's delivery loop, provably the same behaviour; `poller` and `pool` on outbox rows; `--pool` on the worker; `pool` and placement source on records and metrics; the scheduler's depth gauges; graceful shutdown; the file secrets provider; the pooled-connection setting. Shop and finance place their reactions and declare one explicit poller each. Migrations for both chains. One PR.
2. **Runtime: calls.** `mode="call"` in-process on its own connection with JSON in and out, the ledger with the derived key and `key=`, replay, in-transaction records for calls, call metrics. Finance's fits become calls where the sweep needs their result. One PR.
3. **Runtime: profiles.** `Profile`, `profile` on reactions, `invokes` checks at call and defer, the session's access check, `ctx.secret` allowlists, `SET LOCAL ROLE` at run start, generated role and grant migrations, `joinable`, the `operator` profile for console and CLI, `profile` on records. Shop declares profiles as the example. One PR.
4. **Cluster.** `Environment` on the App, `manifests` and `deploy` commands, per-pool login roles and scoped secret mounts, Ingress, PgBouncer, Prometheus, Alloy and Loki, NetworkPolicy, `make cluster`, `make run-all`, `make secrets`, `make cluster-down`, `make e2e-k8s`, the minikube and GPU checks, the CI manifest diff. One PR.
5. **Calls across processes.** `POST /calls` on the worker with the profile check, the client in `ctx.trigger`, refusal at `concurrency`. One PR.
6. **Console.** Pools, pollers, profiles and secrets pages, with the Kubernetes read and the Secret write. One PR, or folded into 4.

Each step leaves compose and the single-host setup working unchanged.

## 15. Considered and set aside

- **Letting placement imply the transaction mode.** Rejected: placement is for scaling and must be free to move any reaction anywhere; the mode is the developer's explicit statement, and a mode that placement cannot support is refused, never substituted.
- **Dropping `call` in favour of defer plus continuation.** Proposed in review because a committed child under a failed parent is a saga. Decided against: calls are how a service estate behaves, the developer chooses the mode, and the runtime makes the choice safe and visible.
- **An ordinal-only idempotency key.** Would replay the wrong child after a deploy that reorders calls while a retry is pending. Replaced by the reaction-plus-input key with an explicit override.
- **A static scan of reaction source for cross-pool joins.** Fragile; the refusal happens at the trigger, at runtime, identically in every environment.
- **Session over the wire for joined remote children.** Every read a network round trip inside a held serializable transaction; `call` covers the cases that want another pod.
- **Savepoints around joined children.** Decided against for now.
- **A synchronous independent child without a ledger.** Doubles effects on retry and breaks the parent's serializability; `call` exists to make this safe.
- **Distributed and autonomous transactions.** Neither gives one transaction across pods.
- **Batch triggers, pull mode with `ctx.take`, and within-flush coalescing** as separate features. All are a `Poller` with the right query; one concept replaced three.
- **A `translate` step on pollers.** Shaping items is the reaction's job, as for API reactions.
- **Key-level exclusivity for grouped pollers.** Unnecessary once a grouped reaction recomputes from current state; rows are exclusive, keys need not be.
- **Consumer-side selectors over the outbox.** A row could match two pollers or none; rows are addressed to one poller instead, and the query filters within that.
- **Entities marked confidential.** Profiles are positive allowlists; nothing is marked on the data.
- **Access declared on pools.** Where a reaction runs and what it may touch are different decisions; profiles carry access and pools derive their memberships.
- **Role switching inside a transaction for joined children.** A joined child runs as its parent; nothing is gained by joining, so there is nothing to switch.
- **Callee-side `triggered_by`.** Caller-side `invokes` answers the common question; a callee-side restriction can be added if a reaction ever needs it.
- **A hand-written per-environment YAML file.** Environments are typed objects on the App; rendered YAML is an artifact.
- **KEDA** for scaling pools on waiting depth and in-flight calls. The natural automation once pools exist, and the scheduler's gauge is exactly its signal; set aside for the first cut. Replicas are set per environment meanwhile.
- **Overflow between pools** and **placement by data locality.** The first hides saturation; the second has nothing local to place by.
- **An operator and CRD**, or **Helm**. Rendered manifests applied with prune give the same outcome for one repo without a controller to run.
- **The runtime provisioning Kubernetes at startup.** Needs write RBAC inside the app, hides drift, and cannot be reviewed before it happens.
- **Pod per run.** A pool with `concurrency=1` and long runs is pod-per-run in practice; a true Job-per-row flag waits for a reaction that proves it needs it.
