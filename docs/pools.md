# Compute Pools and Service Calls

Status: design, revised 2026-10-31 after review. Decided, not built. Companion to [design.md](design.md), which it extends; where the two disagree, this document is newer. Sections 1 to 10 are the design; section 11 is the order of work; section 12 lists what was considered and set aside.

## Intent

Run every reaction where its work belongs and scale it on its own evidence. Today an app runs as four process types: api, worker, scheduler, admin. Every worker claims every outbox row, so a reaction that needs a GPU and a reaction that totals an order scale together, on the same machines, with the same resources. The aim is to let the app declare **pools** of compute, say which reactions run in which pool, down to one pool per reaction, have a deploy create and scale those pools in Kubernetes, and watch them in the console.

Pools make the runtime behave like a microservice estate: a reaction in another pool is another service, and a reaction that triggers it is, when the developer says so, a **call**: it commits on its own and returns a result, as a service call does. The runtime keeps the two other relationships a trigger can have, joining the parent's transaction or running after it, and makes the developer name which one they mean. Picking the right one is their obligation; making each one safe, and showing in the records which one happened, is the runtime's.

Cost is minimised by scaling each pool on its own queue and load, to zero where the developer allows; availability is kept by pools declaring a floor. Both are properties of the declaration, visible in the repo and the console.

## 1. The transaction decides the shape

Everything here follows from one fact of Postgres: a transaction belongs to one connection. A second connection cannot join it, see its uncommitted writes, or be rolled back with it. Savepoints are scoped to one transaction; snapshot export shares a read view, not a transaction; two-phase commit makes two transactions agree on an outcome without making them one.

Therefore a sub-reaction that shares its parent's transaction can only run in the parent's process, and a sub-reaction in another pod is in another transaction, by construction. There are exactly three relationships a trigger can have with its parent, and every trigger is one of them.

## 2. Three trigger modes

```python
await ctx.trigger(Child, input)                          # join: same transaction, same session, in-process
await ctx.trigger(Child, input, mode="call")             # call: own run, own transaction, now; returns a result
await ctx.trigger(Child, input, mode="call", key="...")  # call with an explicit idempotency key
await ctx.trigger(Child, input, deferred=True)           # defer: outbox row at flush; runs after commit
```

`mode` is `"join"` (default), `"call"` or `"defer"`; `deferred=True` remains as the spelling of `mode="defer"`.

**join.** Unchanged. The child runs in the parent's session and transaction, depth-first, in the parent's process, whatever pool the child's own runs are declared in, and returns its context. Its writes commit with the run or not at all. A joined child that raises aborts the run unless the parent catches the exception, in which case the child's writes stay in the session, as today; no savepoint is added (decided). Join is the right mode whenever the child's writes must stand or fall with the parent's.

**call.** The service call. The child runs now as a run of its own, through `run()`, in its own serializable transaction, in the pool that owns it, and commits. The parent's transaction is untouched; the parent blocks until the child returns, within the parent's remaining budget. The parent receives a `CallResult`: the child's saved vars, its `reaction_id`, and `replayed` (section 3). If the child raises, the parent gets the exception, as with join. If the parent later fails, **the child's commit stands**. Undoing a committed call when the parent fails is a compensating reaction the developer triggers from the failure path; that is a saga, and the guide says so. Call is the right mode when the child is another service: its own data, its own pace, its own scaling, a result the parent needs now.

**defer.** Unchanged. An outbox row written at flush, claimed by the owning pool's worker after the parent commits, retried and dead-lettered by the outbox's rules. The parent gets no result; a result travels the event-driven way, through an entity the child writes and a trigger on it. Defer is the right mode when the parent does not need the result and the work should not hold the parent's transaction.

Entities are shared within a run and across nothing else; vars belong to one invocation; a child sees only its input. These rules from [design.md](design.md) §4 are unchanged and are what make a call's input and result serialisable: input is a JSON object, the result is the child's vars as JSON.

## 3. Call idempotency: the ledger

A run is retried from the seed on any exception, so a parent that calls a child and then fails would call it again. Service calls need an idempotency key, and the runtime derives one so that the common case needs no thought and the uncommon case has an escape hatch.

**The key.** `uuid5(parent_reaction_id, reaction + canonical_json(input) + n)`, where the parent's `reaction_id` is stable across its attempts, `canonical_json` is the input with sorted keys, and `n` is how many earlier calls in this attempt had the same reaction and input (almost always 0). The same call with the same input on a retry gets the same id. Adding, removing or reordering *other* calls in the parent between deploys does not change it, which is the flaw an ordinal-only key would have had: a retry across a deploy would replay the wrong child's result. Two identical calls in one parent are told apart by `n`. When the input carries something that varies between attempts, a timestamp or a fresh uuid, the developer passes `key="..."` and owns stability; the guide says a call's input should be a function of the parent's input, as everything else in a run is.

**The record is the ledger.** For a call, the run record is written **inside the child's own transaction**. The child is its own transaction, so this is possible, and it makes "the child committed" and "the ledger says the child committed" one fact. The record carries `source="call"`, the caller's `root_id` and `parent_reaction_id`, the derived id, and its outcome.

**Replay.** Before running a call, the runtime looks up the id in `_run`. A committed record means the child already ran: its saved vars are returned with `replayed = true` and nothing executes. A failed record, or none, means run it. A `running` record means an earlier attempt's child is still executing or died mid-way; the caller waits up to its budget for it to resolve and treats a stale one as failed (the scheduler's lost sweep applies).

Consequences: a child's commit that causes the parent's serialization failure does not run the child twice on retry; a parent that fails after a call and is retried sees the result the first attempt saw; a call's effects happen exactly once per committed call, or not at all. Not bought: undoing the child if the parent gives up for good.

## 4. Pools

```python
from backend_builder import App, Pool

pools = [
    Pool("api", min=2, max=10, cpu="500m", memory="512Mi"),
    Pool("default", min=1, max=8, concurrency=16),
    Pool("pricing", reactions=[PriceOrder], min=1, max=20, concurrency=8),
    Pool("ml", reactions=[FitModel, Infer], min=0, max=4, concurrency=1,
         cpu="4", memory="8Gi", gpu=1),
]
app = App(package="finance", endpoints=endpoints, triggers=triggers, pools=pools)
```

A `Pool` declares: `name`; `reactions`, the classes whose **runs** it executes; `min` and `max` replicas; `concurrency`, runs in flight per replica; `cpu`, `memory`, `gpu` requests as defaults the environment may override. One pool per reaction is an ordinary configuration, not a special case.

**A pool places runs, not invocations.** A reaction's pool is where its runs execute: outbox deliveries, scheduled occurrences, and calls to it. A joined child is not a run; it executes in its parent's process, wherever that is, inside the parent's transaction. So declaring `PriceOrder` in the `pricing` pool means every *call* to it and every *deferred* row for it goes to `pricing` replicas, while a reaction that joins it runs it in-process. There is no cross-pool rule for join and nothing to analyse statically; the records' `pool` column shows where each run executed, and a developer who joins a GPU reaction from an api request will see `api` there.

Rules checked when the App is built:

- Every reaction belongs to exactly one pool. `default` owns every reaction not listed elsewhere and exists even if undeclared. `api` exists even if undeclared and owns no reactions; it is where endpoints run their seed.
- `concurrency >= 1`, `0 <= min <= max`, `max >= 1`.

Guidance, not rules: a pool that receives calls should keep `min >= 1`, because a cold start inside a caller's held transaction is latency the caller pays; the console's pools page shows a pool's cold starts so the choice is informed. A GPU pool is typically `concurrency=1`.

What a pool does not declare: image, tag, database URL, secrets, namespace, node selectors, tolerations, scaling thresholds, environment-specific replica counts. Those are the environment's (section 7), and every resource request in code is a default the environment file may override per pool.

Pools are visible wherever the app describes itself: `backend-builder graph` prints them with their reactions; the console shows them (section 9); `describe()` includes them.

## 5. Runtime changes

**Worker.** `backend-builder worker <app> --pool <name>` claims only rows whose reaction belongs to that pool, `reaction = any(:names)` inside the same `FOR UPDATE SKIP LOCKED` statement, so pools never contend. A worker for a pool the app does not declare refuses to start. Without `--pool`, a worker claims for every pool, which is what compose and single-host setups keep using.

**Scheduler.** Unchanged, one replica, restarted by its Deployment if it dies; nothing is lost while it is down, since seeding and the lost sweep catch up on the next tick. Seeded rows are claimed by the owning pool like any other.

**Records.** `_run` gains `pool` on every row and `source` gains `"call"`. A call's record is written in-transaction (section 3); every other record is written after the run, as now. A call child appears in its caller's tree as its own run, linked, with `replayed` marked.

**Metrics.** `pool` label on `bb_runs_total`, `bb_run_duration_seconds` and `bb_outbox_deliveries_total`; new gauges `bb_outbox_pending{pool}` and `bb_runs_in_flight{pool}`; new counter `bb_calls_total{reaction, outcome}` with outcomes `ok`, `error`, `replayed`, `unreachable`, and histogram `bb_call_duration_seconds{reaction}`, which includes any cold start the caller waited through.

**Graceful shutdown.** On `SIGTERM` a worker stops claiming and accepting calls, lets in-flight runs finish within the pod's grace period, and releases what it cannot finish. The manifests set `terminationGracePeriodSeconds` above the pool's longest trigger timeout. A pod killed harder loses only its in-flight runs, which roll back and are redelivered when their leases lapse, as today.

**Secrets.** A file provider in front of the environment provider: `ctx.secret(name)` reads `/run/secrets/backend-builder/SECRET_<NAME>` if the file exists, else the variable. Reading stays on access, so a rotated mounted Secret is picked up without a restart. Compose, tests and host setups keep using variables and notice nothing. Redaction is unchanged.

**Postgres through PgBouncer.** The manifests put PgBouncer in transaction pooling mode in front of Postgres, since connections grow with pods and the runtime's transactions are short and per run. asyncpg's prepared-statement cache does not survive transaction pooling, so the store sets `statement_cache_size=0` when `BACKEND_BUILDER_DATABASE_POOLED=true`, which the manifests set. Nothing else in the store changes; the runtime never uses session-level state across transactions.

## 6. Remote calls

A call whose child lives in another pool is a request to that pool. Each pool's worker process also serves, on its metrics port, `POST /calls`: body `{"run_id", "reaction", "input", "root_id", "parent_reaction_id", "deadline"}`, response the child's vars and outcome. The worker executes it through the same `run()` as an outbox delivery, with the ledger check first, counting it against its `concurrency`. The caller connects to the pool's Kubernetes Service, so load balancing is the cluster's, and passes its remaining budget as the deadline; the callee refuses work whose deadline has passed and sheds calls when at `concurrency`, which the caller treats as unreachable and retries within its budget.

Failure mapping: a non-2xx or a dropped connection is an exception in the caller, so the run fails or retries like any other error, and the retry consults the ledger, so a child that did commit is replayed rather than re-run. A caller that times out while the child is still running leaves a `running` ledger row that resolves when the child finishes or is swept as lost.

In-process calls, when caller and callee share a pool, skip the network and execute `run()` directly on a second connection. Semantics are identical; compose and tests exercise exactly the same ledger and replay paths without a cluster.

Joined children never cross the network: there is no session-over-the-wire protocol (section 12).

## 7. Manifests and deploy

`backend-builder manifests <app> --env <name>` renders Kubernetes YAML from the App's declaration merged with `deploy/<name>.yaml`, the environment file: namespace, image and tag, an optional image per pool (the GPU pool carries CUDA libraries; the others need not), node selectors and tolerations per pool, replica and resource overrides per pool, scaling thresholds, database URL source, the secret source, the api's host and TLS. Rendered objects, in a namespace per app:

- one Deployment per pool running `worker <app> --pool <name>`, with the pool's resources, `nvidia.com/gpu` when `gpu` is set, the secret volume, a Service, and a KEDA `ScaledObject` between `min` and `max` on two triggers: pending outbox rows for the pool's reactions (a Postgres scaler) and in-flight calls per replica (a Prometheus scaler on `bb_runs_in_flight{pool}`), so a pool scales on its own queue and its own load, to zero when `min = 0`;
- the `api` Deployment running `serve`, its Service, an Ingress with the environment's host and TLS, and an HPA on CPU between `min` and `max`;
- one scheduler Deployment, one admin Deployment with its Service and an Ingress behind the admin token;
- a migrate Job, applied first and waited on before anything else rolls;
- a PodDisruptionBudget for every pool with `min >= 2`;
- PgBouncer, and Prometheus scraping every process's metrics port, which KEDA's Prometheus scaler and Grafana both read;
- Alloy as a DaemonSet and Loki with a volume, from [logs.md](logs.md) §2, with the one-day retention.

Deploy is `kubectl apply --prune -l app=<name> -n <namespace>` of the rendered YAML, run by the pipeline after the image is pushed. A pool added in code appears on the next deploy; a pool removed is deleted. The YAML is committed, so a PR that adds a pool shows the Deployment it creates, the same way a PR that adds a field shows its migration. The runtime never talks to the Kubernetes API except the console's read-only view (section 9); nothing provisions itself at startup.

**Rolling deploys and schema.** The migrate Job runs before new pods roll, so old pods briefly run against the new schema. Migrations are therefore expand-then-contract: add columns and tables in one release, drop in a later one, once no running code references them. The guide carries this rule, and `makemigration`'s autogenerated drop operations are the thing to hold back.

## 8. Cost and availability

Each pool's cost is `replicas × resources`, and the declaration decides replicas within `min` and `max`: KEDA raises them on queue depth or in-flight calls and lowers them when idle, to zero where allowed. Pools that serve calls or requests keep a floor; everything that only consumes the outbox can sleep. A PodDisruptionBudget keeps floors through node drains; node groups and spot are the environment's business. The console's pools page shows replicas over time against queue depth and p95, so a `min` or `max` is a number with evidence behind it, and a pool whose replicas sit at `max` with a growing queue is visible before it is a page.

## 9. Console

A pools page: for each pool, its reactions, desired and ready replicas, pending outbox rows, runs in flight, runs in the last 24 hours by outcome, p95 duration, calls received with their outcomes, and cold starts observed. Replica counts come from the Kubernetes API with read-only RBAC on Deployments in the app's namespace, shown only when the console runs in-cluster; everything else comes from `_run`, `_outbox` and the metrics. The run page shows the pool on every invocation row. The console never writes to Kubernetes.

## 10. Local cluster

`make run-all` brings the whole system up on this machine, every process a pod:

1. **minikube** with the Docker driver, started with `--gpus=all` and the `nvidia-device-plugin` addon, so a pool with `gpu=1` gets the machine's GPU. kind cannot pass a GPU through and is not used. `make cluster` checks for minikube, kubectl and the NVIDIA container toolkit and prints the install commands when missing; it does not install system packages. KEDA and Prometheus are installed into the cluster by the same target.
2. Build the app image, and the GPU pool's image if the environment names one, and load them into the cluster; there is no local registry.
3. Postgres as a StatefulSet in the cluster, one database per app, the same init script compose uses, behind PgBouncer as in any environment.
4. **Secrets from the repo's `.env`**, which is gitignored. A committed `.env.example` lists every `SECRET_*` key; `make run-all` stops with the missing names if `.env` lacks any. The `SECRET_*` lines become one Kubernetes Secret via dry-run and apply, never written to YAML or git, mounted as files in every pool's pods. Editing `.env` and running `make secrets` rotates values live. The Makefile refuses this path when the kube context is not minikube, so a laptop `.env` cannot reach another cluster by habit. In a real cluster the same mount is filled by an external secrets operator from a vault; the pods cannot tell.
5. Render and apply the manifests for `--env local`; the migrate Job first.
6. Port-forward api, admin and Grafana and print the URLs.

Re-running `make run-all` is a rolling redeploy. `make cluster-down` deletes the cluster. The existing host and systemd targets stay as they are. A `make e2e-k8s` target runs the end-to-end suite against the cluster instead of compose; compose remains the fast loop.

One physical GPU means `max=1` for the GPU pool locally, which the local environment file sets. The GTX 1080 is Pascal (compute capability 6.1): current CUDA 12, XGBoost and LightGBM GPU builds support it; recent PyTorch builds may not, in which case the fitting code's `device="auto"` falls back to CPU.

## 11. Order of work

1. **Runtime: pools.** `Pool`, validation on `App`, `--pool` on the worker, claim-by-pool, `pool` on records and metrics, graceful shutdown, the file secrets provider, the pooled-connection setting. Shop and finance declare pools. Migrations for both chains. One PR.
2. **Runtime: calls.** `mode="call"` in-process with the ledger, the derived key and `key=`, replay, in-transaction records for calls, `bb_calls_total` and `bb_call_duration_seconds`. Finance's fits become calls where the sweep needs their result. One PR.
3. **Cluster.** `manifests` command with the environment file, `deploy/local.yaml`, KEDA and Prometheus, Ingress, PgBouncer, Alloy and Loki, `make cluster`, `make run-all`, `make secrets`, `make cluster-down`, `make e2e-k8s`, the minikube and GPU checks. One PR.
4. **Remote calls.** `POST /calls` on the worker, the client in `ctx.trigger`, shedding at `concurrency`. One PR.
5. **Console pools page**, with the Kubernetes read and cold starts. One PR, or folded into 3.

Each step leaves compose and the single-host setup working unchanged.

## 12. Considered and set aside

- **Dropping `call` in favour of defer plus continuation.** Proposed in review because a committed child under a failed parent is a saga, and because fits run minutes. Decided against: calls are how a microservice estate behaves, the developer chooses the mode, and the runtime's job is to make the choice safe and visible, which the ledger and the records do. The guide states the consequences plainly.
- **An ordinal-only idempotency key.** The first draft derived a call's id from its position among the parent's calls. A deploy that adds or reorders calls while a retry is pending would replay the wrong child's result. Replaced by the reaction-plus-input key with an explicit override (section 3).
- **A cross-pool rule for joined triggers**, enforced by scanning reaction source. Dropped: a pool places runs, and a joined child is not a run; it executes with its parent, and the records show where.
- **Session over the wire for joined remote children.** The parent keeps the transaction and the child's store calls travel back to it. Correct and buildable, but every read is a network round trip inside a held serializable transaction, and `call` covers the cases that want another pod.
- **Savepoints around joined children.** Decided against for now; a caught child's writes stay in the session.
- **A synchronous independent child without a ledger.** It would double its effects on every retry and break the parent's serializability. `call` exists precisely to make this safe.
- **Distributed transactions** (two-phase commit) and **autonomous transactions**. Neither gives one transaction across pods.
- **Scaling by hand from the console.** The first draft set `replicas: min` and left scaling to a human reading depth. Reversed in review: scaling on each pool's own queue and load is the point of pools and the main cost lever, so KEDA is in the first cluster.
- **An operator and CRD**, or **Helm**. Generated manifests applied with prune give the same outcome for one repo without a controller to run.
- **The runtime provisioning Kubernetes at startup.** Needs write RBAC inside the app, hides drift, and cannot be reviewed before it happens.
- **Pod per run**, a Job per outbox row. A pool with `concurrency=1` and long runs is pod-per-run in practice; a true Job-per-row flag waits for a reaction that proves it needs it.
- **Secret scoping per pool.** Needs per-reaction secret declarations; set aside until a pool runs less trusted code.
