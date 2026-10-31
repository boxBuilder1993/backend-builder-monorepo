# Compute Pools and Service Calls

Status: design, revised 2026-10-31. Decided, not built. Companion to [design.md](design.md), which it extends; where the two disagree, this document is newer. Sections 1 to 10 are the design; section 11 is the order of work; section 12 lists what was considered and set aside.

## Intent

Scale horizontally by moving reactions between pools of compute. Today an app runs as four process types: api, worker, scheduler, admin. Every worker claims every outbox row, so a reaction that needs a GPU and a reaction that totals an order scale together, on the same machines, with the same resources. The aim is to let the app declare **pools**, place any reaction's work in any pool by several means, down to one pool per reaction, have a deploy create those pools in Kubernetes, and watch them in the console.

Placement and transactions are kept apart. A pool is where work runs. The transactional relationship between a reaction and a child it triggers is a separate, explicit choice: join the parent's transaction, call as another service would be called, or defer until after commit. Placement never changes that choice behind the developer's back. The one thing placement does is make some choices unavailable: a child in another pool is in another process, and no process can join another's transaction. The developer is told so, at the trigger, and picks another mode. Knowing which pools can and cannot share a transaction is part of knowing the app.

## 1. The transaction decides what is possible

Everything here follows from one fact of Postgres: a transaction belongs to one connection. A second connection cannot join it, see its uncommitted writes, or be rolled back with it. Savepoints are scoped to one transaction; snapshot export shares a read view, not a transaction; two-phase commit makes two transactions agree on an outcome without making them one.

Therefore a child that shares its parent's transaction must run in the parent's process, and a child in another pool is another transaction, by construction. There are exactly three relationships a trigger can have with its parent.

## 2. Three trigger modes

```python
await ctx.trigger(Child, input)                          # join: same transaction, same session, same process
await ctx.trigger(Child, input, mode="call")             # call: own run, own transaction, now; returns a result
await ctx.trigger(Child, input, mode="call", key="...")  # call with an explicit idempotency key
await ctx.trigger(Child, input, deferred=True)           # defer: outbox row at flush; runs after commit
```

`mode` is `"join"` (default), `"call"` or `"defer"`; `deferred=True` remains as the spelling of `mode="defer"`. The mode is the developer's statement of what the child's writes mean to the parent's.

**join.** Unchanged. The child runs in the parent's session and transaction, depth-first, and returns its context. It sees the parent's uncommitted entities and mutates them in place; its writes commit with the run or not at all. A joined child that raises aborts the run unless the parent catches the exception, in which case the child's writes stay in the session, as today; no savepoint is added (decided). **Join requires the child to resolve to the parent's pool** (section 4). A join whose child resolves elsewhere raises `TriggerError` before anything runs: *join is not supported across pools; use mode="call" or deferred=True*. Nothing is converted silently.

**call.** The service call. The child runs now as a run of its own, through `run()`, in its own serializable transaction, in whichever pool it resolves to, the parent's included, and commits. It sees committed state only: **not** the parent's uncommitted writes, so everything it needs arrives in its input. The parent's transaction is untouched; the parent blocks until the child returns, within the parent's remaining budget. The parent receives a `CallResult`: the child's saved vars as JSON, its `reaction_id`, and `replayed` (section 3). If the child raises, the parent gets the exception. If the parent later fails, **the child's commit stands**; undoing it is a compensating reaction the developer triggers from the failure path, which is a saga, and the guide says so. A call behaves identically whether the child's pool shares the parent's process or sits across the cluster: input and result always cross as JSON, and a non-JSON value in either, an entity object for one, fails at the call site, so code behaves the same under compose and in the cluster.

**defer.** Unchanged. An outbox row written at flush, claimed by a worker of the child's pool after the parent commits, retried and dead-lettered by the outbox's rules. The parent gets no result; a result travels the event-driven way, through an entity the child writes and a trigger on it.

Across every mode: input is the only thing a parent gives a child, vars belong to one invocation, and a child's saved vars reach the parent only through the returned context or `CallResult`. These rules from [design.md](design.md) §4 are unchanged.

## 3. Call idempotency: the ledger

A run is retried from the seed on any exception, so a parent that calls a child and then fails would call it again. Service calls need an idempotency key, and the runtime derives one so the common case needs no thought and the uncommon case has an escape hatch.

**The key.** `uuid5(parent_reaction_id, reaction + canonical_json(input) + n)`, where the parent's `reaction_id` is stable across its attempts, `canonical_json` is the input with sorted keys, and `n` counts earlier calls in this attempt with the same reaction and input (almost always 0). The same call with the same input on a retry gets the same id. Adding, removing or reordering *other* calls between deploys does not change it; an ordinal-only key would have replayed the wrong child after a deploy that reordered calls while a retry was pending. When the input carries something that varies between attempts, the developer passes `key="..."` and owns stability; the guide says a call's input should be a function of the parent's input, as everything else in a run is.

**The record is the ledger.** For a call, the run record is written **inside the child's own transaction**, so "the child committed" and "the ledger says the child committed" are one fact. The record carries `source="call"`, the caller's `root_id` and `parent_reaction_id`, the derived id, and its outcome.

**Replay.** Before running a call, the runtime looks up the id in `_run`. A committed record means the child already ran: its saved vars are returned with `replayed = true` and nothing executes. A failed record, or none, means run it. A `running` record means an earlier attempt's child is still executing or died mid-way; the caller waits up to its budget for it to resolve and treats a stale one as failed (the scheduler's lost sweep applies).

Consequences: a child's commit that causes the parent's serialization failure does not run the child twice on retry; a retried parent sees the result the first attempt saw; a call's effects happen exactly once per committed call, or not at all. Not bought: undoing the child if the parent gives up for good.

## 4. Placement

A pool is where a reaction's work executes. Placement is resolved before a run or an invocation starts, from these sources, lowest precedence first:

1. **Default.** The `default` pool, which exists even if undeclared.
2. **Inherited.** A child runs in its parent's pool. A seed runs where the thing that started it says: an endpoint's seed in `api` unless the endpoint names a pool, a schedule's or data-change trigger's run in the pool the trigger names, else `default`.
3. **Trigger-site.** `OnCreate(Order, Notify, pool="side")`, `OnSchedule("0 3 * * *", Rollup, pool="nightly")`, `Endpoint(..., pool="reports")`. Where a seed runs is a property of what starts it; a heavy synchronous endpoint does not belong in `api`.
4. **Class pin.** `pool = "ml"` on the reaction class. The reaction runs there whoever triggers it, for a resource it needs: a GPU, memory, a licence. A pin is a constraint: a call site naming a different pool for a pinned reaction is a `TriggerError`.
5. **Call-site.** `ctx.trigger(X, input, mode="call", pool="batch")` or `deferred=True, pool=...`. The same reaction can be cheap in one context and heavy in another, and the caller may know what the class cannot. Not valid with `join`, which has no choice of pool.
6. **By input.** Wherever a pool name is accepted, a function of the input is accepted too: `pool=lambda input: "gpu" if input["rows"] > 1_000_000 else "cpu"`, or one pool per tenant. Resolved at the trigger, recorded like any other placement.

The resolved pool is recorded on every invocation's row, so the console always shows where work ran and the records show why a join was refused.

**Exclusive pools.** A pool that owns one pinned reaction and nothing else never shares a process with other work. It is pattern 4 with a dedicated pool, not a new mechanism; the console marks a pool exclusive when exactly one reaction can resolve to it.

Patterns left out: overflow from a saturated pool to another, which hides saturation instead of showing it, the outbox queue being the honest signal; and placement by data locality, since nothing in the runtime is local to a node.

## 5. Pools

```python
from backend_builder import App, Pool

pools = [
    Pool("api", min=2, max=10, cpu="500m", memory="512Mi"),
    Pool("default", min=1, max=8, concurrency=16),
    Pool("pricing", min=1, max=20, concurrency=8),
    Pool("ml", min=0, max=4, concurrency=1, cpu="4", memory="8Gi", gpu=1),
]
app = App(package="finance", endpoints=endpoints, triggers=triggers, pools=pools)
```

A `Pool` declares shape only: `name`, `min` and `max` replicas, `concurrency` (runs in flight per replica), and `cpu`, `memory`, `gpu` as defaults the environment may override. It lists no reactions; reactions and triggers name their pool (section 4). `default` and `api` exist even if undeclared; `api` runs endpoint seeds and whatever they join, and nothing else resolves to it unless placed there.

Rules checked when the App is built: every pool named anywhere, on a class, a trigger, an endpoint, is declared; `concurrency >= 1`, `0 <= min <= max`, `max >= 1`. Guidance, not rules: a pool that receives calls should keep `min >= 1`, since a cold start inside a caller's held transaction is latency the caller pays; the console's pools page shows cold starts so the choice is informed. A pool that only receives deferred rows can sit at `min = 0`. A GPU pool is typically `concurrency = 1`.

What a pool does not declare: image, tag, database URL, secrets, namespace, node selectors, tolerations, environment-specific replica counts. Those are the environment's (section 7), and every resource request in code is a default the environment file may override per pool.

Pools are visible wherever the app describes itself: `backend-builder graph` prints them with the reactions that can resolve to each; the console shows them (section 9); `describe()` includes them.

## 6. Runtime changes

**Worker.** `backend-builder worker <app> --pool <name>` claims only outbox rows resolved to that pool. Rows carry their resolved `pool`, set when enqueued, so the claim is `pool = :name` inside the same `FOR UPDATE SKIP LOCKED` statement and pools never contend. A worker for a pool the app does not declare refuses to start. Without `--pool`, one worker serves every pool, which is what compose and single-host setups keep using; placement is still resolved and recorded, joins are still refused across pools, and cross-pool calls still run as calls on their own connection, so behaviour does not depend on how many processes there are.

**Calls across processes.** A call whose child resolves to a pool served by another process is a request to that pool: each pool's worker serves `POST /calls` on its metrics port, body `{"run_id", "reaction", "input", "root_id", "parent_reaction_id", "deadline"}`, response the child's vars and outcome. The worker executes it through the same `run()` as an outbox delivery, ledger check first, counting it against its `concurrency` and shedding when full, which the caller treats as unreachable and retries within its budget. The caller connects to the pool's Kubernetes Service and passes its remaining budget as the deadline; the callee refuses work whose deadline has passed. A non-2xx or a dropped connection is an exception in the caller, so the run fails or retries like any other error, and the retry consults the ledger. Pods trust each other inside the app's namespace; a NetworkPolicy limits `/calls` to the app's pods.

**Scheduler.** Unchanged, one replica, restarted by its Deployment if it dies; nothing is lost while it is down, since seeding and the lost sweep catch up on the next tick. It also exports pending outbox depth per pool, since a pool at zero replicas has no worker to export anything.

**Records.** `_run` gains `pool` on every row and `source` gains `"call"`. `_outbox` gains `pool`. A call's record is written in-transaction (section 3); every other record is written after the run, as now. A call child appears in its caller's tree as its own run, linked, with `replayed` marked. A refused join is recorded as the parent's error.

**Metrics.** `pool` label on `bb_runs_total`, `bb_run_duration_seconds` and `bb_outbox_deliveries_total`; gauges `bb_outbox_pending{pool}` (from the scheduler) and `bb_runs_in_flight{pool}`; counter `bb_calls_total{reaction, outcome}` with outcomes `ok`, `error`, `replayed`, `unreachable`, `refused`; histogram `bb_call_duration_seconds{reaction}`, which includes any cold start the caller waited through.

**Graceful shutdown.** On `SIGTERM` a worker stops claiming and accepting calls, lets in-flight runs finish within the pod's grace period, and releases what it cannot finish. The manifests set `terminationGracePeriodSeconds` above the pool's longest trigger timeout. A pod killed harder loses only its in-flight runs, which roll back and are redelivered when their leases lapse, as today.

**Secrets.** A file provider in front of the environment provider: `ctx.secret(name)` reads `/run/secrets/backend-builder/SECRET_<NAME>` if the file exists, else the variable. Reading stays on access, so a rotated mounted Secret is picked up without a restart. Compose, tests and host setups keep using variables and notice nothing. Redaction is unchanged.

**Postgres through PgBouncer.** The manifests put PgBouncer in transaction pooling mode in front of Postgres, since connections grow with pods and the runtime's transactions are short and per run. asyncpg's prepared-statement cache does not survive transaction pooling, so the store sets `statement_cache_size=0` when `BACKEND_BUILDER_DATABASE_POOLED=true`, which the manifests set.

## 7. Manifests and deploy

`backend-builder manifests <app> --env <name>` renders Kubernetes YAML from the App's declaration merged with `deploy/<name>.yaml`, the environment file: namespace, image and tag, an optional image per pool (the GPU pool carries CUDA libraries; the others need not), node selectors and tolerations per pool, replica and resource overrides per pool, database URL source, the secret source, the api's host and TLS. Rendered objects, in a namespace per app:

- one Deployment per pool running `worker <app> --pool <name>`, with the pool's resources, `nvidia.com/gpu` when `gpu` is set, `replicas` from the environment's override or `min`, the secret volume, and a Service;
- the `api` Deployment running `serve`, its Service, an Ingress with the environment's host and TLS, and an HPA on CPU between `min` and `max`;
- one scheduler Deployment, one admin Deployment with its Service and an Ingress behind the admin token;
- a migrate Job, applied first and waited on before anything else rolls;
- a PodDisruptionBudget for every pool with `min >= 2`; a NetworkPolicy for `/calls`;
- PgBouncer, and Prometheus scraping every process's metrics port, which Grafana reads;
- Alloy as a DaemonSet and Loki with a volume, from [logs.md](logs.md) §2, with the one-day retention.

Worker pools do not autoscale in the first cut: `replicas` is `min` or the environment's override per pool, and the console shows pending depth and in-flight runs per pool so the number is chosen with evidence. Automation on those signals comes back when there is a case for it (section 12).

Deploy is `kubectl apply --prune -l app=<name> -n <namespace>` of the rendered YAML, run by the pipeline after the image is pushed. A pool added in code appears on the next deploy; a pool removed is deleted. The YAML is committed, so a PR that adds a pool shows the Deployment it creates, the same way a PR that adds a field shows its migration. The runtime never talks to the Kubernetes API except the console's read-only view (section 9); nothing provisions itself at startup.

**Rolling deploys and schema.** The migrate Job runs before new pods roll, so old pods briefly run against the new schema. Migrations are therefore expand-then-contract: add in one release, drop in a later one, once no running code references what is dropped. The guide carries this rule, and `makemigration`'s autogenerated drop operations are the thing to hold back.

## 8. Cost and availability

Each pool's cost is `replicas × resources`. Pools that serve calls or requests keep a floor; everything that only consumes the outbox can sit at zero and be raised by the environment when depth warrants. A PodDisruptionBudget keeps floors through node drains; node groups and spot are the environment's business. The console's pools page shows replicas against queue depth and p95, so a `min` or `max` is a number with evidence behind it, and a pool whose queue grows while its replicas sit still is visible before it is a page.

## 9. Console

A pools page: for each pool, the reactions that can resolve to it and how (pin, trigger, endpoint, call-site), desired and ready replicas, pending outbox rows, runs in flight, runs in the last 24 hours by outcome, p95 duration, calls received with their outcomes, cold starts observed, and whether it is exclusive. Replica counts come from the Kubernetes API with read-only RBAC on Deployments in the app's namespace, shown only when the console runs in-cluster; everything else comes from `_run`, `_outbox` and the metrics. The run page shows the pool and the placement source on every invocation row. The console never writes to Kubernetes.

## 10. Local cluster

`make run-all` brings the whole system up on this machine, every process a pod:

1. **minikube** with the Docker driver, started with `--gpus=all` and the `nvidia-device-plugin` addon, so a pool with `gpu=1` gets the machine's GPU. kind cannot pass a GPU through and is not used. `make cluster` checks for minikube, kubectl and the NVIDIA container toolkit and prints the install commands when missing; it does not install system packages. Prometheus is installed into the cluster by the same target.
2. Build the app image, and the GPU pool's image if the environment names one, and load them into the cluster; there is no local registry.
3. Postgres as a StatefulSet in the cluster, one database per app, the same init script compose uses, behind PgBouncer as in any environment.
4. **Secrets from the repo's `.env`**, which is gitignored. A committed `.env.example` lists every `SECRET_*` key; `make run-all` stops with the missing names if `.env` lacks any. The `SECRET_*` lines become one Kubernetes Secret via dry-run and apply, never written to YAML or git, mounted as files in every pool's pods. Editing `.env` and running `make secrets` rotates values live. The Makefile refuses this path when the kube context is not minikube, so a laptop `.env` cannot reach another cluster by habit. In a real cluster the same mount is filled by an external secrets operator from a vault; the pods cannot tell.
5. Render and apply the manifests for `--env local`; the migrate Job first.
6. Port-forward api, admin and Grafana and print the URLs.

Re-running `make run-all` is a rolling redeploy. `make cluster-down` deletes the cluster. The existing host and systemd targets stay as they are. A `make e2e-k8s` target runs the end-to-end suite against the cluster instead of compose; compose remains the fast loop.

One physical GPU means `max=1` for the GPU pool locally, which the local environment file sets. The GTX 1080 is Pascal (compute capability 6.1): current CUDA 12, XGBoost and LightGBM GPU builds support it; recent PyTorch builds may not, in which case the fitting code's `device="auto"` falls back to CPU.

## 11. Order of work

1. **Runtime: placement and pools.** `Pool`, the `pool` attribute on reactions, triggers and endpoints, `pool=` at call sites including the input function, resolution with precedence and the pin constraint, the join-across-pools refusal, `--pool` on the worker, `pool` on outbox rows, records and metrics, the scheduler's depth gauge, graceful shutdown, the file secrets provider, the pooled-connection setting. Shop and finance place their reactions. Migrations for both chains. One PR.
2. **Runtime: calls.** `mode="call"` in-process on its own connection with JSON in and out, the ledger with the derived key and `key=`, replay, in-transaction records for calls, call metrics. Finance's fits become calls where the sweep needs their result. One PR.
3. **Cluster.** `manifests` command with the environment file, `deploy/local.yaml`, Ingress, PgBouncer, Prometheus, Alloy and Loki, NetworkPolicy, `make cluster`, `make run-all`, `make secrets`, `make cluster-down`, `make e2e-k8s`, the minikube and GPU checks. One PR.
4. **Calls across processes.** `POST /calls` on the worker, the client in `ctx.trigger`, shedding at `concurrency`. One PR.
5. **Console pools page**, with the Kubernetes read, placement sources and cold starts. One PR, or folded into 3.

Each step leaves compose and the single-host setup working unchanged.

## 12. Considered and set aside

- **Letting placement imply the transaction mode** (same pool joins, another pool calls). Rejected: placement is for scaling and must be free to move any reaction anywhere; the transaction mode is the developer's explicit statement, and a mode that placement cannot support is refused, never substituted.
- **Dropping `call` in favour of defer plus continuation.** Proposed in review because a committed child under a failed parent is a saga. Decided against: calls are how a service estate behaves, the developer chooses the mode, and the runtime makes the choice safe and visible.
- **An ordinal-only idempotency key.** Would replay the wrong child after a deploy that reorders calls while a retry is pending. Replaced by the reaction-plus-input key with an explicit override.
- **A static scan of reaction source for cross-pool joins.** Fragile; the refusal happens at the trigger, at runtime, and is identical in every environment.
- **Session over the wire for joined remote children.** Correct and buildable, but every read is a network round trip inside a held serializable transaction; `call` covers the cases that want another pod.
- **Savepoints around joined children.** Decided against for now.
- **A synchronous independent child without a ledger.** Doubles its effects on retry and breaks the parent's serializability; `call` exists to make this safe.
- **Distributed and autonomous transactions.** Neither gives one transaction across pods.
- **KEDA** for scaling pools on outbox depth and in-flight calls. The natural automation once pools exist, and the scheduler's depth gauge is exactly the signal it would use; set aside for the first cut to keep the cluster small. Replicas are set per pool in the environment meanwhile.
- **Overflow between pools** when one is saturated, and **placement by data locality**. The first hides saturation; the second has nothing local to place by.
- **An operator and CRD**, or **Helm**. Generated manifests applied with prune give the same outcome for one repo without a controller to run.
- **The runtime provisioning Kubernetes at startup.** Needs write RBAC inside the app, hides drift, and cannot be reviewed before it happens.
- **Pod per run.** A pool with `concurrency=1` and long runs is pod-per-run in practice; a true Job-per-row flag waits for a reaction that proves it needs it.
- **Secret scoping per pool.** Needs per-reaction secret declarations; set aside until a pool runs less trusted code.
