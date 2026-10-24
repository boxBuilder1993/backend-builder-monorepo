# Compute Pools and Service Calls

Status: design, 2026-10-23. Decided in discussion, not built. Companion to [design.md](design.md), which it extends; where the two disagree, this document is newer. Sections 1 to 9 are the design; section 10 is the order of work; section 11 lists what was considered and set aside.

## Intent

Scale units of work independently without giving up the runtime's guarantees. Today an app runs as four process types: api, worker, scheduler, admin. Every worker claims every outbox row, so a reaction that needs a GPU and a reaction that totals an order scale together, on the same machines, with the same resources. The aim is to let the app declare **pools** of compute, say which reactions run in which pool, have a deploy create those pools in Kubernetes, and watch them in the console.

A second aim falls out of the first. Once reactions run in different pods, a reaction that triggers another is sometimes a call to another service. Microservices make such a call commit on its own and return. The runtime should let a developer ask for exactly that, as a declared expectation, with the consequences enforced, while keeping the existing modes where a child joins its parent's transaction or runs after it.

Cost is minimised by scaling each pool to what its work needs, down to zero where latency allows; availability is kept by letting the pools that serve requests declare a floor. Both are properties of the declaration, visible in the repo and the console.

## 1. The transaction decides the shape

Everything here follows from one fact of Postgres: a transaction belongs to one connection. A second connection cannot join it, see its uncommitted writes, or be rolled back with it. Savepoints are scoped to one transaction; snapshot export shares a read view, not a transaction; two-phase commit makes two transactions agree on an outcome without making them one.

Therefore an immediate sub-reaction that shares its parent's transaction, today's `ctx.trigger`, can only run in the parent's process. A sub-reaction in another pod is in another transaction, by construction, and the developer must say which transactional relationship they mean. There are exactly three, and every trigger is one of them.

## 2. Three trigger modes

```python
await ctx.trigger(Child, input)                   # join: same transaction, same session, in-process
await ctx.trigger(Child, input, mode="call")      # call: own run, own transaction, now; returns its vars
await ctx.trigger(Child, input, deferred=True)    # defer: outbox row at flush; runs after commit
```

`mode` is `"join"` (default), `"call"` or `"defer"`; `deferred=True` remains as the spelling of `mode="defer"`.

**join.** Unchanged. The child runs in the parent's session and transaction, depth-first, and returns its context. Its writes commit with the run or not at all. A joined child that raises aborts the run unless the parent catches the exception, in which case the child's writes stay in the session, as today; no savepoint is added (decided). A joined trigger may only target a reaction in the same pool; the App refuses otherwise (section 4).

**call.** The service call. The child runs now as a run of its own, through `run()`, in its own serializable transaction, in whichever pool owns it, and commits. The parent's transaction is untouched; the parent blocks until the child returns, within the parent's remaining budget. The parent receives a `CallResult`: the child's saved vars, its `reaction_id`, and `replayed` (section 3). If the child raises, the parent gets the exception, as with join. If the parent later fails or is rolled back, **the child's commit stands**. Anything that must stand or fall with the parent is join or defer. Undoing a committed call when the parent fails is a compensating reaction the developer triggers from the failure path; the guide says so and says it is a saga.

**defer.** Unchanged. An outbox row written at flush, claimed by the owning pool's worker after the parent commits, retried and dead-lettered by the outbox's rules. The parent gets no result; a result travels the event-driven way, through an entity the child writes and a trigger on it.

Entities are shared within a run and across nothing else; vars belong to one invocation; a child sees only its input. These rules from [design.md](design.md) §4 are unchanged and are what make a call's input and result serialisable: input is a JSON object, the result is the child's vars as JSON.

## 3. Call idempotency: the ledger

A run is retried from the seed on any exception, so a parent that calls a child and then fails would call it again. Service calls need an idempotency key, and the runtime has one: the parent's `reaction_id` is the same across its attempts.

- The child's run id is derived: `uuid5(parent_reaction_id, n)` where `n` is the ordinal of this call among the parent's calls in this attempt, counted in the order they are made. Two calls in one parent get different ids; the same call in a retry gets the same id.
- For a call, the run record is written **inside the child's own transaction**, not after it. The child is its own transaction, so this is possible, and it makes "the child committed" and "the ledger says the child committed" one fact. The child's record carries `source="call"`, `parent_reaction_id` and `root_id` of the caller, and its outcome.
- Before running a call, the runtime looks up the derived id in `_run`. A committed record (`committed = true`) means the child already ran: its saved vars are returned with `replayed = true` and nothing executes. A failed record, or none, means run it. A `running` record means an earlier attempt's child is still executing or died mid-way; the caller waits up to its budget for it to resolve, then treats a stale one as failed (the scheduler's lost sweep applies).

Consequences this buys: a child's commit that causes the parent's serialization failure does not run the child twice on retry; a parent that fails after a call and is retried sees the same result the first attempt saw; and a call's effects are exactly once per committed call, or zero if the child never committed. What it does not buy: undoing the child if the parent gives up for good.

The ordinal must be stable across attempts. A parent whose calls depend on non-deterministic branching would break this, exactly as it would break any retry of that parent; the existing rule, a run is a function of its input and the store, already forbids it.

## 4. Pools

```python
from backend_builder import App, Pool

pools = [
    Pool("api", min=2, max=10, cpu="500m", memory="512Mi"),
    Pool("default", min=1, max=8, concurrency=16),
    Pool("ml", reactions=[FitModel, Infer], min=0, max=4, concurrency=1,
         cpu="4", memory="8Gi", gpu=1),
]
app = App(package="finance", endpoints=endpoints, triggers=triggers, pools=pools)
```

A `Pool` declares: `name`; `reactions`, the classes it owns; `min` and `max` replicas; `concurrency`, runs in flight per replica; `cpu`, `memory`, `gpu` requests. Rules checked when the App is built:

- Every reaction belongs to exactly one pool. `default` owns every reaction not listed elsewhere and exists even if undeclared. `api` exists even if undeclared and owns no reactions; it is where endpoints run their seed.
- A joined trigger across pools is an error. The App walks each reaction's source for `ctx.trigger` calls it can resolve statically; what it cannot resolve is checked at runtime, where a cross-pool join raises `TriggerError` before the child runs. Crossing a pool boundary means crossing a transaction boundary; the developer picks call or defer.
- A pool targeted by any call has `min >= 1`. A call runs inside the caller's held transaction and budget; a cold start there is a cost the caller did not choose. A pool that only receives deferred rows may have `min = 0`.
- `concurrency` is per replica; a GPU pool is typically 1.

What a pool deliberately does not declare: image tag, database URL, secrets, namespace, node labels, environment-specific overrides. Those are the environment's (section 6).

Pools are visible wherever the app describes itself: `backend-builder graph` prints them with their reactions; the console shows them (section 8); `describe()` includes them.

## 5. Runtime changes

**Worker.** `backend-builder worker <app> --pool <name>` claims only rows whose reaction belongs to that pool. `claim` gains a `reactions` filter, `reaction = any(:names)` inside the same `FOR UPDATE SKIP LOCKED` statement, so pools never contend for rows. A worker started for a pool the app does not declare refuses to start. Without `--pool`, a worker claims for every pool, which is what compose and single-host setups keep using.

**Scheduler.** Unchanged, one replica. Seeded rows are claimed by the owning pool like any other.

**Records.** `_run` gains `pool`, the pool the invocation ran in, on every row. `source` gains `"call"`. A call's record is written in-transaction (section 3); every other record is written after the run, as now.

**Metrics.** `pool` label on `bb_runs_total`, `bb_run_duration_seconds` and `bb_outbox_deliveries_total`; new gauges `bb_outbox_pending{pool}` and `bb_runs_in_flight{pool}`; new counter `bb_calls_total{reaction, outcome}` with outcomes `ok`, `error`, `replayed`, `unreachable`.

**Graceful shutdown.** On `SIGTERM` a worker stops claiming, lets in-flight runs finish within the pod's grace period, and releases what it cannot finish. The manifests set `terminationGracePeriodSeconds` above the pool's longest trigger timeout. A pod killed harder loses only its in-flight runs, which roll back and are redelivered when their leases lapse, as today.

**Secrets.** A file provider in front of the environment provider: `ctx.secret(name)` reads `/run/secrets/backend-builder/SECRET_<NAME>` if the file exists, else the variable. Reading stays on access, so a rotated mounted Secret is picked up without a restart. Compose, tests and host setups keep using variables and notice nothing. Redaction is unchanged.

## 6. Remote calls

A call whose child lives in another pool is a request to that pool. Each pool's worker process also serves, on its metrics port, `POST /calls`: body `{"run_id", "reaction", "input", "root_id", "parent_reaction_id", "deadline"}`, response the child's vars and outcome. The worker executes it through the same `run()` as an outbox delivery, with the ledger check first, counting it against its `concurrency`. The caller connects to the pool's Kubernetes Service, so load balancing is the cluster's, and passes its remaining budget as the deadline; the callee refuses work whose deadline has passed.

Failure mapping: a non-2xx or a dropped connection is an exception in the caller, so the run fails or retries like any other error, and the retry consults the ledger, so a child that did commit is replayed rather than re-run. A caller that times out while the child is still running leaves a `running` ledger row that resolves when the child finishes or is swept as lost.

In-process calls, when caller and callee share a pool, skip the network and execute `run()` directly on a second connection. Semantics are identical; compose and tests exercise exactly the same ledger and replay paths without a cluster.

Joined children never cross the network: there is no session-over-the-wire protocol. This was considered and set aside (section 11).

## 7. Manifests and deploy

`backend-builder manifests <app> --env <name>` renders Kubernetes YAML from the App's declaration merged with `deploy/<name>.yaml`, the environment file: image and tag, namespace, node selectors and tolerations per pool (for example which node group has GPUs), replica overrides, database URL source, the secret mount. Rendered objects, all labelled `app=<name>`:

- one Deployment per pool running `worker <app> --pool <name>`, with the pool's resources, `nvidia.com/gpu` when `gpu` is set, `replicas: min`, the secret volume, and a Service for pools that receive calls;
- the `api` Deployment running `serve`, its Service, and an HPA on CPU between `min` and `max`;
- one scheduler Deployment, one admin Deployment with its Service;
- a migrate Job, applied first and waited on before anything else rolls;
- a PodDisruptionBudget for every pool with `min >= 2`;
- PgBouncer in transaction pooling mode in front of Postgres, since connections grow with pods and the runtime's transactions are short and per run.

Scaling between `min` and `max` for worker pools is not automated in the first cut (KEDA was set aside, section 11): `replicas` is `min`, and the environment file may override it per pool. The console shows pending rows per pool so the number can be chosen with evidence; automation follows when there is a case for it.

Deploy is `kubectl apply --prune -l app=<name>` of the rendered YAML, run by the pipeline after the image is pushed. A pool added in code appears on the next deploy; a pool removed is deleted. The YAML is committed, so a PR that adds a pool shows the Deployment it creates, the same way a PR that adds a field shows its migration. The runtime never talks to the Kubernetes API except the console's read-only view (section 8); nothing provisions itself at startup.

## 8. Console

A pools page: for each pool, its reactions, desired and ready replicas, pending outbox rows, runs in flight, runs in the last 24 hours by outcome, p95 duration, and calls received with their outcomes. Replica counts come from the Kubernetes API with read-only RBAC on Deployments in the app's namespace, and are shown only when the console runs in-cluster; everything else comes from `_run` and `_outbox`. The run page shows the pool on every invocation row; a call child appears in its caller's tree as its own run, linked, marked `replayed` when a retry served it from the ledger. The console never writes to Kubernetes.

## 9. Local cluster

`make run-all` brings the whole system up on this machine, every process a pod:

1. **minikube** with the Docker driver, started with `--gpus=all` and the `nvidia-device-plugin` addon, so a pool with `gpu=1` gets the machine's GPU. kind cannot pass a GPU through and is not used. `make cluster` checks for minikube, kubectl and the NVIDIA container toolkit and prints the install commands when missing; it does not install system packages.
2. Build the app image and load it into the cluster; there is no local registry.
3. Postgres as a StatefulSet in the cluster, one database per app, the same init script compose uses.
4. **Secrets from the repo's `.env`**, which is gitignored. A committed `.env.example` lists every `SECRET_*` key; `make run-all` stops with the missing names if `.env` lacks any. The `SECRET_*` lines become one Kubernetes Secret via dry-run and apply, never written to YAML or git, mounted as files in every pool's pods. Editing `.env` and running `make secrets` rotates values live. The Makefile refuses this path when the kube context is not minikube, so a laptop `.env` cannot reach another cluster by habit. In a real cluster the same mount is filled by an external secrets operator from a vault; the pods cannot tell.
5. Render and apply the manifests for `--env local`; the migrate Job first.
6. Port-forward api and admin and print the URLs.

Re-running `make run-all` is a rolling redeploy. `make cluster-down` deletes the cluster. The existing host and systemd targets stay as they are. A `make e2e-k8s` target runs the end-to-end suite against the cluster instead of compose; compose remains the fast loop.

One physical GPU means `max=1` for the GPU pool locally, which the local environment file sets. The GTX 1080 is Pascal (compute capability 6.1): current CUDA 12, XGBoost and LightGBM GPU builds support it; recent PyTorch builds may not, in which case the fitting code's `device="auto"` falls back to CPU. The GPU pool's image needs the CUDA user-space libraries, which argues for a per-pool image override in the environment file; the first cut uses one image.

## 10. Order of work

1. **Runtime: pools and calls.** `Pool`, validation on `App`, `--pool` on the worker, claim-by-pool, `pool` on records and metrics, graceful shutdown; `mode="call"` in-process with the ledger and replay; the file secrets provider. Shop and finance declare pools; finance's fits become `call` children where the sweep needs their result, deferred where it does not. Migrations for both chains. One PR.
2. **Cluster.** `manifests` command with the environment file, `deploy/local.yaml`, `make cluster`, `make run-all`, `make secrets`, `make cluster-down`, `make e2e-k8s`, the minikube and GPU checks, PgBouncer. One PR.
3. **Remote calls.** `POST /calls` on the worker, the client in `ctx.trigger`, `bb_calls_total`, the cross-pool join refusal at runtime. One PR.
4. **Console pools page**, with the Kubernetes read. One PR, or folded into 2.

Each step leaves compose and the single-host setup working unchanged.

## 11. Considered and set aside

- **Session over the wire for joined remote children.** The parent keeps the transaction and the child's store calls travel back to it. Correct and buildable, but every read is a network round trip inside a held serializable transaction, and the `call` mode covers the cases that want another pod. Revisit only if a reaction needs the parent's uncommitted state and cannot run in its pool.
- **Savepoints around joined children.** A joined child that raises and is caught would leave nothing behind. Decided against for now; a caught child's writes stay in the session, as today.
- **A synchronous independent child without a ledger.** It would double its effects on every retry and break the parent's serializability. `call` exists precisely to make this safe.
- **Distributed transactions** (two-phase commit) and **autonomous transactions**. Postgres offers the first and not the second; neither gives one transaction across pods.
- **KEDA** for scale-to-zero on outbox depth. The right automation later; not needed to get pools running, and one less component in the first cluster.
- **An operator and CRD**, or **Helm**. Generated manifests applied with prune give the same outcome for one repo and one cluster without a controller to run.
- **The runtime provisioning Kubernetes at startup.** Needs write RBAC inside the app, hides drift, and cannot be reviewed before it happens.
- **Pod per run**, a Job per outbox row, for true per-run isolation. Right for a reaction that could eat a node; wrong for anything fast. A per-pool flag later, if a reaction proves it needs it.
- **Secret scoping per pool**, giving each pool only the keys its reactions read. Needs per-reaction secret declarations; set aside until a pool runs less trusted code.
- **Session-shared vars across a call.** A call's input and result are JSON by necessity; a joined child's vars are already its own. Nothing to share.
