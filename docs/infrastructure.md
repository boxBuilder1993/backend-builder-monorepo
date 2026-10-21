# Repository and Infrastructure Specification

Status: draft, 2026-10-20, with a scope note added 2026-10-21. Companion to [design.md](design.md), which specifies the runtime and is the authority on what is built and decided. This document specifies the repository that holds it, how it is built, how it runs in each environment, and how it is deployed. Everything in this document runs on a developer machine. Cloud deployment is deliberately out of scope until the system is verified locally.

**Scope note, 2026-10-21.** The runtime design has moved on from the draft this document was written against. The following parts of this document are parked until the design doc's "Later" section brings them back: the WAL worker and `backend-builder wal`, the logical replication settings (`wal_level`, replication slots, `max_slot_wal_keep_size`), the `wal-worker` workload and its disruption budget, the WAL fixture and `@pytest.mark.wal`, and the slot-lag metric. The deferred worker and the outbox remain planned. Two things changed for the test strategy: the test database comes from the `pgserver` package, which bundles Postgres and starts it from `pip install` with no Docker and no sudo, so the suite and CI need no service container; and the per-test database is created from a template as described in section 7. Compose still provides Postgres for running the service locally. The `backend_builder.toml` app manifest is not needed; the `App` object is the manifest. Since then, the `Dockerfile`, the compose `shop` profile (migrate, api, worker, scheduler) and the end-to-end suite that runs against it exist; design.md section 17 describes them. The `scheduler` is its own process, not part of the worker. Everything else here stands.

## 1. Goals

- One repository holds the core library, the use-case applications built on it, and everything needed to run them: container image, Kubernetes manifests, CI.
- The core library and the use-case applications are separate packages. The core has no knowledge of any application. Applications depend on the core and on nothing else in the repo.
- Three runtime environments, `test`, `dev` and `prod`, selected by one variable. The same image and the same manifests run in all three. Only configuration differs.
- Kubernetes is the deployment target from the start. The repository defines every workload that runs in the cluster, and the cluster is a local `kind` cluster until further notice.
- A developer or an agent can go from a clean checkout to a running system with one command per environment, and verify every piece of it on this machine.

## 2. Repository layout

```
backend-builder/
  pyproject.toml              # uv workspace root; no code
  uv.lock
  justfile                    # task runner; only the recipes that exist today
  compose.yaml                # local Postgres, profiles per environment
  Dockerfile                  # one image, all processes
  .github/workflows/ci.yaml
  docs/
    design.md
    infrastructure.md
  packages/
    backend_builder/          # the core library and its CLI
    backend_builder_testing/  # pytest plugin: fixtures and harness for apps
  apps/
    shop/                     # first use-case app, doubles as the integration suite
  deploy/
    k8s/
      base/                   # workloads shared by every environment
      overlays/
        dev/
        prod/
  config/
    base.toml
    test.toml
    dev.toml
    prod.toml
```

### 2.1 uv workspace

The root `pyproject.toml` declares a workspace with members `packages/*` and `apps/*`. Each member is its own distribution with its own `pyproject.toml`, dependencies and tests. One `uv.lock` at the root pins every member consistently. `uv sync` at the root installs all members in one virtual environment; `uv sync --package shop` installs one member and its dependencies.

| Package | Distribution | Import name | Depends on | Purpose |
| --- | --- | --- | --- | --- |
| `packages/backend_builder` | `backend-builder` | `backend_builder` | SQLAlchemy, asyncpg, Alembic, Starlette, uvicorn, typer, pydantic | The runtime: entities, context, triggers, reconcile, outbox, workers, HTTP, and the `backend-builder` CLI |
| `packages/backend_builder_testing` | `backend-builder-testing` | `backend_builder_testing` | `backend-builder`, pytest, pytest-asyncio | Fixtures: database per test, in-process worker drains, WAL slot fixture, app loading |
| `apps/shop` | `shop` | `shop` | `backend-builder`, `backend-builder-testing` (dev) | Order-to-cash use case: entities, config, reactions, and the integration tests that prove the runtime |

Rules:

- The core never imports from `apps/`. A lint rule enforces it.
- Each app is a complete deployable: it has its own `entities/`, `reactions/`, `consumers.py`, and a `backend_builder.toml` naming its entry module. The CLI runs any app by name.
- Unit tests live next to the code in each package. Integration tests live in the app that exercises them. The core's own tests use a minimal fixture schema, not the shop.
- Every member uses the same tool configuration, inherited from the root: ruff for lint and format, pyright in strict mode, pytest with asyncio mode auto.

### 2.2 The `backend-builder` CLI

The core library ships one console script, `backend-builder`. It is the process launcher for every runtime process and the operator's tool. The container image's entrypoint is this command. The first argument is a subcommand; most take an app name.

| Command | Runs |
| --- | --- |
| `backend-builder serve <app>` | the HTTP API process |
| `backend-builder worker <app>` | the deferred worker |
| `backend-builder wal <app>` | the WAL worker, including the scheduler |
| `backend-builder migrate <app>` | generate and apply schema migrations |
| `backend-builder trigger <app> <reaction> --input <json>` | enqueue one reaction with the given input |
| `backend-builder backfill <app> <reaction> --entity <Entity> --sql <select>` | enqueue synthetic changes for matching rows |
| `backend-builder graph <app>` | print the consumer config as a dependency map |

Subcommands are added as the runtime milestones that need them land. The list above is the full v1 surface.

### 2.3 The task runner

A `justfile` at the root is the single interface for developers, agents and CI. It contains only recipes that exist and are used. New recipes are added when a task needs them, not in anticipation.

Initial recipes:

| Recipe | Does |
| --- | --- |
| `just setup` | `uv sync` and install pre-commit hooks |
| `just db` | start Postgres from compose for the current environment |
| `just test [package]` | run tests, all or one member, against a fresh test database |
| `just lint` | ruff check and format check, pyright, import-boundary check |

## 3. Environments and configuration

### 3.1 Environment selection

`RUNTIME_ENV` is one of `test`, `dev`, `prod`. It is the only variable that selects behaviour. Unset means `dev` on a developer machine and is an error inside a container.

| | `test` | `dev` | `prod` |
| --- | --- | --- | --- |
| Where | developer machine, CI | developer machine, `kind` | `kind` for now; a real cluster later |
| Database | fresh per test from a template, dropped after | persistent compose Postgres, or the in-cluster StatefulSet | in-cluster StatefulSet for now; managed Postgres later |
| Workers | in-process, drained explicitly by tests | separate processes or pods | separate pods |
| Logging | captured by pytest | human-readable to stdout | JSON to stdout |
| Secrets | `SECRET_*` set by the test | `SECRET_*` from compose or a gitignored `.env` | `SECRET_*` injected from Kubernetes Secret objects |
| Reload | n/a | on | off |

The `prod` environment exists from the start so that its configuration, manifests and behaviour are exercised locally. Nothing about it assumes a cloud.

### 3.2 Configuration layering

Settings are a typed model loaded in this order, each layer overriding the last:

1. `config/base.toml`: every setting with its default.
2. `config/<RUNTIME_ENV>.toml`: environment overrides, committed.
3. Environment variables prefixed `BACKEND_BUILDER_`, mapped by name: `BACKEND_BUILDER_DATABASE_URL`, `BACKEND_BUILDER_OUTBOX_POLL_SECONDS`.
4. Kubernetes injects layer 3 from a ConfigMap for non-secret values and a Secret for the rest.

Secrets are not settings. They are `SECRET_<NAME>` environment variables read by reactions through `ctx.secret(name)` (design doc §22), injected by the platform and never written to a TOML file.

The settings model is the single source of truth for what is configurable. It is defined once in the core and extended by apps for their own settings.

### 3.3 Database per environment

The runtime needs `wal_level = logical`, `max_replication_slots >= 2`, `max_wal_senders >= 2`, and `max_slot_wal_keep_size` set. Every environment must provide these.

- `test` and `dev` on the developer machine: the compose Postgres is started with these as command-line flags, so no configuration file is managed.
- `dev` and `prod` in `kind`: a Postgres StatefulSet in the namespace, same flags, a persistent volume. Disposable.

## 4. Containers

### 4.1 One image

A single multi-stage `Dockerfile` builds one image per git SHA. The entrypoint is the `backend-builder` CLI; the process is selected by subcommand. One image for every process means one build and one tag to promote.

Stages:

1. `base`: Python 3.13 slim, `uv` installed, non-root user.
2. `deps`: `uv sync --frozen --no-dev` against the lock file only, cached as its own layer.
3. `app`: copy the workspace, `uv sync --frozen --no-dev` again to install the members. `RUNTIME_ENV` is unset so a missing value fails fast.
4. `dev`: from `app`, add dev dependencies and reload tooling. Used by compose in `dev`, never deployed.

Images are tagged `backend-builder:<git-sha>`. Nothing is ever tagged `latest`.

### 4.2 Compose for local environments

`compose.yaml` uses profiles so one file serves both local environments:

- `--profile test`: Postgres only, data on tmpfs for speed. Tests connect to it and manage their own databases.
- `--profile dev`: Postgres on a named volume, plus the API with reload, the deferred worker and the WAL worker from the `dev` image stage with the repo bind-mounted.

Postgres runs with the required flags in its `command`.

## 5. Kubernetes

### 5.1 Workloads

Everything the runtime needs in a cluster is defined under `deploy/k8s/base` and is the same in every environment. Overlays change only replicas, resources, image tag, hostnames and config values.

| Resource | Kind | Notes |
| --- | --- | --- |
| `api` | Deployment | `backend-builder serve <app>`. Horizontally scaled. Readiness probe on a health endpoint that checks the database. |
| `deferred-worker` | Deployment | `backend-builder worker <app>`. Horizontally scaled. Each replica claims its own outbox rows. |
| `wal-worker` | Deployment, 1 replica, `Recreate` strategy | `backend-builder wal <app>`. Exactly one consumer of the replication slot. Includes the scheduler. A PodDisruptionBudget keeps it from being evicted during node drains. |
| `migrate` | Job | `backend-builder migrate <app>`. Runs before any rollout; the rollout waits for it. |
| `postgres` | StatefulSet, Service | Postgres with the logical replication flags and a persistent volume. Present in both overlays for now. |
| `api` | Service and Ingress | TLS at the ingress in `prod`; plain in `dev`. |
| `config` | ConfigMap | Non-secret settings for the environment. |
| `secrets` | Secret | Created out of band for `kind`. Never committed. |
| `api-hpa`, `worker-hpa` | HorizontalPodAutoscaler | CPU for the API; outbox depth, exported as a metric, for the worker. |

Pods run as non-root with a read-only root filesystem. Resource requests and limits are set in every overlay.

### 5.2 Overlays

Kustomize was chosen over Helm because the manifests are ours, there is one application shape, and overlays are plain YAML an agent can read and diff.

- `overlays/dev`: one replica each, small resources, image tag from the local build, hostname under a local domain.
- `overlays/prod`: replicas and resources sized for load, image tag pinned to a SHA, production-shaped config. Runs in `kind` today with the in-cluster Postgres.

Rendering is `kubectl kustomize deploy/k8s/overlays/<env>`. Applying is `kubectl apply -k` against a kube context whose name matches the environment; a wrapper recipe will enforce the match when it is added.

### 5.3 Rollout order

1. Build the image for the SHA and load it into `kind`.
2. Apply the `migrate` Job with the new image. Wait for completion.
3. Apply everything else. The API and deferred worker roll with zero downtime. The WAL worker restarts with a brief gap; the slot retains WAL during it, so nothing is lost.

Migrations must therefore be backward compatible with the previous image for the duration of a rollout: add columns before code uses them, drop columns one release after code stops.

### 5.4 Local cluster

`kind` runs a cluster on the developer machine. Creating it, loading the local image, applying an overlay and port-forwarding the API are the steps by which manifest changes are tested, and by which an agent verifies a deployment task. Recipes for these are added to the `justfile` when the Kubernetes tasks begin.

## 6. Continuous integration

GitHub Actions, one workflow, `ci.yaml`, on every pull request:

1. `just lint`.
2. `just test` against a Postgres service container started with the logical replication flags. Every test, including WAL worker tests, runs here.
3. Build the image to prove it builds. Not pushed anywhere.
4. Render both Kubernetes overlays and validate them with `kubeconform`.

A pull request cannot merge red. This is the first reviewer of every agent's work. No deployment workflows exist; deployment is a local operation against `kind`.

## 7. Testing infrastructure

Specified here because it is shared by every package and is infrastructure in its own right. Lives in `packages/backend_builder_testing`.

- **Database per test.** A session fixture creates a template database with the app's schema applied once. A function fixture creates a database from the template, yields a connection URL, and drops it. Creation from a template is milliseconds. No transaction-rollback isolation, because flush ordering and serialization conflicts must be tested for real.
- **In-process workers.** `drain_outbox()` and `process_wal_once()` are functions. A test triggers, commits, drains, asserts. No background processes.
- **WAL fixture.** Creates a publication and a replication slot on the per-test database, yields, drops both. Slots are per database so isolation holds.
- **Two-connection fixture** for concurrency tests that force a serialization failure.
- **App fixture.** Loads an app's entities and config into a context factory so integration tests start from a real schema and optional seed rows.
- **Marks.** `@pytest.mark.wal` for tests needing a slot, `@pytest.mark.slow` for concurrency tests. CI runs everything; `just test` runs everything; a developer can deselect.

## 8. Observability

In scope for the infrastructure even though the runtime's tracing is a later milestone.

- Structured JSON logs in `prod`, with a correlation id per run that the HTTP layer returns in error responses.
- Metrics endpoint on every process: request latency, run duration, reconcile levels per run, outbox depth, outbox age of oldest row, WAL slot lag in bytes, deferred attempts and failures. Outbox depth drives the worker autoscaler.
- Health endpoints: liveness is process-up; readiness checks the database and, for the WAL worker, that the slot is active.

## 9. Infrastructure task list

In dependency order. Each is a unit an agent can take with a definition of done. All of them run on this machine.

| # | Task | Done when |
| --- | --- | --- |
| 1 | Root workspace: `pyproject.toml`, `uv.lock`, three members with empty packages, shared tool config, `justfile` with `setup`, `lint`, `test` | `just setup && just lint && just test` pass on an empty suite |
| 2 | Import boundary lint: core cannot import apps | a deliberate violation fails `just lint` |
| 3 | `compose.yaml` with `test` and `dev` profiles, Postgres with logical replication flags, `just db` | `just db` starts Postgres; `psql` shows `wal_level = logical` |
| 4 | Settings model with layering and `RUNTIME_ENV` | a test proves each layer overrides the last; `prod` refuses default secrets |
| 5 | Testing package: template database, per-test database, two-connection fixture | a test creates, uses and drops a database in under 100 ms |
| 6 | Testing package: WAL fixture | a test commits a row and reads it from the slot |
| 7 | `Dockerfile` with the four stages, SHA tagging | image runs `backend-builder --help`; `dev` stage reloads |
| 8 | Compose `dev` profile runs API, worker and WAL worker from the image | the API responds on localhost |
| 9 | `ci.yaml` with lint, test against a Postgres service, build, render | a pull request shows all four checks |
| 10 | Kubernetes base manifests and `dev` overlay with in-cluster Postgres | the rendered manifests pass `kubeconform` |
| 11 | `kind` recipes: create, load image, apply, port-forward | the API responds through the port-forward |
| 12 | `prod` overlay | renders, passes `kubeconform`, and runs in `kind` |
| 13 | Metrics and health endpoints, worker HPA on outbox depth | metrics visible in the cluster; the worker scales with outbox depth |

Tasks 1 through 9 need Docker. Tasks 10 through 13 also need `kind` and `kubectl`.

## 10. Prerequisites outside the repository

- Docker Engine on the developer machine. The only system-level install; everything else is pinned in the repo.
- `kind` and `kubectl`, installed by `just setup` into the user's path if absent.

## 11. Decisions taken in this draft

- **No cloud until local verification is complete.** Terraform, managed databases, registries and deployment pipelines are deferred entirely. Nothing here forecloses them.
- **Kustomize over Helm**: our manifests, one shape, plain YAML for agents to read.
- **One image for all processes**: one build and one promotion path.
- **`just` over `make`**: readable recipes, no tab semantics.
- **GitHub Actions for CI only**: lint, test, build, render. No deploy jobs.
- **Fully spelled names everywhere**: `RUNTIME_ENV`, `BACKEND_BUILDER_` as the settings prefix, `backend-builder` as the CLI, `backend_builder.toml` as the app manifest.
