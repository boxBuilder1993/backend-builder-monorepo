# Repository and Infrastructure Specification

Status: draft for review, 2026-10-20. Companion to [design.md](design.md), which specifies the runtime. This document specifies the repository that holds it, how it is built, how it runs in each environment, and how it is deployed.

## 1. Goals

- One repository holds the core library, the use-case applications built on it, and everything needed to run them: container images, Kubernetes manifests, cloud infrastructure, CI.
- The core library and the use-case applications are separate packages. The core has no knowledge of any application. Applications depend on the core and on nothing else in the repo.
- Three runtime environments, `test`, `dev` and `prod`, selected by one variable. The same image and the same manifests run in all three. Only configuration differs.
- Kubernetes is the deployment target from the start. The repository defines every workload that runs in the cluster.
- A developer or an agent can go from a clean checkout to a running system with one command per environment, and from a change to a deployed change with one pull request.

## 2. Repository layout

```
backend-builder/
  pyproject.toml              # uv workspace root; no code
  uv.lock
  justfile                    # task runner; every task agents and CI use
  compose.yaml                # local services, profiles per environment
  Dockerfile                  # one image, all processes
  .github/workflows/
  docs/
    design.md
    infrastructure.md
  packages/
    backend_builder/          # the core library
    backend_builder_testing/  # pytest plugin: fixtures and harness for apps
  apps/
    shop/                     # first use-case app, doubles as the integration suite
  deploy/
    k8s/
      base/                   # workloads shared by every environment
      overlays/
        dev/
        prod/
    terraform/
      modules/
      envs/
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
| `packages/backend_builder` | `backend-builder` | `backend_builder` | SQLAlchemy, asyncpg, Alembic, Starlette, uvicorn, typer, pydantic | The runtime: entities, context, triggers, reconcile, outbox, workers, HTTP, CLI |
| `packages/backend_builder_testing` | `backend-builder-testing` | `backend_builder_testing` | `backend-builder`, pytest, pytest-asyncio | Fixtures: database per test, in-process worker drains, WAL slot fixture, example seeding |
| `apps/shop` | `shop` | `shop` | `backend-builder`, `backend-builder-testing` (dev) | Order-to-cash use case: entities, config, reactions, and the integration tests that prove the runtime |

Rules:

- The core never imports from `apps/`. A lint rule enforces it.
- Each app is a complete deployable: it has its own `entities/`, `reactions/`, `consumers.py`, and a `bb.toml` naming its entry module. The CLI runs any app by name.
- Unit tests live next to the code in each package. Integration tests live in the app that exercises them. The core's own tests use a minimal fixture schema, not the shop.
- Every member uses the same tool configuration, inherited from the root: ruff for lint and format, pyright in strict mode, pytest with asyncio mode auto.

### 2.2 The task runner

A `justfile` at the root is the single interface. CI calls these recipes. Agents call these recipes. Nothing is documented as a raw command.

| Recipe | Does |
| --- | --- |
| `just setup` | `uv sync`, install pre-commit hooks, pull images |
| `just db` | start Postgres from compose for the current environment |
| `just migrate [app]` | generate and apply migrations for an app |
| `just test [package]` | run tests, all or one member, against a fresh test database |
| `just lint` | ruff check and format check, pyright, import-boundary check |
| `just serve [app]` | API process with reload, `dev` environment |
| `just worker [app]` | deferred worker, `dev` environment |
| `just wal [app]` | WAL worker, `dev` environment |
| `just build` | build the image, tag with git SHA |
| `just k8s-render <env>` | render manifests for an environment with kustomize, no apply |
| `just k8s-apply <env>` | apply manifests to the current kube context, guarded by env confirmation |
| `just kind-up` / `just kind-down` | local cluster for manifest testing |
| `just tf <env> <plan|apply>` | Terraform for an environment |

## 3. Environments and configuration

### 3.1 Environment selection

`BB_ENV` is one of `test`, `dev`, `prod`. It is the only variable that selects behaviour. Unset means `dev` locally and is an error in a container.

| | `test` | `dev` | `prod` |
| --- | --- | --- | --- |
| Where | developer machine, CI | developer machine, dev cluster | prod cluster |
| Database | fresh per test from a template, dropped after | persistent local or dev-cluster Postgres | managed Postgres, external to the cluster |
| Workers | in-process, drained explicitly by tests | separate processes or pods | separate pods, autoscaled |
| Logging | captured by pytest | human-readable to stdout | JSON to stdout |
| Secrets | fixed values in `test.toml` | `.env` file, gitignored | Kubernetes secrets from an external store |
| Reload | n/a | on | off |
| Serializable retry | on, low count | on | on |

### 3.2 Configuration layering

Settings are a typed model loaded in this order, each layer overriding the last:

1. `config/base.toml`: every setting with its default.
2. `config/<BB_ENV>.toml`: environment overrides, committed.
3. Environment variables prefixed `BB_`, mapped by name: `BB_DATABASE_URL`, `BB_OUTBOX_POLL_SECONDS`.
4. Kubernetes injects layer 3 from a ConfigMap for non-secret values and a Secret for the rest.

Secrets never appear in a TOML file except `test.toml`. The settings model refuses to start in `prod` if a secret has its default value.

The settings model is the single source of truth for what is configurable. It is defined once in the core and extended by apps for their own settings.

### 3.3 Database per environment

The runtime needs `wal_level = logical`, `max_replication_slots >= 2`, `max_wal_senders >= 2`, and `max_slot_wal_keep_size` set. Every environment must provide these.

- `test` and `dev` local: the compose Postgres is started with these as command-line flags, so no configuration file is managed.
- `dev` cluster: a Postgres StatefulSet in the namespace, same flags, a persistent volume, no backups. Disposable by design.
- `prod`: Amazon RDS for PostgreSQL with a parameter group setting `rds.logical_replication = 1` and the slot limits. Provisioned by Terraform. The cluster reaches it through a private subnet.

## 4. Containers

### 4.1 One image

A single multi-stage `Dockerfile` builds one image per git SHA. The entrypoint is the `bb` CLI; the process is selected by command: `bb serve shop`, `bb worker shop`, `bb wal shop`, `bb migrate shop`. One image for every process means one build, one scan, one tag to promote.

Stages:

1. `base`: Python 3.13 slim, `uv` installed, non-root user.
2. `deps`: `uv sync --frozen --no-dev` against the lock file only, cached as its own layer.
3. `app`: copy the workspace, `uv sync --frozen --no-dev` again to install the members, `BB_ENV` unset so a missing value fails fast.
4. `dev`: from `app`, add dev dependencies and reload tooling. Used by compose in `dev`, never deployed.

Images are tagged `<registry>/backend-builder:<git-sha>` and additionally `:dev` or `:prod` on promotion. Nothing is ever tagged `latest`.

### 4.2 Compose for local environments

`compose.yaml` uses profiles so one file serves both local environments:

- `--profile test`: Postgres only. Tests connect to it and manage their own databases.
- `--profile dev`: Postgres, the API with reload, the deferred worker, the WAL worker, all from the `dev` image stage, with the repo bind-mounted.

Postgres runs with the required flags in its `command`. A named volume holds `dev` data; `test` uses tmpfs for speed.

## 5. Kubernetes

### 5.1 Workloads

Everything the runtime needs in a cluster is defined under `deploy/k8s/base` and is the same in every environment. Overlays change only replicas, resources, image tag, hostnames and config values.

| Resource | Kind | Notes |
| --- | --- | --- |
| `api` | Deployment | `bb serve <app>`. Horizontally scaled. Readiness probe on a health endpoint that checks the database. |
| `deferred-worker` | Deployment | `bb worker <app>`. Horizontally scaled. Each replica claims its own outbox rows. |
| `wal-worker` | Deployment, 1 replica, `Recreate` strategy | `bb wal <app>`. Exactly one consumer of the replication slot. Includes the scheduler. A PodDisruptionBudget keeps it from being evicted during node drains. |
| `migrate` | Job | `bb migrate <app>`. Runs before any rollout; the rollout waits for it. |
| `api` | Service and Ingress | TLS at the ingress. |
| `config` | ConfigMap | Non-secret settings for the environment. |
| `secrets` | ExternalSecret | Synced from AWS Secrets Manager by the External Secrets operator. Never committed. |
| `api-hpa`, `worker-hpa` | HorizontalPodAutoscaler | CPU for the API; outbox depth, exported as a metric, for the worker. |

Pods run as non-root with a read-only root filesystem. Resource requests and limits are set in every overlay.

### 5.2 Overlays

Kustomize was chosen over Helm because the manifests are ours, there is one application shape, and overlays are plain YAML an agent can read and diff. Helm's templating earns its complexity when packaging for third parties, which is not the case here.

- `overlays/dev`: one replica each, small resources, the in-cluster Postgres StatefulSet, image tag from the latest `main` build, hostname under a dev domain.
- `overlays/prod`: replicas and resources sized for load, no Postgres in the cluster, image tag pinned to a promoted SHA, production hostname, External Secrets enabled.

`just k8s-render <env>` produces the full manifest set for review. `just k8s-apply <env>` applies it and refuses if the current kube context does not match the environment's expected cluster name.

### 5.3 Rollout order

1. Build and push the image for the SHA.
2. Apply the `migrate` Job with the new image. Wait for completion.
3. Apply everything else. The API and deferred worker roll with zero downtime. The WAL worker restarts with a brief gap; the slot retains WAL during it, so nothing is lost.

Migrations must therefore be backward compatible with the previous image for the duration of a rollout: add columns before code uses them, drop columns one release after code stops.

### 5.4 Local cluster

`kind` runs a cluster on the developer machine for testing manifests. `just kind-up` creates it, loads the local image, applies the `dev` overlay with the in-cluster Postgres, and port-forwards the API. This is how changes to `deploy/k8s` are tested before CI, and how agents verify manifest changes.

## 6. Cloud infrastructure

Terraform under `deploy/terraform` provisions everything outside the cluster for each environment. Modules are shared; `envs/<env>` composes them with environment-specific values and holds its own state in an S3 backend with DynamoDB locking.

| Module | Provisions |
| --- | --- |
| `network` | VPC, public and private subnets across three zones, NAT |
| `eks` | the cluster, a managed node group, the OIDC provider for IRSA, the External Secrets and metrics-server addons |
| `rds` | Postgres with the logical replication parameter group, in private subnets, automated backups, a security group admitting only the cluster |
| `ecr` | the image repository, lifecycle policy keeping the last N images |
| `secrets` | Secrets Manager entries for the database URL and anything else the app declares |
| `iam` | the roles pods assume through IRSA, and the deploy role CI assumes through OIDC |

`dev` provisions a small single-node cluster and a small RDS instance. `prod` provisions multi-zone nodes and a Multi-AZ RDS. The modules are the same.

Terraform is applied by a human or by a CI job that requires approval. Agents may run `just tf <env> plan` freely and never `apply`.

## 7. Continuous integration and delivery

GitHub Actions, one workflow file per concern.

**`ci.yaml`, on every pull request:**

1. `just lint`.
2. `just test` against a Postgres service container started with the logical replication flags. Every test, including WAL worker tests, runs here.
3. `just build` to prove the image builds. Not pushed.
4. `just k8s-render dev` and `just k8s-render prod` to prove the manifests render, with `kubeconform` validating them.

A pull request cannot merge red. This is the first reviewer of every agent's work.

**`deploy-dev.yaml`, on merge to `main`:**

1. Build and push the image tagged with the SHA.
2. Assume the deploy role through OIDC.
3. Apply the `dev` overlay with the new tag, migrate Job first.
4. Run a smoke test against the dev API.

**`deploy-prod.yaml`, manual dispatch with a SHA:**

1. Require that the SHA was deployed to `dev` and its smoke test passed.
2. Require a reviewer approval through a GitHub environment protection rule.
3. Tag the image `prod`, apply the `prod` overlay, migrate first.

## 8. Testing infrastructure

Specified here because it is shared by every package and is infrastructure in its own right. Lives in `packages/backend_builder_testing`.

- **Database per test.** A session fixture creates a template database with the app's schema applied once. A function fixture creates a database from the template, yields a connection URL, and drops it. Creation from a template is milliseconds. No transaction-rollback isolation, because flush ordering and serialization conflicts must be tested for real.
- **In-process workers.** `drain_outbox()` and `process_wal_once()` are functions. A test triggers, commits, drains, asserts. No background processes.
- **WAL fixture.** Creates a publication and a replication slot on the per-test database, yields, drops both. Slots are per database so isolation holds.
- **Two-connection fixture** for concurrency tests that force a serialization failure.
- **App fixture.** Loads an app's entities and config into a context factory so integration tests start from a real schema and optional seed rows.
- **Marks.** `@pytest.mark.wal` for tests needing a slot, `@pytest.mark.slow` for concurrency tests. CI runs everything; `just test` runs everything; a developer can deselect.

## 9. Observability

In scope for the infrastructure even though the runtime's tracing is a later milestone.

- Structured JSON logs in `prod`, with a correlation id per run that the HTTP layer returns in error responses.
- Metrics endpoint on every process: request latency, run duration, reconcile levels per run, outbox depth, outbox age of oldest row, WAL slot lag in bytes, deferred attempts and failures. Outbox depth drives the worker autoscaler; slot lag has an alert.
- Health endpoints: liveness is process-up; readiness checks the database and, for the WAL worker, that the slot is active.

## 10. Infrastructure task list

In dependency order. Each is a unit an agent can take with a definition of done.

| # | Task | Done when |
| --- | --- | --- |
| 1 | Root workspace: `pyproject.toml`, `uv.lock`, three members with empty packages, shared tool config, `justfile` with `setup`, `lint`, `test` | `just setup && just lint && just test` pass on an empty suite |
| 2 | Import boundary lint: core cannot import apps | a deliberate violation fails `just lint` |
| 3 | `compose.yaml` with `test` and `dev` profiles, Postgres with logical replication flags | `just db` starts Postgres; `psql` shows `wal_level = logical` |
| 4 | Settings model with layering and `BB_ENV` | a test proves each layer overrides the last; `prod` refuses default secrets |
| 5 | Testing package: template database, per-test database, two-connection fixture | a test creates, uses and drops a database in under 100 ms |
| 6 | Testing package: WAL fixture | a test commits a row and reads it from the slot |
| 7 | `Dockerfile` with the four stages, SHA tagging, `just build` | image runs `bb --help`; `dev` stage reloads |
| 8 | Compose `dev` profile runs API, worker and WAL worker from the image | `just serve shop` responds on localhost |
| 9 | `ci.yaml` with lint, test against a Postgres service, build, render | a pull request shows all four checks |
| 10 | Kubernetes base manifests and `dev` overlay with in-cluster Postgres | `just k8s-render dev` passes `kubeconform` |
| 11 | `kind` recipes | `just kind-up` serves the API through a port-forward |
| 12 | `prod` overlay with External Secrets and HPAs | `just k8s-render prod` passes `kubeconform` |
| 13 | Terraform modules and `dev` environment | `just tf dev plan` is clean against an empty account |
| 14 | `deploy-dev.yaml` with OIDC, push, migrate, apply, smoke test | merge to `main` deploys to the dev cluster |
| 15 | Terraform `prod` environment | `just tf prod plan` is clean |
| 16 | `deploy-prod.yaml` with promotion gate and approval | a dispatched run deploys a dev-verified SHA |
| 17 | Metrics and health endpoints, worker HPA on outbox depth, slot lag alert | metrics visible in the cluster; a stalled WAL worker alerts |

Tasks 1 through 9 need only this machine with Docker installed. Tasks 10 through 12 need `kind`. Tasks 13 onward need an AWS account and are the only ones that cost money.

## 11. Prerequisites outside the repository

- Docker Engine on the developer machine. The only system-level install; everything else is pinned in the repo.
- `kind` and `kubectl`, installed by `just setup` into the user's path if absent.
- An AWS account with a bootstrap user able to create the Terraform state bucket. After that, CI uses OIDC and no long-lived credentials exist.
- A domain for the dev and prod ingresses.

## 12. Decisions taken in this draft

- **Kustomize over Helm**: our manifests, one shape, plain YAML for agents to read.
- **Terraform over CDK or Pulumi**: the widest module ecosystem and the most readable plans.
- **One image for all processes**: one build and one promotion path.
- **RDS over in-cluster Postgres in prod**: backups, failover and logical replication are someone else's pager.
- **`just` over `make`**: readable recipes, no tab semantics, cross-platform.
- **External Secrets operator**: secrets stay in Secrets Manager; the cluster syncs them.
- **GitHub Actions**: the repo is on GitHub and OIDC to AWS removes stored credentials.
