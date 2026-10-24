# Backend Builder

A Python runtime for building a company's whole backend as one repository of **entities** and **reactions**.

Entities are plain dataclasses. Reactions are small async classes that read and write entities through a context. Everything else, HTTP endpoints, background work, schedules, retries, transactions, migrations, metrics and an admin console, comes from declaring when reactions run. There is no ORM to learn, no message broker to operate and no service boundary to cross: a request that touches orders, ledgers and stock is one reaction tree, one transaction, one commit.

```python
from backend_builder import Entity, Reaction, Context, App, OnCreate


class Order(Entity):
    user_id: str
    total: int = 0


class OrderLine(Entity):
    order_id: str
    price: int
    qty: int


class CreateOrder(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = ctx.create(Order, user_id=ctx.input["user_id"])
        for line in ctx.input["lines"]:
            ctx.create(OrderLine, order_id=order.id, price=line["price"], qty=line["qty"])
        ctx.save("order", order)


class OrderTotal(Reaction):
    """Runs after commit, by a worker, whenever a line is created."""

    async def calculate(self, ctx: Context) -> None:
        line = await ctx.get_by_id(OrderLine, ctx.input["id"])
        order = await ctx.get_by_id(Order, line.order_id)
        lines = await ctx.query(
            OrderLine, "select * from order_line where order_id = :o", o=order.id
        )
        order.total = sum(l.price * l.qty for l in lines)


app = App(package="myshop", triggers=[OnCreate(OrderLine, OrderTotal)])
```

Mutating a loaded entity is the write. Nothing persists until the run ends; if anything raises, nothing persists at all.

## What you get

- **One map of the business.** `backend-builder graph myshop` lists every entity, reaction, endpoint and trigger. That list is the system.
- **Whole-operation transactions.** A run is one serializable Postgres transaction, retried on conflict. No sagas.
- **Uniform testing.** Build a store, run a reaction, inspect the store. No HTTP, no mocks.
- **Infra as flags.** In memory with no configuration. Postgres by setting one URL. Background work by adding a trigger.
- **Operations built in.** Outbox-backed deferred work with retries and dead letters, cron schedules, per-run timeouts, run records, Prometheus metrics on every process, and an admin console that shows runs, span trees, the outbox and the graph.
- **One image, one CLI.** `serve`, `worker`, `scheduler`, `admin`, `migrate`: the same container, a different command.

## Try the example

The `shop` app under `apps/shop` is an order-to-cash example that exercises every feature. With Docker installed:

```sh
docker compose --profile shop up --build -d     # Postgres, migrations, API, worker, scheduler, admin
curl -s -X POST localhost:8000/orders -H 'content-type: application/json' \
  -d '{"user_id": "u1", "lines": [{"price": 10, "qty": 2}]}'
open http://localhost:8000/docs                  # OpenAPI
open http://localhost:9200                       # admin console
docker compose --profile shop down
```

Without Docker, an app with no triggers runs entirely in memory, and any app's graph and tests run anywhere:

```sh
uv sync --all-packages
uv run backend-builder graph shop
uv run pytest
```

## Documentation

- [Builder's guide](docs/guide.md): how to write an app. Entities, reactions, the context, triggers, endpoints, secrets, migrations, testing. Start here.
- [Design](docs/design.md): why it is the way it is. The full model, every decision and its reasoning, what is parked for later.
- [Infrastructure](docs/infrastructure.md): repository layout, environments and the deployment path. Partly a draft.
- [Logs](docs/logs.md): structured lines to stdout, collected into Loki with one day of retention, shown per run in the console. Decided, not built.
- [Compute pools and service calls](docs/pools.md): the design for running reactions in Kubernetes pools, with `call` as the transactional service-call mode. Decided, not built.
- [Running on a host](docs/running.md): `make up APP=<app>`, one shared Postgres in Docker and the app's processes as systemd user units that start at boot.
- [The forecast model](docs/finance-forecast.md): the first ML problem of the finance app, the expected price Δ minutes ahead, and the order of work.

## Repository

```
packages/backend-builder/          the runtime and its CLI
packages/backend-builder-testing/  pytest plugin: Postgres fixtures for app tests
apps/shop/                         the example app and its end-to-end suite
docs/                              guide, design, infrastructure
compose.yaml, Dockerfile           local stack and the image every app ships as
```

Python 3.13, [uv](https://docs.astral.sh/uv/) workspace, ruff, pyright strict, pytest. `uv run pytest` runs the unit suites; `uv run pytest -m e2e apps/shop/tests/e2e` builds the image and drives the whole stack. CI runs both on every pull request.
