# Running apps on a host

How the apps in this repository run on one machine, outside Kubernetes: one shared Postgres in Docker,
every app's processes as systemd user units that start at boot. `make` at the root lists the targets;
[guide.md §14](guide.md#14-running-it) describes the processes themselves and
[infrastructure.md](infrastructure.md) the cluster path.

## One command

```sh
make up APP=finance
```

does, in order, and idempotently:

1. `pg`: `docker compose up -d --wait postgres`. One Postgres 17 container per host, `restart:
   unless-stopped`, data in the `postgres` volume. Every app gets its own database inside it, so the
   runtime's `_outbox`, `_run` and `_schedule` tables and every migration chain stay per app.
2. `db`: creates the database named after the app if it does not exist.
3. `env`: writes `~/.config/backend-builder/<app>.env` if it does not exist: the database URL, a port
   block, the roles, a random admin token. Secrets the app's reactions read with `ctx.secret` go in this
   file as `SECRET_<NAME>=...`; it is mode 600 and never in the repo.
4. `migrate`: applies the app's migration chain.
5. Installs `systemd/bb@.service` as a user unit template, enables `bb@<app>:worker`,
   `bb@<app>:scheduler` and `bb@<app>:admin` (plus `bb@<app>:serve` when `BB_ROLES` in the env file
   includes `serve`), enables linger so they run without a login session, and prints their state.

After a reboot: Docker brings Postgres back, systemd brings the units back, `bin/bb-run` waits for
Postgres before starting each process. Nothing to do.

## The pieces

**`bin/bb-run <app>:<role>`** is what every unit runs. It loads the env file, gives the role its ports
from `BB_PORT_BASE` (console on the base, API on base+1, metrics for worker, scheduler and serve on
base+2, base+3 and base+5; the admin process serves no metrics; so several apps and their processes never
collide on one host), waits for Postgres, and
`exec`s `uv run --frozen backend-builder <role> <app>`.

**`systemd/bb@.service`** is one template; the instance name carries the app and the role. `make up`
copies it with the checkout path filled in. `Restart=always`, `RUNTIME_ENV=prod`.

**Port blocks** are allocated by `make env`: the first app gets 9300, the next 9400, and so on, read
from the env files already present.

## Day to day

| Command | Does |
| --- | --- |
| `make status APP=x` | each unit's state and the console URL |
| `make logs APP=x` | follow the journal of the app's units |
| `make restart APP=x` | restart the processes, for a changed env file |
| `make deploy APP=x` | `git pull --ff-only`, `uv sync`, migrate, restart: a merged PR reaching the host |
| `make down APP=x` | stop and disable the units; Postgres and the database stay |
| `make apps` | every app with an env file on this host and the state of its units |
| `make pg-down` | stop Postgres; the volume keeps the data |

Logs are the journal: `journalctl --user -u 'bb@finance:*'`. The admin console is on the app's base
port, bound to localhost, behind the token in the env file. Metrics and `/health` are on the role's port.

## Requirements on the host

Docker with the user in the `docker` group (`sudo usermod -aG docker $USER`, then log in again), `uv`,
and systemd with user sessions. Linger is enabled by `make up` if it is not already.

## Memory on a shared host

Postgres has no memory ceiling in `compose.yaml` and needs none: it uses tens of megabytes here. It
has a floor instead (`mem_reservation: 1g`), the lowest priority for the kernel's out-of-memory
killer (`oom_score_adj: -900`), and a 1 GB `/dev/shm`. The processes that fill the box are the apps':
the finance worker runs one deferred run at a time (`BACKEND_BUILDER_OUTBOX_CONCURRENCY=1` in its env
file; a fit streams its rows in chunks of `FINANCE_FIT_BATCH_ROWS`, so it holds a few GB), its row
builds cap DuckDB at `FINANCE_DUCKDB_MEMORY` (12 GB), and every `bb@` unit has `MemoryMax=24G` in
`systemd/bb@.service`, so a runaway process is killed inside its own unit and nothing else on the box
is a candidate. The unit file is installed by `make up`; after changing it, run `make up APP=finance`
(or copy it to `~/.config/systemd/user/` and `systemctl --user daemon-reload`) and restart the roles. A
change to the postgres service takes effect on `docker compose up -d postgres`, which recreates the
container: do it when no app run is mid-transaction, since every connection drops.
