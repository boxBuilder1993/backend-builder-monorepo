# Postgres backups

The shared Postgres holds every app's control state: the runtime's tables and the app's entities. The
data itself lives in the Hugging Face store and its mirror, so the databases are small, megabytes, and
cheap to dump every night. `bin/bb-backup` does it; `backup.mk` (included by the root Makefile) wraps
it; a systemd user timer runs it daily.

| Command | Does |
| --- | --- |
| `make pg-backup [DB=x]` | `pg_dump -Fc` of every app database, or one, into `~/.local/share/backend-builder/backups/<db>/<db>_<UTC>.dump`; prune; upload when configured |
| `make pg-backups [DB=x]` | local dumps with size and age, and the remote ones |
| `make pg-fetch DB=x [NAME=f]` | pull the newest remote dump, or a named one, into the local directory |
| `make pg-restore DB=x FILE=f [INTO=y]` | `pg_restore` into a fresh database `x_restored` by default; `INTO=x` replaces the live one and asks first |
| `make pg-backup-install` | daily at 03:00, `Persistent=true`, so a night the box was off runs at boot |
| `make pg-backup-remove` | stop the timer |

**Retention**: the newest 14 dumps per database (`BACKUP_KEEP_DAILY`) plus the first dump of every month.

**Off-box**: `~/.config/backend-builder/backup.env` with `BACKUP_REPO=<owner>/<private dataset repo>` and
`HF_TOKEN=...` makes every dump also go to `<repo>/<db>/<file>`. A separate repo from the data store, so
a restore never depends on the thing being restored. The repo must be private: the dumps carry run
inputs and vars.

**What is not backed up**: the env files with the tokens (`~/.config/backend-builder/*.env`), the
mirror (it is the store's copy), the model cache. Keep the env files in a password manager.

**Verifying a backup** is a restore into a scratch database followed by the migration check:

```sh
make pg-restore DB=finance FILE=finance_20261023T030000Z.dump INTO=finance_check
BACKEND_BUILDER_DATABASE_URL=postgresql://backend_builder:backend_builder@127.0.0.1:5432/finance_check \
  uv run backend-builder migrate finance --check
docker compose exec -T postgres psql -U backend_builder -c 'drop database finance_check'
```

Do that once a week. An untested backup is a hope.

The dumps are taken through `docker compose exec postgres`, so no Postgres client is needed on the host
and the dump version always matches the server. Everything here is plain `make`, `docker compose` and
the `hf` CLI: it works when the apps are down, which is when it matters.
