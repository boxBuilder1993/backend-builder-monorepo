# Finance app

Status: proposal, 2026-10-22. The market-data poller that lives today in the `indian-finance` repo
(`ifin`, a Makefile and a systemd timer) becomes an app on Backend Builder, and the `mlkit` library and
its experiments follow it. This document is the design of the poller as an app and the order of the move.
[design.md](design.md) is the authority on the runtime; this document only says how the finance work uses it.

## Why move it

The poller is a nightly batch job with hand-written state: `.nodata` marker files decide what is missing,
a JSON manifest decides what to publish, a shell script decides when to run, and a log file is the only
record of what happened. Each of those is something the runtime already provides: a ledger is an entity,
a schedule is `OnSchedule`, a retry is the outbox, a record is the run table and the admin console. The
exchanges, Yahoo and Kite are external systems that fail in ordinary ways, and the outbox's retry, backoff
and dead letters are the right tool for that.

What does not move: the data. Parquet and raw files stay files, published to the private Hugging Face
dataset repo `shr22avan/finance-data` and read by DuckDB. Postgres holds the ledger and the ML control
state only, kilobytes a day. The invariants of the old repo carry over unchanged: raw files are never
rewritten, one Parquet per day per table, reference tables byte-deterministic, recent days re-checked,
weekends never skipped by rule, the date inside every download validated before it is saved, Kite login
manual, every token under `~/.config/ifin/`, both repos private.

## Layout

```
packages/ifin/            the existing indian_finance package as a library: fetchers, builders, DuckDB views,
                          the Hugging Face store, the research code. Keeps its `ifin` CLI for manual use.
packages/mlkit/           the ML library (docs/mlkit/). No dependency on backend-builder.
apps/finance/             this app: entities, reactions, triggers. Depends on ifin and mlkit.
  src/finance/
    __init__.py           app = App(package="finance", triggers=triggers, timeout=..., max_attempts=1)
    entities.py
    poller.py             the reactions in this document
    ml.py                 the mlkit reactions, later
    triggers.py
    migrations/
  tests/
```

`ifin` stays a library with no knowledge of the runtime, so its functions are called from reactions and
from its own CLI alike, and its tests keep running on files. The app owns no fetching logic; it owns when
things run, what is recorded, and what is retried.

The data directory on the worker's host is `FINANCE_DATA_DIR`, the working copy of the store, the same
layout as today's `data/`. The Hugging Face token is `SECRET_HF_TOKEN`, read with `ctx.secret`, so it is
redacted from run records. Kite's daily token stays a file under `~/.config/ifin/`, written by the manual
login; the reaction reads it from disk, and a missing or expired token is an outcome, not an error.

## Entities

Every row is the ledger of one unit of work. Nothing here duplicates the files; the files stay the data,
the rows say what was done to them.

```python
class SourceDay(Entity):
    """One source on one trading date: nse_cm, nse_full, nse_index, mcx."""
    source: str
    day: date
    status: str              # "fetched" | "no_data" | "failed"
    path: str | None = None  # the raw file, relative to the data dir
    bytes: int = 0
    attempts: int = 0
    error: str | None = None
    checked_at: datetime | None = None   # last time this day was tried; recent no_data days are tried again

class BuiltDay(Entity):
    """One Parquet file: table on day."""
    table: str
    day: date
    path: str
    rows: int
    built_at: datetime
    published_at: datetime | None = None

class ReferenceFile(Entity):
    """A reference table written by write_if_changed: corp_actions, holidays, symbol_changes, ..."""
    name: str
    path: str
    sha256: str
    written_at: datetime
    published_at: datetime | None = None

class UsSymbol(Entity):
    """Yahoo state per symbol, replacing the dormant file."""
    symbol: str              # Yahoo form, BRK-B
    last_date: date | None = None
    empty_runs: int = 0
    dormant: bool = False

class KiteRange(Entity):
    """One historical request: an instrument over at most 60 days."""
    symbol: str
    start: date
    end: date
    status: str              # "fetched" | "failed"
    candles: int = 0
    error: str | None = None

class Publish(Entity):
    """One commit to the Hugging Face store."""
    files: int
    bytes: int
    commit: str | None = None
    status: str              # "ok" | "failed"
    error: str | None = None
```

Uniqueness, `(source, day)`, `(table, day)`, `name`, `symbol`, `(symbol, start, end)`, is a migration, as the
runtime requires. The gap report is a query: trading dates from the holiday table minus `SourceDay` rows
with `status = 'fetched'`, and no-data rows inside the last seven days whose `checked_at` is before
today. The DuckDB view of what is on disk, which `ifin status` prints today, becomes a SQL query on these
tables with the same answer.

## Reactions

Fetching is work against an external system, so it is deferred work, retried by the worker. Each leg is
one run per night, sequential inside, because each source has its own pacing (NSE 0.25 s between requests,
Kite 3 a second) and that pacing is simplest to keep inside one process. A run that is retried after a
failure skips every file already on disk, so a retry costs the remaining work, not the whole leg.

```python
class NightlySync(Reaction):
    """Scheduled 14:30 and 16:30 UTC (20:00 and 22:00 IST). Decides what is missing and defers one run
    per leg. Input: {"scheduled_for": ...}. Writes nothing itself."""
    # defers FetchReference, FetchSources, SyncUs, SyncKite, each with its own budget

class FetchReference(Reaction):
    """Corporate actions, holidays, symbol changes, equity master. Rewrites a ReferenceFile row only when
    the bytes changed (write_if_changed), so an unchanged table is not republished."""

class FetchSources(Reaction):
    """Every missing (source, day) from the backfill start to today, and every no_data day inside
    RECENT_DAYS not checked today. One SourceDay row per day tried. Then builds the Parquet for every
    day whose sources are complete and whose BuiltDay is missing, one BuiltDay row per file."""

class SyncUs(Reaction):
    """Yahoo in batches; UsSymbol rows carry the per-symbol state. Failed batches are left for the next
    night, as today."""

class SyncKite(Reaction):
    """Minute candles for the universe, one KiteRange row per request, then the equity_1m Parquet per
    day. A missing or expired Kite token saves {"kite": "login needed"} and returns: an outcome."""

class PublishChanged(Reaction):
    """Scheduled every 30 minutes. Uploads every BuiltDay and ReferenceFile with published_at null in one
    Hugging Face commit, in chunks of at most 1000 files or 1 GB, stamping published_at per chunk. Refuses
    a repo that is not private. At-least-once is harmless: the store dedupes identical bytes."""
```

The reactions call `ifin` functions that do the HTTP and the file writes; the reaction's own job is the
rows. File writes are not part of the transaction, which is why the order inside a reaction is always
file first, row second, and why every file write is idempotent: a raw file that exists is never
rewritten, a Parquet write compares bytes. A retry after a crash between the two finds the file and
writes the row.

The fetchers are synchronous `requests` code. A reaction runs them with `asyncio.to_thread` so the worker's
loop, its poll and its metrics endpoint stay responsive. One worker runs the legs; `OUTBOX_CONCURRENCY`
stays at the default four so the four legs run side by side.

Budgets: `NightlySync` 60 s. `FetchSources` 3600 s, `SyncUs` 3600 s, `SyncKite` 7200 s (the backfill is
9,000 requests at 3 a second; a normal night is a minute). `PublishChanged` 3600 s. `App.max_attempts=1`:
a leg that fails is released to the outbox with backoff, not re-run from the top inside the same budget,
and the next night's `NightlySync` covers whatever the backoff did not.

Triggers:

```python
triggers = [
    OnSchedule("30 14 * * *", NightlySync, timeout=60),
    OnSchedule("30 16 * * *", NightlySync, timeout=60),
    OnSchedule("*/30 * * * *", PublishChanged, timeout=3600),
]
```

No data-change triggers in the first version. `OnChange(SourceDay, ["status"], BuildDay)` is the natural
next step once the legs are chunked per day, and would make the build follow every fetch instead of
running at the end of the leg.

## What the runtime does not have yet

Three things came up in the design and are worth doing in the runtime rather than around it. None blocks
the first version.

1. **Progress visible during a run.** A leg is one serializable transaction, so a `SourceDay` row written
   at the start of an hour-long run is invisible until the commit, and the console shows a running
   `SyncKite` with no rows. A sanctioned write outside the run's transaction, `ctx.note(entity, id,
   changes)` in its own short transaction, would let a long reaction report progress. Until then the
   runtime's span log and the row count at the end are what there is.
2. **Chunked legs with pacing.** The right shape for `FetchSources` is one `FetchDay(source, day)` per
   missing day spread across workers. That needs a shared rate limit per source, which the outbox does not
   have: a `queue` on a deferred trigger and `--queue` with a concurrency on the worker. With that, the
   legs become hundreds of short runs and the `OnChange` build trigger above follows.
3. **Per-trigger attempts.** `App.max_attempts` is one policy for every reaction. The legs want one attempt
   and the outbox's backoff; a future HTTP endpoint on this app wants three. `OnSchedule(...,
   max_attempts=1)` and the same on `ctx.trigger(deferred=True)` would let both coexist.

## Order of the move

1. **Library in.** `packages/ifin` is the `src/indian_finance` tree and its tests, unchanged apart from the
   package layout; the DuckDB views, fetchers, builders, store and research code come with it. The
   monorepo is Python 3.13; `numba` and `kiteconnect` are checked against it first.
2. **App.** `apps/finance` with the entities, reactions and triggers above, a `finance` compose profile
   (Postgres, migrate, worker, scheduler, admin) and the migrations. Tests: the gap query and the
   row-writing of each reaction on the in-memory store with the fetchers stubbed; `migrations.check`
   against Postgres.
3. **Shadow nights.** The new worker runs on the Linux box with `PublishChanged` disabled and
   `FINANCE_DATA_DIR` pointing at a copy of the poller's data. For a week its ledger is compared with the
   files the systemd poller produced the same night. Then the timer is disabled, `PublishChanged` is
   enabled, and the old poller's clone is kept as a cold copy of the data.
4. **The old repo.** `indian-finance` on GitHub becomes the archive of the data history and nothing else;
   the code lives here. Rewriting its history to drop the 5 GB of data is a separate decision.
5. **mlkit.** `packages/mlkit` and the `Experiment`, `FitRun`, `Model` and `InferenceRun` entities with
   their reactions, per `docs/mlkit/`. The first two framework items above matter more here: a fit is an
   hour of GPU time, and it wants a subprocess and a progress row.

## Decisions

- Files are the data and are published to the Hugging Face store; Postgres holds only the ledger and the
  ML control state. No market data is loaded into Postgres.
- One run per leg per night, sequential inside, until the runtime has queues with pacing.
- File first, row second, every file write idempotent. The transaction covers the rows, never the files.
- Legs run with one attempt; the outbox's backoff is the retry.
- The Kite login stays manual and outside the app; a missing token is an outcome.
- `ifin` stays a library and a CLI; the app owns scheduling, the ledger and retries, not fetching.
