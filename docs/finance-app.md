# Finance app

Status: proposal, 2026-10-22. A new app, `apps/finance`, written from scratch on Backend Builder, takes
over the market-data poller that lives today in the `indian-finance` repo (`ifin`, a Makefile and a
systemd timer); the `mlkit` library and its experiments follow it. The old repo is the reference for
every source quirk and the record of what the data looks like; none of its code is imported or moved.
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
apps/finance/
  src/finance/
    __init__.py           app = App(package="finance", triggers=triggers, timeout=..., max_attempts=1)
    entities.py           the ledger, below
    reactions.py          the reactions, below
    triggers.py
    sources/              one module per external source: HTTP, pacing, the raw-file format, the date check
      nse.py              cm bhavcopy (old and UDiFF layouts), sec_bhavdata_full, ind_close_all, the meta files
      mcx.py              futures bhavcopy
      yahoo.py            US equities in batches, the listings
      kite.py             minute candles, the instrument master, the login state
    build/                raw -> Parquet, one module per table family; byte-deterministic writers
    store.py              the Hugging Face store: changed files, chunked commits, the private check
    calendar.py           trading dates per market from the holiday table; weekends are not skipped by rule
    migrations/
  tests/
packages/mlkit/           the ML library (docs/mlkit/), later. No dependency on backend-builder.
```

Everything under `sources/` and `build/` is plain Python with no knowledge of the runtime: functions that
take dates and paths and return what they fetched or wrote. Reactions call them and write rows. That split
is what keeps the fetch code testable on files and the reactions testable on the in-memory store.

The old repo's README and its parsers are the specification for `sources/` and `build/`: the UDiFF cutover
dates, the 404 after July 2024 for the old cm layout, MCX trading on NSE holidays, Yahoo's symbol form
versus Nasdaq's, Kite's 60-day ceiling and the 15:14 end of its history for some symbols, ETFs duplicated
in the Kite backfill, every one of them found by probing and recorded there. Each is carried over as a
test here before the source that needs it is written. The file formats and paths stay identical, so the
store the old poller filled is the store this app continues.

The data directory on the worker's host is `FINANCE_DATA_DIR`, the working copy of the store, the same
layout as today's `data/`. The Hugging Face token is `SECRET_HF_TOKEN`, read with `ctx.secret`, so it is
redacted from run records. Kite's daily token stays a file under `~/.config/ifin/`, written by the manual
login; the reaction reads it from disk, and a missing or expired token is an outcome, not an error.

## Entities

One entity. Every file on disk gets a row, whatever kind it is, and the rows are the ledger: nothing here
duplicates the files, the files stay the data, the rows say what was done to them.

```python
class DataFile(Entity):
    """One file in the store, or one (source, day) known to have no file."""
    path: str                       # relative to the data dir; unique (migration)
    kind: str                       # "raw" | "parquet" | "reference"
    source: str                     # raw: nse_cm, nse_full, nse_index, mcx, yahoo, kite_1m; parquet: the table
    day: date | None                # None for reference files and the Yahoo state file
    status: str                     # "ok" | "no_data"   (no_data rows have no file)
    bytes: int = 0
    sha256: str | None = None       # reference files, whose rewrite is detected by content
    written_at: datetime
    checked_at: datetime            # last time the source was asked; recent no_data days are asked again
    published_at: datetime | None = None
```

Everything the old poller kept as side files or derived from directory scans is a query on this table:

- the gap report is trading dates per market (from the holiday reference file) minus `ok` rows per source,
  plus `no_data` rows inside the last seven days whose `checked_at` is before today;
- what to publish is rows with `published_at` null;
- what to build is days with every raw source `ok` and no `parquet` row for the table;
- Kite needs no range table, because its raw is already one file per day.

The per-symbol Yahoo state (last date, empty runs, dormant) stays the small JSON file the old poller keeps
under `raw/`; it is a `DataFile` like any other and is published with the rest. Unique `path` and an index
on `(source, day)` are a migration.

## Reactions

Three reactions. Fetching is work against an external system, so it is deferred work, retried by the
worker. Each leg is one run per night, sequential inside, because each source has its own pacing (NSE
0.25 s between requests, Kite 3 a second) and that pacing is simplest to keep inside one process. A run
that is retried after a failure skips every file already on disk, so a retry costs the remaining work.

```python
class NightlySync(Reaction):
    """Scheduled 14:30 and 16:30 UTC (20:00 and 22:00 IST). Defers one SyncLeg per leg, each with its own
    budget. Input: {"scheduled_for": ...}. Writes nothing itself."""

class SyncLeg(Reaction):
    """Input: {"leg": "reference" | "nse" | "us" | "kite"}. Calls the leg's function in sources/, which
    fetches what is missing and writes files, then builds the Parquet for the days that are now complete,
    then writes one DataFile row per file written or day found empty. The leg name is saved to vars so
    the console can filter runs by leg.

    reference: corporate actions, holidays, symbol changes, equity master; write_if_changed, a row only
               when the bytes changed.
    nse:       every missing (source, day) for nse_cm, nse_full, nse_index and mcx from the backfill start,
               every no_data day inside RECENT_DAYS not checked today; then equity, index and commodity
               Parquet for complete days.
    us:        Yahoo in batches; failed batches wait for the next night, as today.
    kite:      minute candles for the universe in 60-day requests, then equity_1m per day. A missing or
               expired Kite token saves {"kite": "login needed"} and returns: an outcome, not an error."""

class Publish(Reaction):
    """Scheduled every 30 minutes. Uploads every DataFile with published_at null to the Hugging Face store
    in commits of at most 1000 files or 1 GB, stamping published_at per commit. Refuses a repo that is not
    private. At-least-once is harmless: the store dedupes identical bytes."""
```

The reactions call the `sources/` and `build/` functions that do the HTTP and the file writes; the
reaction's own job is the rows. File writes are not part of the transaction, which is why the order inside
a reaction is always file first, row second, and why every file write is idempotent: a raw file that
exists is never rewritten, a Parquet write compares bytes. A retry after a crash between the two finds the
file and writes the row.

The fetchers are synchronous `requests` code. A reaction runs them with `asyncio.to_thread` so the worker's
loop, its poll and its metrics endpoint stay responsive. One worker runs the legs; `OUTBOX_CONCURRENCY`
stays at the default four so the four legs run side by side.

Budgets: `NightlySync` 60 s. `SyncLeg` 3600 s for reference, nse and us, 7200 s for kite (the backfill is
9,000 requests at 3 a second; a normal night is a minute). `Publish` 3600 s. `App.max_attempts=1`: a leg
that fails is released to the outbox with backoff, not re-run from the top inside the same budget, and the
next `NightlySync` covers whatever the backoff did not.

Triggers:

```python
triggers = [
    OnSchedule("30 14,16 * * *", NightlySync, timeout=60),
    OnSchedule("*/30 * * * *", Publish, timeout=3600),
]
```

**Catch-up.** Every `NightlySync` recomputes the gaps over the whole range, so a night the machine was off,
a leg that died, or a file the exchange published late is filled by the next run that happens, with no
manual step. A failed leg is retried by the outbox with backoff up to five times, then marked dead and
shown in the console; even then nothing is lost, the next `NightlySync` defers the leg again. A tick that
falls while the machine is off is skipped, as cron does: the data is caught up at the next scheduled run,
not at boot. That is deliberate; a machine switched on in the afternoon should not start a backfill.

No data-change triggers in the first version. `OnChange(DataFile, ["status"], ...)` is the natural next
step once the legs are chunked per day, and would make the build follow every fetch instead of running at
the end of the leg.

## What the runtime does not have yet

Four things came up in the design and the build and are worth doing in the runtime rather than around
it. None blocks the first version.

1. **Progress visible during a run.** A leg is one serializable transaction, so a `DataFile` row written
   at the start of an hour-long run is invisible until the commit, and the console shows a running
   kite leg with no rows. A sanctioned write outside the run's transaction, `ctx.note(entity, id,
   changes)` in its own short transaction, would let a long reaction report progress. Until then the
   runtime's span log and the row count at the end are what there is.
2. **Chunked legs with pacing.** The right shape for the nse leg is one `FetchDay(source, day)` per
   missing day spread across workers. That needs a shared rate limit per source, which the outbox does not
   have: a `queue` on a deferred trigger and `--queue` with a concurrency on the worker. With that, the
   legs become hundreds of short runs and the `OnChange` build trigger above follows.
3. **Per-trigger attempts.** `App.max_attempts` is one policy for every reaction. The legs want one attempt
   and the outbox's backoff; a future HTTP endpoint on this app wants three. `OnSchedule(...,
   max_attempts=1)` and the same on `ctx.trigger(deferred=True)` would let both coexist.
4. **Several schedules for one reaction.** The `_schedule` table is keyed by reaction name, so two
   `OnSchedule` entries for the same reaction see each other's cron as a change and reseed every tick.
   The app uses one cron with two hours; a key of (reaction, cron) would lift the restriction.

## Order of the move

1. **Skeleton.** `apps/finance` in the workspace with `DataFile`, the three reactions with the legs
   stubbed, the triggers, the migrations and a `finance` compose profile (Postgres, migrate, worker,
   scheduler, admin). `backend-builder graph finance` prints it; `migrations.check` passes.
2. **Sources, one at a time**, each with its quirk tests, its reaction and its builder: NSE cm and full
   bhavcopy first, since `equity` depends on both, then the NSE indices, MCX, the reference files, Yahoo,
   Kite. After each, the reaction is run by hand against a copy of the store and its Parquet compared
   byte for byte with the old poller's file for the same day. The old poller keeps running throughout.
3. **Shadow nights.** The new worker runs on the Linux box with `Publish` disabled and
   `FINANCE_DATA_DIR` pointing at a copy of the poller's data. For a week its ledger is compared with the
   files the systemd poller produced the same night. Then the timer is disabled, `Publish` is
   enabled, and the old poller's clone is kept as a cold copy of the data.
4. **The old repo.** `indian-finance` on GitHub becomes the archive: the data history and the reference
   for quirks. The research code (dip search, signal scans, the notebooks) stays there until the mlkit
   work gives it a home. Rewriting its history to drop the 5 GB of data is a separate decision.
5. **mlkit.** `packages/mlkit` and the `Experiment`, `FitRun`, `Model` and `InferenceRun` entities with
   their reactions, per `docs/mlkit/`. The first two framework items above matter more here: a fit is an
   hour of GPU time, and it wants a subprocess and a progress row.

## Decisions

- Files are the data and are published to the Hugging Face store; Postgres holds only the ledger and the
  ML control state. No market data is loaded into Postgres.
- One entity, the file ledger; three reactions. State the files already hold is not duplicated in rows.
- One run per leg per night, sequential inside, until the runtime has queues with pacing.
- Missed ticks are not replayed at boot; the next scheduled run catches up.
- File first, row second, every file write idempotent. The transaction covers the rows, never the files.
- Legs run with one attempt; the outbox's backoff is the retry.
- The Kite login stays manual and outside the app; a missing token is an outcome.
- The app is written anew; nothing is imported from the old repo. Its quirks become tests here, and the
  byte-for-byte comparison of Parquet files is the acceptance test for each source.
- `sources/` and `build/` know nothing about the runtime; reactions know nothing about HTTP or Parquet.
