from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from backend_builder import PostgresStore, Worker, migrations, run
from finance import app, sources
from finance import store as hfstore
from finance.entities import DataFile
from finance.reactions import NightlySync, Publish, SyncLeg, nodata_path
from finance.settings import HF_REPO
from finance.sources import Ledger, LegResult, NoData, Written


def test_app_scope() -> None:
    assert {e.__name__ for e in app.entities} == {"DataFile"}
    assert {r.__name__ for r in app.reactions} == {"NightlySync", "SyncLeg", "Publish"}
    assert [t.cron for t in app.schedules] == ["30 14 * * *", "30 16 * * *", "*/30 * * * *"]
    assert set(sources.LEGS) == set(sources.BUDGETS) == set(sources.SOURCES)


def test_migrations_match_entities(migrated_database_url: str) -> None:
    migrations.check(app, migrated_database_url)


async def test_nightly_sync_defers_one_run_per_leg(store: PostgresStore) -> None:
    ctx = await run(store, NightlySync, {"scheduled_for": "2026-10-22T14:30:00+00:00"})
    assert ctx.load("deferred") == ["reference", "nse", "us", "kite"]
    rows = await store.query_raw(
        "select reaction, input, timeout_seconds from _outbox order by created_at"
    )
    assert [r["input"]["leg"] for r in rows] == ["reference", "nse", "us", "kite"]
    assert all(r["reaction"] == "finance.reactions.SyncLeg" for r in rows)
    assert {r["timeout_seconds"] for r in rows} == {3600, 7200}


async def test_unimplemented_leg_is_an_outcome(store: PostgresStore, data_dir: Path) -> None:
    ctx = await run(store, SyncLeg, {"leg": "kite"})
    assert ctx.load("implemented") is False
    assert await store.query_raw("select count(*) as n from data_file") == [{"n": 0}]


async def test_unknown_leg_fails() -> None:
    from backend_builder import InMemoryStore

    with pytest.raises(LookupError):
        await run(InMemoryStore(), SyncLeg, {"leg": "bonds"}, max_attempts=1)


def _fake_leg(seen: list[Ledger]) -> sources.Leg:
    def leg(data_dir: Path, today: date, ledger: Ledger) -> LegResult:
        seen.append(ledger)
        f = data_dir / "raw/nse_cm/2026/nse_cm_20261021.zip"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * 10)
        return LegResult(
            written=[
                Written(
                    "raw/nse_cm/2026/nse_cm_20261021.zip", "raw", "nse_cm", date(2026, 10, 21), 10
                )
            ],
            no_data=[NoData("nse_cm", date(2026, 10, 19))],
            notes={"requests": 2},
        )

    return leg


async def test_sync_leg_writes_rows_idempotently(
    store: PostgresStore, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Ledger] = []
    monkeypatch.setitem(sources.LEGS, "nse", _fake_leg(seen))
    ctx = await run(store, SyncLeg, {"leg": "nse"})
    assert (ctx.load("written"), ctx.load("no_data"), ctx.load("notes")) == (1, 1, {"requests": 2})
    assert seen[0] == Ledger()  # nothing known yet

    rows = await store.query_raw(
        "select path, status, bytes, published_at from data_file order by path"
    )
    assert [(r["path"], r["status"], r["bytes"]) for r in rows] == [
        (nodata_path("nse_cm", date(2026, 10, 19)), "no_data", 0),
        ("raw/nse_cm/2026/nse_cm_20261021.zip", "ok", 10),
    ]
    assert all(r["published_at"] is None for r in rows)

    await run(store, SyncLeg, {"leg": "nse"})  # second night: same files, no new rows
    assert await store.query_raw("select count(*) as n from data_file") == [{"n": 2}]
    assert ("nse_cm", date(2026, 10, 21)) in seen[1].ok
    assert seen[1].no_data == frozenset()  # the no_data day is recent, so it is asked again


async def test_publish_stamps_rows(
    store: PostgresStore, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET_HF_TOKEN", "hf_test_token_000000000000")
    uploaded: list[tuple[list[str], str, str]] = []

    def fake_upload(
        root: Path, paths: list[str], repo: str, token: str, message: str
    ) -> list[hfstore.Commit]:
        uploaded.append((paths, repo, token))
        return [hfstore.Commit(tuple(paths), 10, "abc")]

    monkeypatch.setattr(hfstore, "upload", fake_upload)
    now = datetime.now(UTC)
    await store.create(
        DataFile, path="raw/a", kind="raw", source="nse_cm", status="ok", bytes=10, written_at=now
    )
    await store.create(
        DataFile, path="nse_cm/.nodata/2026-10-19", kind="raw", source="nse_cm", status="no_data"
    )
    await store.create(
        DataFile, path="raw/b", kind="raw", source="mcx", status="ok", bytes=5, published_at=now
    )

    ctx = await run(store, Publish, {"scheduled_for": "x"})
    assert ctx.load("files") == 1 and ctx.load("commits") == ["abc"]
    assert uploaded == [(["raw/a"], HF_REPO, "hf_test_token_000000000000")]
    rows = await store.query_raw("select path from data_file where published_at is null")
    assert [r["path"] for r in rows] == ["nse_cm/.nodata/2026-10-19"]

    ctx = await run(store, Publish, {"scheduled_for": "y"})  # nothing left: no upload call
    assert ctx.load("files") == 0 and len(uploaded) == 1


async def test_worker_delivers_the_legs(store: PostgresStore, data_dir: Path) -> None:
    await run(store, NightlySync, {"scheduled_for": "x"}, triggers=app.triggers)
    worker = Worker(app, store)
    assert await worker.run_once() == 4
    rows = await store.query_raw(
        "select reaction, outcome, vars from _run where reaction like '%SyncLeg'"
        " order by started_at"
    )
    assert len(rows) == 4 and {r["outcome"] for r in rows} == {"ok"}
    assert all(r["vars"]["implemented"] is False for r in rows)


def test_chunks_respect_both_limits() -> None:
    sizes = {"a": 300, "b": 300, "c": 300, "d": 2000, "e": 1}
    assert hfstore.chunks(list(sizes), sizes, max_files=10, max_bytes=700) == [
        ["a", "b"],
        ["c"],
        ["d"],
        ["e"],
    ]
    assert hfstore.chunks(list(sizes), sizes, max_files=2, max_bytes=10**9) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]
    assert hfstore.chunks([], sizes) == []
