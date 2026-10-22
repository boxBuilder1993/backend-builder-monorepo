"""PullStore and the mirror: current is reused, a moved store is pulled, a store outage is an
outcome. The store itself is faked; mirror_dir is a convention under the XDG data home."""

from __future__ import annotations

from pathlib import Path

import pytest
from backend_builder import PostgresStore, run
from finance import mirror
from finance.reactions import PullStore


def test_mirror_dir_is_a_convention_under_the_data_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert mirror.mirror_dir("shr22avan/finance-data") == (
        Path.home() / ".local/share/finance/mirror/shr22avan--finance-data"
    )
    monkeypatch.setenv("XDG_DATA_HOME", "/data")
    assert mirror.mirror_dir("a/b") == Path("/data/finance/mirror/a--b")


class FakeStore:
    def __init__(self, revision: str) -> None:
        self.revision = revision
        self.pulls: list[tuple[str, list[str], Path]] = []
        self.down = False

    def store_revision(self, repo: str, token: str) -> str:
        if self.down:
            raise ConnectionError("huggingface.co: 503")
        return self.revision

    def pull(
        self, repo: str, token: str, revision: str, patterns: list[str], root: Path
    ) -> mirror.Pulled:
        self.pulls.append((revision, patterns, root))
        (root / "parquet").mkdir(parents=True, exist_ok=True)
        (root / "parquet" / f"{revision}.parquet").write_bytes(b"x" * 7)
        return mirror.Pulled(files=1, bytes=7, seconds=0.5)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    monkeypatch.setenv("SECRET_HF_TOKEN", "hf_test_token_000000000000")
    f = FakeStore("rev1")
    monkeypatch.setattr(mirror, "store_revision", f.store_revision)
    monkeypatch.setattr(mirror, "pull", f.pull)
    return f


async def test_pull_then_current_then_pull_again_when_the_store_moves(
    store: PostgresStore, data_dir: Path, fake: FakeStore
) -> None:
    ctx = await run(store, PullStore, {})
    assert (ctx.load("status"), ctx.load("revision"), ctx.load("files")) == ("pulled", "rev1", 1)
    assert fake.pulls == [("rev1", ["parquet/**"], data_dir)]

    ctx = await run(store, PullStore, {})  # nothing moved: no download, the mirror is reused
    assert ctx.load("status") == "current" and len(fake.pulls) == 1

    fake.revision = "rev2"
    ctx = await run(store, PullStore, {})
    assert ctx.load("status") == "pulled" and fake.pulls[-1][0] == "rev2"

    ctx = await run(store, PullStore, {"patterns": ["parquet/**", "raw/**"]})  # wider than known
    assert ctx.load("status") == "pulled" and fake.pulls[-1][1] == ["parquet/**", "raw/**"]

    rows = await store.query_raw(
        "select revision, status, files from store_sync order by created_at"
    )
    assert [(r["revision"], r["status"], r["files"]) for r in rows] == [
        ("rev1", "pulled", 1), ("rev1", "current", 0), ("rev2", "pulled", 1), ("rev2", "pulled", 1)
    ]  # fmt: skip


async def test_store_outage_is_an_outcome(
    store: PostgresStore, data_dir: Path, fake: FakeStore
) -> None:
    fake.down = True
    ctx = await run(store, PullStore, {})
    assert ctx.load("status") == "failed" and "503" in ctx.load("error")
    rows = await store.query_raw("select status, error from store_sync")
    assert rows[0]["status"] == "failed" and fake.pulls == []
