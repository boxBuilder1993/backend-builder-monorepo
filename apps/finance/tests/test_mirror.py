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


def test_pattern_prefix_and_data_revision_leave_the_pipelines_outputs_out() -> None:
    assert mirror.pattern_prefix("parquet/**") == "parquet"
    assert mirror.pattern_prefix("parquet/equity/*.parquet") == "parquet/equity"
    assert mirror.pattern_prefix("*") == ""
    assert mirror.pattern_prefix("raw/nse_cm/2026*") == "raw/nse_cm"
    assert mirror.matches("parquet/equity/a.parquet", ["parquet/**"])
    assert not mirror.matches("raw/x.csv", ["parquet/**"])
    base = {
        "parquet/equity/a.parquet": "b1",
        "parquet/inference/i.parquet": "b2",
        "models/m/x": "b3",
    }
    same_data = {**base, "parquet/inference/i.parquet": "b9", "models/m/x": "b8"}
    moved = {**base, "parquet/equity/a.parquet": "b7"}
    assert mirror.data_revision(base) == mirror.data_revision(same_data)
    assert mirror.data_revision(base) != mirror.data_revision(moved)


def test_pull_fetches_only_what_the_manifest_does_not_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = {"parquet/a.parquet": "b1", "parquet/b.parquet": "b2", "parquet/c.parquet": "b3"}
    sizes = {"parquet/a.parquet": 3, "parquet/b.parquet": 3, "parquet/c.parquet": 3}
    fetched: list[str] = []

    def fake_tree(repo: str, token: str, revision: str, patterns: list[str]) -> dict[str, str]:
        return dict(listing)

    def fake_sizes(repo: str, token: str, revision: str, patterns: list[str]) -> dict[str, int]:
        return dict(sizes)

    def fake_download(repo: str, token: str, revision: str, path: str, root: Path) -> None:
        fetched.append(path)
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(b"new")

    monkeypatch.setattr(mirror, "tree", fake_tree)
    monkeypatch.setattr(mirror, "_sizes", fake_sizes)
    monkeypatch.setattr(mirror, "download", fake_download)
    root = tmp_path / "mirror"
    (root / "parquet").mkdir(parents=True)
    (root / "parquet" / "a.parquet").write_bytes(b"old")  # right size: kept on the bootstrap
    (root / "parquet" / "b.parquet").write_bytes(b"wrong size")
    got = mirror.pull("r", "t", "rev1", ["parquet/**"], root)
    assert sorted(fetched) == ["parquet/b.parquet", "parquet/c.parquet"] and got.files == 2
    assert got.data_revision == mirror.data_revision(listing)
    assert mirror.read_manifest(root) == listing
    # the store moves one blob: only that file is fetched; a pipeline output does not move the data
    fetched.clear()
    listing["parquet/b.parquet"] = "b2x"
    listing["parquet/inference/i.parquet"] = "i1"
    got = mirror.pull("r", "t", "rev2", ["parquet/**"], root)
    assert sorted(fetched) == ["parquet/b.parquet", "parquet/inference/i.parquet"]
    assert got.data_revision == mirror.data_revision(listing)
    assert mirror.read_manifest(root)["parquet/b.parquet"] == "b2x"
    # nothing moved: nothing fetched
    fetched.clear()
    assert mirror.pull("r", "t", "rev2", ["parquet/**"], root).files == 0 and fetched == []
