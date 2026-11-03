"""The local mirror of the store. The store is the source of truth; nothing reads data that has not
been brought to the store's current revision, and a current mirror is reused untouched.

The mirror's location is a convention, not configuration: ``mirror_dir(repo)`` under the XDG data
home. Every reader and writer in the app goes through it, the poller's legs included.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# huggingface_hub types its listings loosely.

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path


def data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()


def mirror_dir(repo: str) -> Path:
    """~/.local/share/finance/mirror/<owner>--<name>/ for "owner/name"."""
    return data_home() / "finance" / "mirror" / repo.replace("/", "--")


def store_revision(repo: str, token: str) -> str:
    """The store's current commit."""
    from huggingface_hub import HfApi

    info = HfApi(token=token).dataset_info(repo)
    sha = info.sha
    if not sha:
        raise RuntimeError(f"{repo}: no revision reported")
    return sha


DERIVED_PREFIXES = (
    "parquet/inference/",
    "parquet/walkforward/",
    "parquet/screens/",
    "parquet/backtests/",
    "parquet/studies/",
    "models/",
)
"""What the pipeline writes to the store itself: model directories, inference tables, study and
backtest results. They move the store's commit every few minutes while a search runs, and nothing a
fit reads lives there, so the data revision leaves them out."""

MANIFEST = ".manifest.json"
"""In the mirror root: path -> blob id of every file the mirror holds from the store."""


@dataclass(frozen=True)
class Pulled:
    files: int
    bytes: int
    seconds: float
    data_revision: str = ""
    """A hash of the market data's (path, blob) pairs at the revision pulled: data_revision()."""


def pattern_prefix(pattern: str) -> str:
    """The folder a glob pattern is confined to: ``parquet/**`` lists ``parquet``, ``*`` all."""
    head = re.split(r"[*?\[]", pattern, maxsplit=1)[0]
    return head.rsplit("/", 1)[0] if "/" in head else ""


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def tree(repo: str, token: str, revision: str, patterns: list[str]) -> dict[str, str]:
    """path -> blob id of every file under the patterns at the revision: one recursive listing per
    pattern folder (about 25 seconds for the 21,000 files of the data store)."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi(token=token)
    out: dict[str, str] = {}
    for prefix in sorted({pattern_prefix(p) for p in patterns}):
        entries = api.list_repo_tree(
            repo, prefix or None, recursive=True, revision=revision, repo_type="dataset"
        )
        for e in entries:
            if isinstance(e, RepoFile) and matches(e.path, patterns):
                out[e.path] = e.blob_id
    return out


def data_revision(listing: dict[str, str]) -> str:
    """A hash of the (path, blob) pairs outside DERIVED_PREFIXES: it moves when the market data
    moves and stays put when the pipeline commits its own outputs, so rows derived from the data can
    be keyed by it."""
    h = hashlib.sha256()
    for path in sorted(listing):
        if not path.startswith(DERIVED_PREFIXES):
            h.update(f"{path}\0{listing[path]}\n".encode())
    return h.hexdigest()


def download(repo: str, token: str, revision: str, path: str, root: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=root, token=token)


def read_manifest(root: Path) -> dict[str, str]:
    f = root / MANIFEST
    if not f.is_file():
        return {}
    loaded = json.loads(f.read_text())
    return {str(k): str(v) for k, v in loaded.items()}


def write_manifest(root: Path, manifest: dict[str, str]) -> None:
    tmp = root / f"{MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, sort_keys=True, indent=0))
    tmp.replace(root / MANIFEST)


def pull(repo: str, token: str, revision: str, patterns: list[str], root: Path) -> Pulled:
    """Bring root to the store at revision for the patterns: list the tree once, fetch the files
    whose blob id differs from the manifest's or that are missing, record the manifest. Nothing is
    deleted. A mirror without a manifest (one that snapshot_download built) keeps every present file
    whose size matches the store's and fetches the rest; from then on the blob ids decide."""
    t0 = time.time()
    root.mkdir(parents=True, exist_ok=True)
    listed = tree(repo, token, revision, patterns)
    known = read_manifest(root)
    bootstrap = not known
    if bootstrap:
        sizes = _sizes(repo, token, revision, patterns)
        known = {
            p: b
            for p, b in listed.items()
            if (root / p).is_file() and (root / p).stat().st_size == sizes.get(p, -1)
        }
    todo = [p for p, b in listed.items() if known.get(p) != b or not (root / p).is_file()]
    bytes_ = 0
    for p in todo:
        download(repo, token, revision, p, root)
        bytes_ += (root / p).stat().st_size
    write_manifest(root, {**known, **listed})
    return Pulled(len(todo), bytes_, time.time() - t0, data_revision(listed))


def _sizes(repo: str, token: str, revision: str, patterns: list[str]) -> dict[str, int]:
    """path -> size, for the one-time bootstrap of a manifest-less mirror."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi(token=token)
    out: dict[str, int] = {}
    for prefix in sorted({pattern_prefix(p) for p in patterns}):
        entries = api.list_repo_tree(
            repo, prefix or None, recursive=True, revision=revision, repo_type="dataset"
        )
        for e in entries:
            if isinstance(e, RepoFile) and matches(e.path, patterns):
                out[e.path] = int(e.size or 0)
    return out
