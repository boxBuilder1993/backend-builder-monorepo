"""The local mirror of the store. The store is the source of truth; nothing reads data that has not
been brought to the store's current revision, and a current mirror is reused untouched.

The mirror's location is a convention, not configuration: ``mirror_dir(repo)`` under the XDG data
home. Every reader and writer in the app goes through it, the poller's legs included.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# huggingface_hub types snapshot_download loosely.

from __future__ import annotations

import os
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


@dataclass(frozen=True)
class Pulled:
    files: int
    bytes: int
    seconds: float


def pull(repo: str, token: str, revision: str, patterns: list[str], root: Path) -> Pulled:
    """Bring root to the store at revision for the patterns. Only files whose hash differs are
    fetched; nothing is deleted. Counts the files written during the call."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=patterns,
        token=token,
    )
    files = bytes_ = 0
    for p in root.rglob("*"):
        if p.is_file() and ".cache" not in p.parts and p.stat().st_mtime >= t0:
            files += 1
            bytes_ += p.stat().st_size
    return Pulled(files, bytes_, time.time() - t0)
