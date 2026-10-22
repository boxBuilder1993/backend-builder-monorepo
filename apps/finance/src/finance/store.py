"""The Hugging Face store: uploads in chunked commits. Runtime-free; the reaction passes the token.

The repo must be private: exchange data is not redistributable, so every upload checks that first
and refuses otherwise. The store deduplicates identical bytes, so re-uploading a file is harmless.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

CHUNK_FILES = 1000
CHUNK_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class Commit:
    paths: tuple[str, ...]
    bytes: int
    sha: str | None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunks(
    paths: list[str],
    sizes: dict[str, int],
    max_files: int = CHUNK_FILES,
    max_bytes: int = CHUNK_BYTES,
) -> list[list[str]]:
    """Split paths, in order, into groups under both limits; an oversize file is its own group."""
    out: list[list[str]] = []
    cur: list[str] = []
    total = 0
    for p in paths:
        if cur and (len(cur) >= max_files or total + sizes[p] > max_bytes):
            out.append(cur)
            cur, total = [], 0
        cur.append(p)
        total += sizes[p]
    if cur:
        out.append(cur)
    return out


def assert_private(repo: str, token: str) -> None:
    from huggingface_hub import HfApi

    info = HfApi(token=token).repo_info(repo, repo_type="dataset")
    if not info.private:
        raise RuntimeError(f"{repo} is not private; refusing to upload exchange data")


def upload(root: Path, paths: list[str], repo: str, token: str, message: str) -> list[Commit]:
    """Commit the files at root/<path> to the dataset repo, in chunks; one Commit per chunk."""
    from huggingface_hub import CommitOperationAdd, HfApi

    assert_private(repo, token)
    api = HfApi(token=token)
    sizes = {p: (root / p).stat().st_size for p in paths}
    groups = chunks(sorted(paths), sizes)
    out: list[Commit] = []
    for i, grp in enumerate(groups, 1):
        part = f", part {i}/{len(groups)}" if len(groups) > 1 else ""
        info = api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(root / p)) for p in grp
            ],
            commit_message=f"{message} ({len(grp)} files{part})",
        )
        out.append(Commit(tuple(grp), sum(sizes[p] for p in grp), getattr(info, "oid", None)))
    return out
