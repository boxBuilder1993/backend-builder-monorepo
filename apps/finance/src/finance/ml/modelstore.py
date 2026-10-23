"""Models in the store: upload a directory to models/<model id>/ and fetch one back by
(repo, path, revision). Runtime-free; the reaction passes the token."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from pathlib import Path


def model_path(model_id: str) -> str:
    return f"models/{model_id}"


def upload(repo: str, token: str, model_id: str, directory: Path) -> str:
    """Upload the directory; returns the revision the store produced. Private repos only."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise RuntimeError(f"{repo} is not private; refusing to upload")
    info = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(directory),
        path_in_repo=model_path(model_id),
        commit_message=f"model {model_id}",
    )
    oid = getattr(info, "oid", None)
    if not oid:
        raise RuntimeError("store returned no revision for the upload")
    return str(oid)


def fetch(repo: str, token: str, path: str, revision: str, dest: Path) -> Path:
    """Bring models/<id>/ at revision into dest; returns the directory. Present files skipped."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=dest,
        allow_patterns=[f"{path}/**"],
        token=token,
    )
    return dest / path
