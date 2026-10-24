"""Fitting in a subprocess: a CUDA fault or an out-of-memory kills the child, not the worker. The
parent passes paths and plain data; the child loads the frame, fits, saves the directory, and
returns what the Model row needs."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from finance.ml import artifacts, data
from finance.ml import estimators as E
from finance.ml.schema import Schema


@dataclass(frozen=True)
class FitJob:
    rows_path: str
    schema: dict[str, Any]
    algorithm: str
    params: dict[str, Any]
    device: str
    seed: int
    out_dir: str
    info: dict[str, Any]


@dataclass(frozen=True)
class FitDone:
    schema: dict[str, Any]
    device: str
    seconds: float
    rows: int
    sha256: str
    bytes: int


def fit_job(job: FitJob) -> FitDone:
    """Runs in the child."""
    t0 = time.time()
    schema = Schema.from_dict(job.schema)
    frame = data.read_rows(Path(job.rows_path), schema, fitting=True)
    prepared = data.prepare(frame, schema, fitting=True)
    del frame  # the fit holds X and Y as float32; the frame is not needed again
    adapter = E.REGISTRY[job.algorithm]
    device = E.pick_device(adapter, job.device)
    assert prepared.Y is not None
    fitted = adapter.fit(prepared.X, prepared.Y, prepared.schema, job.params, device, job.seed)
    out = Path(job.out_dir)
    info = dict(job.info, algorithm=job.algorithm, params=job.params, device=device, seed=job.seed)
    artifacts.save(out, fitted, prepared.schema, info)
    sha, size = artifacts.archive_hash(out)
    return FitDone(
        prepared.schema.to_dict(), device, time.time() - t0, prepared.X.shape[0], sha, size
    )


def fit_in_subprocess(job: FitJob) -> FitDone:
    """Synchronous; the reaction runs it in a thread. One child per fit, spawned fresh."""
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        return pool.submit(fit_job, job).result()


def job_dict(job: FitJob) -> dict[str, Any]:
    return asdict(job)
