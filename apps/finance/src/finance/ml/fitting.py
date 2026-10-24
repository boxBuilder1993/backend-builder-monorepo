"""Fitting in a subprocess: a CUDA fault or an out-of-memory kills the child, not the worker. The
parent passes paths and plain data; the child completes the schema from the file, streams it in
chunks of ``batch_rows`` to the adapter, saves the directory, and returns what the Model row needs.
The whole matrix is never in memory: a 20-million-row fit is a chunk plus the estimator's own
state (xgboost's quantised copy, the ridge's Gram matrix)."""

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
    batch_rows: int = data.FIT_BATCH_ROWS


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
    path = Path(job.rows_path)
    schema = data.complete_schema(path, Schema.from_dict(job.schema))
    rows = data.count_rows(path)
    adapter = E.REGISTRY[job.algorithm]
    device = E.pick_device(adapter, job.device)

    def batches() -> Any:
        return data.iter_batches(path, schema, job.batch_rows)

    try:
        fitted = adapter.fit_batches(batches, schema, job.params, device, job.seed)
    except Exception as exc:
        if device == "cpu" or not cuda_out_of_memory(exc):
            raise
        # the GPU could not hold the quantised matrix (an 8 GB card against a 20-million-row
        # fit); the same fit on the CPU is slower and never short of memory
        device = "cpu"
        fitted = adapter.fit_batches(batches, schema, job.params, device, job.seed)
    out = Path(job.out_dir)
    info = dict(
        job.info,
        algorithm=job.algorithm,
        params=job.params,
        device=device,
        seed=job.seed,
        rows=rows,
        batch_rows=job.batch_rows,
    )
    artifacts.save(out, fitted, schema, info)
    sha, size = artifacts.archive_hash(out)
    return FitDone(schema.to_dict(), device, time.time() - t0, rows, sha, size)


def cuda_out_of_memory(exc: BaseException) -> bool:
    text = str(exc)
    return "cudaErrorMemoryAllocation" in text or "CUDA out of memory" in text


def fit_in_subprocess(job: FitJob) -> FitDone:
    """Synchronous; the reaction runs it in a thread. One child per fit, spawned fresh."""
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        return pool.submit(fit_job, job).result()


def job_dict(job: FitJob) -> dict[str, Any]:
    return asdict(job)
