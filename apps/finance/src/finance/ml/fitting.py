"""Fitting in a subprocess: a CUDA fault or an out-of-memory kills the child, not the worker. The
parent passes paths and plain data; the child completes the schema from the file, streams it in
chunks of ``batch_rows`` to the adapter, saves the directory, and returns what the Model row needs.
The whole matrix is never in memory: a 20-million-row fit is a chunk plus the estimator's own
state (xgboost's quantised copy, the ridge's Gram matrix)."""

from __future__ import annotations

import resource
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
    peak_rss_mb: int


INCORE_BYTES_PER_CELL = 4.5
"""What xgboost's in-core quantised matrix costs on the host while it is built from chunks: measured
at 320 MB per million rows of 88 features on 2026-11-05 (the second pass over the chunks), four
times the byte per cell the finished index holds."""

INCORE_BUDGET = 8 * 1024**3
"""The host memory a GPU attempt may spend on that matrix: the card holds 8 GB, so a matrix past
this never trains on it anyway, and building it (20 GB at 40 million rows) is what took the worker."""


def gpu_fits(rows: int, features: int, budget: int | None = None) -> bool:
    """Whether a fit of this size may try the GPU: its in-core matrix must stay within ``budget``
    (the module's ``INCORE_BUDGET`` by default)."""
    limit = INCORE_BUDGET if budget is None else budget
    return rows * features * INCORE_BYTES_PER_CELL <= limit


def fit_job(job: FitJob) -> FitDone:
    """Runs in the child."""
    t0 = time.time()
    path = Path(job.rows_path)
    schema = data.complete_schema(path, Schema.from_dict(job.schema))
    rows = data.count_rows(path)
    adapter = E.REGISTRY[job.algorithm]
    device = E.pick_device(adapter, job.device)
    if device == "cuda" and job.device == "auto" and not gpu_fits(rows, len(schema.feature_names)):
        # too big for the card, and the host matrix the attempt would build is what the box
        # cannot hold either: straight to the CPU, which pages its matrix through the disk
        device = "cpu"

    def batches() -> Any:
        return data.iter_batches(path, schema, job.batch_rows)

    try:
        fitted = adapter.fit_batches(batches, schema, job.params, device, job.seed)
    except Exception as exc:
        if device == "cpu" or not cuda_out_of_memory(exc):
            raise
        # the GPU could not hold the quantised matrix (an 8 GB card against a 20-million-row
        # fit); the same fit on the CPU is slower and pages its matrix through the disk. The retry
        # is outside this block so the failed attempt's matrix is freed first: a traceback
        # keeps its frames alive.
        device = "cpu"
        fitted = None
    if fitted is None:
        fitted = adapter.fit_batches(batches, schema, job.params, device, job.seed)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    out = Path(job.out_dir)
    info = dict(
        job.info,
        algorithm=job.algorithm,
        params=job.params,
        device=device,
        seed=job.seed,
        rows=rows,
        batch_rows=job.batch_rows,
        peak_rss_mb=peak_mb,
    )
    artifacts.save(out, fitted, schema, info)
    sha, size = artifacts.archive_hash(out)
    return FitDone(schema.to_dict(), device, time.time() - t0, rows, sha, size, peak_mb)


def cuda_out_of_memory(exc: BaseException) -> bool:
    text = str(exc)
    return "cudaErrorMemoryAllocation" in text or "CUDA out of memory" in text


def fit_in_subprocess(job: FitJob) -> FitDone:
    """Synchronous; the reaction runs it in a thread. One child per fit, spawned fresh."""
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        return pool.submit(fit_job, job).result()


def job_dict(job: FitJob) -> dict[str, Any]:
    return asdict(job)
