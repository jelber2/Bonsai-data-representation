"""Thread-pool replacement for mpi_wrapper.py.

Designed for Python 3.14 with free-threaded mode (no GIL), where threading
is truly parallel for CPU-bound work.  Falls back gracefully to sequential
execution on older Python builds.

Usage
-----
    import bonsai_multiproc.mp_wrapper as mp_wrapper
    mp_wrapper.mp_init(n_workers=8)

Interface is intentionally compatible with mpi_wrapper so callers need
minimal changes.  In this model there is always exactly one "rank" (0);
parallelism is achieved by calling map_parallel() from within parallel
sections rather than running N copies of the whole program.
"""

import concurrent.futures
import os
from collections import namedtuple

import numpy as np

_n_workers: int = 1


def mp_init(n_workers: int | None = None) -> None:
    """Initialise the thread pool size.  Call once at program start."""
    global _n_workers
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    _n_workers = max(1, n_workers)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def get_n_workers() -> int:
    return _n_workers


# ── mpi_wrapper-compatible shims ──────────────────────────────────────────────

def get_mpi_info(singleProcess: bool = False):
    """Always returns rank=0, size=1.

    There is only one virtual process (the main thread).  True parallelism
    is achieved via map_parallel(), which internally uses _n_workers threads.
    Returning size=1 here ensures that legacy getMyTaskNumbers / rank-guard
    branches see a single-process view and don't accidentally skip work.
    """
    MpInfo = namedtuple("MpInfo", ['rank', 'size'])
    return MpInfo(0, 1)


def get_process_rank() -> int:
    return 0


def get_process_size() -> int:
    return 1


def is_first_process() -> bool:
    return True


def world_allgather(data):
    return [data]


def gather(data, root: int = 0):
    return [data]


def Gather(data, root: int = 0):
    return [data]


def GatherNpUnknownSize(data, root: int = 0):
    return data


def Bcast(data, root: int = 0, type: str = 'double'):
    return data


def bcast(data, root: int = 0):
    return data


def barrier() -> None:
    pass


# ── Parallel execution ─────────────────────────────────────────────────────────

def map_parallel(fn, tasks: list) -> list:
    """Run fn(task) for every task in parallel; return results in task order.

    Uses a ThreadPoolExecutor so all threads share memory — ideal for
    Python 3.14 free-threaded builds where the GIL is disabled and threads
    achieve true CPU parallelism without pickling overhead.

    Falls back to a plain list-comprehension when n_workers == 1 or
    len(tasks) <= 1 to avoid thread-pool overhead for small workloads.
    """
    n = len(tasks)
    if n == 0:
        return []
    workers = min(_n_workers, n)
    if workers <= 1:
        return [fn(t) for t in tasks]

    results: list = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(fn, t): i for i, t in enumerate(tasks)}
        for f in concurrent.futures.as_completed(future_to_idx):
            results[future_to_idx[f]] = f.result()
    return results
