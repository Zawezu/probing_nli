"""Helpers for running independent experiment combinations in parallel.

Each (model, language) / (model, language_pair) combination in experiments 1 and 2
produces its own ExperimentResult(s) and is fully independent of the others (probe
training is deterministic via fixed seeds and depends only on that combination's
data). They can therefore be run in separate processes for a near-linear speedup
without changing any results.

Process-based parallelism (rather than threads) is used because the work is CPU-bound
in NumPy/scikit-learn, which hold the GIL for parts of the computation. To avoid
oversubscribing the CPU, each worker caps the number of BLAS/OpenMP threads it uses so
that ``num_workers * threads_per_worker`` stays close to the core count.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Holds the threadpool-limit controller for the lifetime of a worker process so the
# limits are not garbage-collected (which would restore the original thread counts).
_thread_limit_controller = None


def resolve_num_workers(requested: int | None, num_tasks: int) -> int:
    """Resolve the effective number of worker processes.

    Args:
        requested: User-requested worker count. None or <= 0 means "auto" (use as many
            workers as there are tasks, capped at the CPU count).
        num_tasks: Number of independent tasks to run.

    Returns:
        A worker count in the range [1, num_tasks].
    """
    num_tasks = max(1, num_tasks)
    cpu_count: int = os.cpu_count() or 1
    if requested is None or requested <= 0:
        return min(num_tasks, cpu_count)
    return max(1, min(requested, num_tasks))


def _init_worker(threads_per_worker: int) -> None:
    """Worker initialiser: cap BLAS/OpenMP threads for this process.

    threadpoolctl is a transitive dependency of scikit-learn, so it is expected to be
    available. If it is missing for any reason, the limit is silently skipped.
    """
    global _thread_limit_controller
    try:
        import threadpoolctl

        _thread_limit_controller = threadpoolctl.threadpool_limits(
            limits=threads_per_worker
        )
    except Exception:
        # Capping threads is a performance optimisation, not a correctness requirement.
        pass


def map_combinations(
    worker: Callable[[T], R],
    combinations: Sequence[T],
    num_workers: int | None,
) -> list[R]:
    """Run ``worker`` over every combination, optionally in parallel, preserving order.

    Args:
        worker: A picklable, module-level function taking one combination and returning
            its result. Must be importable in a fresh process (no closures/lambdas).
        combinations: The list of independent combinations to process.
        num_workers: Requested worker count (see ``resolve_num_workers``). A resolved
            value of 1 runs everything in-process, which keeps behaviour identical to the
            original sequential code and eases debugging.

    Returns:
        Results in the same order as ``combinations``.
    """
    effective_workers: int = resolve_num_workers(num_workers, len(combinations))

    if effective_workers <= 1 or len(combinations) <= 1:
        return [worker(combo) for combo in combinations]

    cpu_count: int = os.cpu_count() or 1
    threads_per_worker: int = max(1, cpu_count // effective_workers)

    print(
        f"Running {len(combinations)} combinations across {effective_workers} worker "
        f"processes ({threads_per_worker} BLAS thread(s) each)."
    )

    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_init_worker,
        initargs=(threads_per_worker,),
    ) as executor:
        return list(executor.map(worker, combinations))
