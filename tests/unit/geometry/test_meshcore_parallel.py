from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from phydrax._meshcore import (
    exact_orient2d,
    meshcore_available,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
    NativeParallelExecution,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(not meshcore_available(), reason="native meshcore is unavailable"),
]


def test_parallel_policy_restores_after_expected_python_failure() -> None:
    with pytest.raises(ValueError, match="intentional consumer failure"):
        with NativeParallelExecution(7):
            raise ValueError(
                "consumer.body must finish; got intentional consumer failure"
            )
    with NativeParallelExecution(1):
        sign = exact_orient2d((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    np.testing.assert_array_equal(sign, np.asarray(1, dtype=np.int8))


def test_parallel_policy_requires_its_creating_thread() -> None:
    policy = NativeParallelExecution(2)
    with policy:
        with ThreadPoolExecutor(max_workers=1) as pool:
            attempted = pool.submit(policy.__exit__, None, None, None)
            with pytest.raises(RuntimeError, match=r"NativeParallelExecution\.lifecycle"):
                attempted.result()


def test_parallel_policy_context_is_single_use() -> None:
    policy = NativeParallelExecution(2)
    with policy:
        with pytest.raises(RuntimeError, match=r"NativeParallelExecution\.lifecycle"):
            policy.__enter__()
    with pytest.raises(RuntimeError, match=r"NativeParallelExecution\.lifecycle"):
        policy.__enter__()


def test_parallel_policy_tiny_deadline_is_timeout_not_invalid_limits() -> None:
    limits = NativeExecutionLimits(0, 0, 0, 1 << 20, np.inf)
    scope = NativeExecutionScope(limits)
    with pytest.raises(MeshcoreError) as captured:
        with NativeParallelExecution(7, np.finfo(np.float64).tiny):
            with scope:
                scope.remaining()
    assert captured.value.status is MeshcoreStatus.TIMEOUT
    assert scope.completed_evidence().status is MeshcoreStatus.TIMEOUT


@pytest.mark.parametrize("threads", (1, 2, 7, 64))
def test_parallel_policy_equal_batch_signs_and_receipts(threads: int) -> None:
    count = 193
    first = np.zeros((count, 2), dtype=np.float64)
    second = np.tile(np.asarray((1.0, 0.0), dtype=np.float64), (count, 1))
    third = np.tile(np.asarray((0.0, 1.0), dtype=np.float64), (count, 1))
    third[::3] = 0.0
    receipts = []
    results = []
    for workers in (1, threads):
        scope = NativeExecutionScope(NativeExecutionLimits(0, 0, 0, 1 << 20, np.inf))
        with NativeParallelExecution(workers):
            with scope:
                results.append(exact_orient2d(first, second, third))
        evidence = scope.completed_evidence()
        receipts.append(
            (
                evidence.status,
                evidence.work_counters.items(),
                evidence.memory_counters.items(),
                evidence.native_primitive_queries,
                evidence.host_storage_peak_bytes_upper,
            )
        )
    np.testing.assert_array_equal(results[0], results[1])
    assert receipts[0] == receipts[1]
