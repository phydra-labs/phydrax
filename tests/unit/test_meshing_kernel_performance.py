# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn

import numpy as np
import pytest

from phydrax import _meshcore as mc
from tools.meshing_qualification import kernel_performance as kernels


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not mc.meshcore_available(), reason="native meshcore is unavailable"
    ),
]

# One logical 64-item chunk: small enough for the declared 1 GiB allowance.
_SMALL_BATCH = 64
# A managed allowance below any predicate arena; work, queries and cavity unlimited.
_TINY_LIMITS = mc.NativeExecutionLimits(0, 0, 0, 1024, 120.0)


def _declared_refusal(message: str) -> mc.MeshcoreError:
    error = mc.MeshcoreError(mc.MeshcoreStatus.CAPACITY_EXCEEDED, message)
    error.batch_requested_upper = (0, 0, 0, 2 << 30)
    return error


@pytest.mark.parametrize("kernel", ("exact_insphere", "point_triangle_locations"))
def test_native_batch_within_declared_allowance_is_a_timing(kernel: str) -> None:
    call = dict(kernels._cases(_SMALL_BATCH))[kernel]
    sample, payload = kernels._sample(
        call, 1, kernel=kernel, batch_count=_SMALL_BATCH, limits=kernels._BATCH_LIMITS
    )

    assert payload is not None
    assert sample["status"] == "OK"
    assert sample["timing_is_headline"] is True
    assert sample["seconds"] > 0.0


@pytest.mark.parametrize("kernel", ("exact_insphere", "symbolic_insphere"))
def test_native_batch_above_declared_allowance_is_refused_without_timing(
    kernel: str,
) -> None:
    call = dict(kernels._cases(_SMALL_BATCH))[kernel]
    sample, payload = kernels._sample(
        call, 1, kernel=kernel, batch_count=_SMALL_BATCH, limits=_TINY_LIMITS
    )

    assert payload is None
    assert sample["status"] == "resource-refused"
    assert sample["seconds"] is None
    assert sample["timing_is_headline"] is False
    assert sample["native_scope_limits"]["bytes"] == 1024
    refusal = sample["refusal"]
    assert refusal["exceeded_limits"] == ["bytes"]
    assert refusal["batch_requested_upper"][3] > 1024
    assert sample["native_evidence"]["phases"][0]["status_name"] == "CAPACITY_EXCEEDED"


@pytest.mark.parametrize(
    "error",
    (
        mc.MeshcoreError(
            mc.MeshcoreStatus.TIMEOUT, "insphere.deadline must not expire; got expiry"
        ),
        mc.MeshcoreError(
            mc.MeshcoreStatus.CAPACITY_EXCEEDED,
            "insphere.inputs must fit the managed allowance; got no declared demand",
        ),
        mc.MeshcoreError(
            mc.MeshcoreStatus.INTERNAL_ERROR,
            "insphere.kernel must return a predicate sign; got a native defect",
        ),
        ValueError("insphere.inputs must be valid; got a consumer defect"),
    ),
    ids=("timeout", "unexplained-capacity", "internal", "consumer"),
)
def test_unexplained_sample_failure_propagates(error: Exception) -> None:
    def failed() -> NoReturn:
        raise error

    with pytest.raises(type(error)) as captured:
        kernels._sample(
            failed,
            1,
            kernel="exact_insphere",
            batch_count=20000,
            limits=kernels._BATCH_LIMITS,
        )
    assert captured.value is error


@pytest.mark.parametrize("refused", (False, True))
def test_batch_campaign_preserves_repetitions_and_refuses_positive_claims(
    monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    calls = 0

    def call() -> np.ndarray:
        nonlocal calls
        calls += 1
        if refused:
            raise _declared_refusal(
                "insphere.inputs must fit the managed allowance; got a bounded sample"
            )
        return np.asarray((1,), dtype=np.int8)

    def cases(count: int) -> list[tuple[str, Callable[[], object]]]:
        assert count == 20000
        return [("exact_insphere", call)]

    monkeypatch.setattr(kernels, "_cases", cases)
    monkeypatch.setattr(kernels, "_runtime_identity", lambda: {})
    monkeypatch.setattr(kernels.os, "sched_getaffinity", lambda _: set(range(32)))
    report = kernels.benchmark_parallel_batches(2, 5)

    assert calls == 3 * (1 + 5)
    assert report["batch_count"] == 20000
    assert report["repeats"] == 5
    assert report["status"] == ("resource-refused" if refused else "completed")
    assert report["timing_is_headline"] is not refused
    samples = report["samples"]
    assert isinstance(samples, list)
    assert len(samples) == 1
    candidates = samples[0]["candidates"]
    assert [candidate["requested_threads"] for candidate in candidates] == [1, 8, 32]
    for candidate in candidates:
        assert len(candidate["warm_samples"]) == 5
        if refused:
            assert candidate["status"] == "resource-refused"
            assert candidate["warm_median_seconds"] is None
            assert candidate["speedup"] is None
            assert candidate["bitwise_equivalent"] is None
            assert candidate["threshold_met"] is False
            assert candidate["timing_is_headline"] is False
        else:
            assert candidate["status"] == "completed"
            assert candidate["warm_median_seconds"] is not None
            assert candidate["speedup"] is not None
            assert candidate["bitwise_equivalent"] is True
    if refused:
        assert samples[0]["deterministic_evidence"] is None


def test_batch_campaign_still_rejects_result_bit_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def call() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray((calls,), dtype=np.int8)

    def cases(count: int) -> list[tuple[str, Callable[[], object]]]:
        assert count == 20000
        return [("exact_orient2d", call)]

    monkeypatch.setattr(kernels, "_cases", cases)
    monkeypatch.setattr(kernels.os, "sched_getaffinity", lambda _: set(range(32)))
    with pytest.raises(ValueError, match="deterministic_evidence"):
        kernels.benchmark_parallel_batches(2, 5)
    assert calls == 2


def test_refused_serial_warm_sample_never_enables_parallel_speedup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def call() -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _declared_refusal(
                "insphere.inputs must fit the managed allowance; got a bounded warm sample"
            )
        return np.asarray((1,), dtype=np.int8)

    def cases(count: int) -> list[tuple[str, Callable[[], object]]]:
        assert count == 20000
        return [("exact_insphere", call)]

    monkeypatch.setattr(kernels, "_cases", cases)
    monkeypatch.setattr(kernels, "_runtime_identity", lambda: {})
    monkeypatch.setattr(kernels.os, "sched_getaffinity", lambda _: set(range(32)))
    report = kernels.benchmark_parallel_batches(2, 5)

    assert calls == 18
    assert report["status"] == "resource-refused"
    assert report["timing_is_headline"] is False
    samples = report["samples"]
    assert isinstance(samples, list)
    for candidate in samples[0]["candidates"]:
        assert candidate["speedup"] is None
        assert candidate["bitwise_equivalent"] is None
        assert candidate["threshold_met"] is False
        assert candidate["timing_is_headline"] is False
