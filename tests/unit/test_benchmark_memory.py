#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np
import pytest

from benchmarks._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
)


def test_memory_gate_decides_at_the_largest_capacity() -> None:
    policy = MemoryGatePolicy()
    capacities = (64, 128, 256)

    fused = evaluate_memory_gate(
        capacities, (4_000, 8_000, 16_000), (1_000, 4_000, 9_000), policy
    )
    bounded = evaluate_memory_gate(
        capacities, (4_000, 16_000, 64_000), (4_000, 8_000, 16_000), policy
    )
    exact_fraction = evaluate_memory_gate(capacities, (10, 20, 40), (40, 40, 20), policy)

    assert fused.decision == "retain-baseline"
    assert fused.capacity == 256
    assert bounded.decision == "adopt"
    assert (bounded.baseline_bytes, bounded.candidate_bytes) == (64_000, 16_000)
    assert exact_fraction.decision == "adopt"
    assert (
        evaluate_memory_gate(capacities, (0, 0, 0), (0, 0, 0), policy).decision
        == "retain-baseline"
    )


@pytest.mark.parametrize(
    ("capacities", "baseline", "candidate"),
    [
        pytest.param((64,), (1,), (1,), id="single-capacity"),
        pytest.param((64, 64), (1, 2), (1, 2), id="non-increasing"),
        pytest.param((64, 128), (1, 2), (1,), id="length-mismatch"),
        pytest.param((64, 128), (1, -2), (1, 2), id="negative-bytes"),
    ],
)
def test_memory_gate_refuses_malformed_series(
    capacities: tuple[int, ...],
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        evaluate_memory_gate(capacities, baseline, candidate, MemoryGatePolicy())


def test_fitted_scaling_exponent_recovers_power_laws() -> None:
    capacities = (16, 32, 64, 128)

    assert fitted_scaling_exponent(
        capacities, tuple(8 * n for n in capacities)
    ) == pytest.approx(1.0)
    assert fitted_scaling_exponent(
        capacities, tuple(8 * n * n for n in capacities)
    ) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        fitted_scaling_exponent(capacities, (0, 1, 2, 3))


def test_host_allocation_peak_observes_released_scratch() -> None:
    scratch_bytes = 8 * 1024 * 1024

    def operation() -> float:
        scratch = np.ones(scratch_bytes // 8, dtype=np.float64)
        return float(scratch.sum())

    value, peak = host_allocation_peak(operation)

    assert value == scratch_bytes // 8
    assert peak >= scratch_bytes
