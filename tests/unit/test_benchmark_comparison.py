#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from benchmarks._comparison import compare_performance, PerformancePolicy
from benchmarks._runtime import DurationDistribution


def _constant(value: float, count: int = 8) -> DurationDistribution:
    return DurationDistribution((value,) * count)


def test_identical_performance_does_not_regress():
    result = compare_performance(
        _constant(1.0),
        _constant(1.0),
        PerformancePolicy(objective="minimize", relative_tolerance=0.05),
        comparison_id="identical",
    )

    assert result.sufficient_samples
    assert result.absolute_degradation == 0.0
    assert result.relative_degradation == 0.0
    assert result.regressed is False


def test_latency_and_throughput_objectives_orient_degradation_consistently():
    latency = compare_performance(
        _constant(1.0),
        _constant(1.2),
        PerformancePolicy(objective="minimize", relative_tolerance=0.1),
        comparison_id="latency",
    )
    throughput = compare_performance(
        _constant(100.0),
        _constant(80.0),
        PerformancePolicy(objective="maximize", relative_tolerance=0.1),
        comparison_id="throughput",
    )

    assert latency.regressed is True
    assert latency.relative_degradation == pytest.approx(0.2)
    assert throughput.regressed is True
    assert throughput.relative_degradation == pytest.approx(0.2)


def test_practical_threshold_boundary_is_not_a_regression():
    result = compare_performance(
        _constant(1.0),
        _constant(1.1),
        PerformancePolicy(objective="minimize", relative_tolerance=0.1),
        comparison_id="boundary",
    )

    assert result.relative_interval == pytest.approx((0.1, 0.1))
    assert result.regressed is False


def test_zero_baseline_requires_or_uses_absolute_tolerance():
    absolute = compare_performance(
        _constant(0.0),
        _constant(0.2),
        PerformancePolicy(
            objective="minimize",
            relative_tolerance=0.1,
            absolute_tolerance=0.1,
        ),
        comparison_id="zero-absolute",
    )
    relative_only = compare_performance(
        _constant(0.0),
        _constant(0.2),
        PerformancePolicy(objective="minimize", relative_tolerance=0.1),
        comparison_id="zero-relative",
    )

    assert absolute.relative_degradation is None
    assert absolute.regressed is True
    assert relative_only.regressed is None
    assert "absolute" in relative_only.reason


def test_insufficient_samples_return_no_decision():
    result = compare_performance(
        _constant(1.0, count=2),
        _constant(2.0, count=2),
        PerformancePolicy(
            objective="minimize",
            relative_tolerance=0.1,
            minimum_samples=3,
        ),
        comparison_id="insufficient",
    )

    assert not result.sufficient_samples
    assert result.regressed is None
    assert "insufficient" in result.reason


def test_bootstrap_is_deterministic_for_one_comparison_identity():
    baseline = DurationDistribution((0.9, 1.0, 1.1, 1.0, 1.05, 0.95))
    candidate = DurationDistribution((1.0, 1.2, 1.15, 1.1, 1.25, 1.05))
    policy = PerformancePolicy(
        objective="minimize",
        relative_tolerance=0.05,
        bootstrap_resamples=500,
    )

    first = compare_performance(
        baseline,
        candidate,
        policy,
        comparison_id="deterministic",
    )
    second = compare_performance(
        baseline,
        candidate,
        policy,
        comparison_id="deterministic",
    )

    assert first.absolute_interval == second.absolute_interval
    assert first.relative_interval == second.relative_interval


def test_pairing_requires_complete_unique_matching_ids():
    baseline = _constant(1.0, count=5)
    candidate = _constant(1.2, count=5)
    policy = PerformancePolicy(objective="minimize", relative_tolerance=0.1)
    ids = tuple(f"sample-{index}" for index in range(5))

    paired = compare_performance(
        baseline,
        candidate,
        policy,
        comparison_id="paired",
        baseline_pair_ids=ids,
        candidate_pair_ids=ids,
    )
    assert paired.paired

    with pytest.raises(ValueError, match="Both"):
        compare_performance(
            baseline,
            candidate,
            policy,
            comparison_id="half-paired",
            baseline_pair_ids=ids,
        )
    with pytest.raises(ValueError, match="match exactly"):
        compare_performance(
            baseline,
            candidate,
            policy,
            comparison_id="mismatch",
            baseline_pair_ids=ids,
            candidate_pair_ids=tuple(reversed(ids)),
        )
    duplicates = ("same",) * 5
    with pytest.raises(ValueError, match="unique"):
        compare_performance(
            baseline,
            candidate,
            policy,
            comparison_id="duplicates",
            baseline_pair_ids=duplicates,
            candidate_pair_ids=duplicates,
        )


def test_policy_rejects_invalid_threshold_and_sampling_contracts():
    with pytest.raises(ValueError, match="At least one"):
        PerformancePolicy(objective="minimize")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        PerformancePolicy(objective="minimize", relative_tolerance=-0.1)
    with pytest.raises(ValueError, match="strictly between"):
        PerformancePolicy(
            objective="minimize",
            relative_tolerance=0.1,
            confidence=1.0,
        )
    with pytest.raises(ValueError, match="must be positive"):
        PerformancePolicy(
            objective="minimize",
            relative_tolerance=0.1,
            bootstrap_resamples=0,
        )
