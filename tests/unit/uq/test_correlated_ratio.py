#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent finite-record references; no statistical random fixtures."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.uq import (
    correlated_ratio_of_means,
    CorrelatedRatioPolicy,
    CorrelatedRatioResult,
    CorrelatedRatioStatus,
)
from phydrax.uq._correlated_ratio import _fieller_set, DenominatorSign, FiellerSet


def _analyze(
    x: NDArray[np.float64] | NDArray[np.complex128],
    y: NDArray[np.float64] | NDArray[np.complex128],
    *,
    valid: NDArray[np.bool_] | None = None,
    policy: CorrelatedRatioPolicy | None = None,
    deterministic: bool = False,
    deterministic_ratio: float | complex | None = None,
) -> CorrelatedRatioResult:
    return correlated_ratio_of_means(
        x,
        y,
        policy=CorrelatedRatioPolicy(max_lag=1) if policy is None else policy,
        stream_ids=("stream-a",),
        dependence_ids=("origin-a",),
        sampling_origin_id="manufactured-finite-records",
        valid=valid,
        deterministic=deterministic,
        deterministic_ratio=deterministic_ratio,
    )


def _alternating() -> NDArray[np.float64]:
    return np.asarray([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0], dtype=np.float64)


def test_joint_cross_covariance_changes_real_ratio_uncertainty() -> None:
    a = _alternating()
    x = (10.0 + a)[None, :]
    y = (20.0 + 0.1 * a + np.asarray([0.1, -0.1] * 4, dtype=np.float64))[None, :]
    result = _analyze(x, y)
    joint = np.cov(np.stack((x[0], y[0])), ddof=1) / x.shape[1]
    gradient = np.asarray([1.0 / 20.0, -10.0 / 400.0], dtype=np.float64)
    expected = gradient @ joint @ gradient
    assert bool(result.statistically_valid)
    assert result.fieller_set == "bounded"
    np.testing.assert_allclose(result.mean_covariance, joint, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(
        result.ratio_covariance, [[expected]], rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(result.value, 0.5, rtol=1e-14)
    diagonal_only = gradient @ np.diag(np.diag(joint)) @ gradient
    assert expected < diagonal_only
    assert not np.isclose(expected, diagonal_only, rtol=0.01)


def test_full_complex_joint_covariance_and_division_reference() -> None:
    amplitudes = (
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 5.0, 6.0],
                [3.0, 5.0, 9.0, 8.0],
                [4.0, 6.0, 8.0, 16.0],
            ],
            dtype=np.float64,
        )
        * 0.1
    )
    deviations = np.stack((amplitudes, -amplitudes), axis=1).reshape((8, 4))
    center = np.asarray([10.0, 3.0, 20.0, 10.0], dtype=np.float64)
    channels = center[None, :] + deviations
    x = (channels[:, 0] + 1j * channels[:, 1])[None, :]
    y = (channels[:, 2] + 1j * channels[:, 3])[None, :]
    result = _analyze(x, y)
    joint = np.cov(channels, rowvar=False, ddof=1) / channels.shape[0]
    # Independent real differential of complex division.
    derivative_x = 1.0 / (20.0 + 10.0j)
    derivative_y = -(10.0 + 3.0j) / (20.0 + 10.0j) ** 2
    derivatives = np.asarray(
        [derivative_x, 1j * derivative_x, derivative_y, 1j * derivative_y],
        dtype=np.complex128,
    )
    jacobian = np.stack((derivatives.real, derivatives.imag))
    assert bool(result.statistically_valid)
    assert result.confidence_kind == "complex-origin-ball"
    assert result.fieller_set == "not-applicable"
    np.testing.assert_allclose(result.mean_covariance, joint, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        result.ratio_covariance, jacobian @ joint @ jacobian.T, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(result.value, (10.0 + 3.0j) / (20.0 + 10.0j), rtol=1e-14)
    expected_radius = result.policy.confidence_multiplier * math.sqrt(
        float(np.trace(joint[2:, 2:]))
    )
    np.testing.assert_allclose(
        result.denominator_confidence_radius, expected_radius, rtol=1e-13
    )


@pytest.mark.parametrize(
    "numerator_complex", [False, True], ids=["real-x-complex-y", "complex-x-real-y"]
)
def test_mixed_dtype_uses_three_real_channels(numerator_complex: bool) -> None:
    a = _alternating()
    if numerator_complex:
        x = (10.0 + a + 1j * (3.0 + 0.2 * a))[None, :]
        y = (20.0 + 0.1 * a)[None, :]
        channels = np.stack((x[0].real, x[0].imag, y[0]), axis=1)
    else:
        x = (10.0 + a)[None, :]
        y = (20.0 + 0.1 * a + 1j * (3.0 + 0.2 * a))[None, :]
        channels = np.stack((x[0], y[0].real, y[0].imag), axis=1)
    result = _analyze(x, y)
    assert bool(result.statistically_valid)
    np.testing.assert_allclose(
        result.mean_covariance,
        np.cov(channels, rowvar=False, ddof=1) / 8.0,
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(result.value, np.mean(x) / np.mean(y), rtol=1e-14)
    assert result.ratio_covariance.shape == (2, 2)


def test_complete_batch_covariance_discards_incomplete_tail() -> None:
    x = np.asarray(
        [[21.0, 1.0, 22.0, 2.0, 19.0, -1.0, 18.0, -2.0, 999.0]], dtype=np.float64
    )
    y = np.full(x.shape, 50.0, dtype=np.float64)
    result = _analyze(
        x, y, policy=CorrelatedRatioPolicy(max_lag=1, block_length=2, minimum_blocks=2)
    )
    batches = np.asarray(
        [[11.0, 50.0], [12.0, 50.0], [9.0, 50.0], [8.0, 50.0]], dtype=np.float64
    )
    expected = np.cov(batches, rowvar=False, ddof=1) / 4.0
    assert int(result.discarded_records) == 1
    np.testing.assert_array_equal(result.retained, [[True] * 8 + [False]])
    np.testing.assert_array_equal(result.block_index, [[0, 0, 1, 1, 2, 2, 3, 3, -1]])
    np.testing.assert_allclose(result.mean_covariance, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(result.numerator_mean, 10.0, rtol=1e-14)


def test_dependent_streams_form_synchronous_joint_block_vectors() -> None:
    a = _alternating()
    x = np.stack((10.0 + a, 14.0 + 2.0 * a))
    y = np.stack((30.0 + 0.1 * a, 50.0 + 0.3 * a))
    result = correlated_ratio_of_means(
        x,
        y,
        policy=CorrelatedRatioPolicy(max_lag=1),
        stream_ids=("a", "b"),
        dependence_ids=("shared-origin", "shared-origin"),
        sampling_origin_id="paired-observations",
    )
    batches = np.stack((np.mean(x, axis=0), np.mean(y, axis=0)), axis=1)
    expected = np.cov(batches, rowvar=False, ddof=1) / 8.0
    assert bool(result.statistically_valid)
    np.testing.assert_allclose(result.mean_covariance, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(result.block_index[0], result.block_index[1])
    np.testing.assert_array_equal(result.block_counts, [8])
    np.testing.assert_allclose(result.value, np.mean(x) / np.mean(y), rtol=1e-14)


@pytest.mark.parametrize("holes", [False, True], ids=["contiguous", "aligned-holes"])
def test_dependent_group_slow_mode_requires_closed_window(holes: bool) -> None:
    fast = np.asarray([2.0, -2.0] * 16, dtype=np.float64)
    slow = np.linspace(-0.1, 0.1, 32, dtype=np.float64)
    x = np.stack((10.0 + fast + slow, 10.0 - fast + slow))
    y = np.full(x.shape, 20.0, dtype=np.float64)
    valid = np.ones(x.shape, dtype=np.bool_)
    if holes:
        valid[0, 8] = False
        valid[1, 19] = False
    result = correlated_ratio_of_means(
        x,
        y,
        policy=CorrelatedRatioPolicy(max_lag=1),
        stream_ids=("a", "b"),
        dependence_ids=("shared", "shared"),
        sampling_origin_id="cancelled-fast-mode",
        valid=valid,
    )
    assert int(result.status) & CorrelatedRatioStatus.CORRELATION_UNRESOLVED
    assert not bool(result.statistically_valid)
    np.testing.assert_array_equal(
        result.diagnostic_window_closed[0], [False, True, False]
    )
    assert math.isnan(float(result.value))
    assert math.isnan(float(result.standard_error))
    if holes:
        np.testing.assert_array_equal(result.retained[:, [8, 19]], False)


def test_dependent_group_multiple_holes_preserve_closed_joint_covariance() -> None:
    # Each aligned run closes at lag one. Hole values must not enter the
    # group averages, and runs must not be concatenated into a new chronology.
    a = np.asarray([1.0, -1.0, 99.0, 2.0, -2.0, 99.0, 3.0, -3.0], dtype=np.float64)
    x = np.stack((10.0 + a, 14.0 + 2.0 * a))
    y = np.stack((30.0 + 0.1 * a, 50.0 + 0.3 * a))
    valid = np.ones(x.shape, dtype=np.bool_)
    valid[0, 2] = False
    valid[1, 5] = False
    result = correlated_ratio_of_means(
        x,
        y,
        policy=CorrelatedRatioPolicy(max_lag=1, minimum_draws=6),
        stream_ids=("a", "b"),
        dependence_ids=("shared", "shared"),
        sampling_origin_id="separate-aligned-runs",
        valid=valid,
    )
    selected = np.all(valid, axis=0)
    batches = np.stack(
        (np.mean(x[:, selected], axis=0), np.mean(y[:, selected], axis=0)), axis=1
    )
    expected = np.cov(batches, rowvar=False, ddof=1) / 6.0
    assert bool(result.statistically_valid)
    np.testing.assert_array_equal(result.retained, np.broadcast_to(selected, x.shape))
    np.testing.assert_array_equal(result.block_index, [[0, 1, -1, 2, 3, -1, 4, 5]] * 2)
    np.testing.assert_allclose(result.mean_covariance, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(
        result.value, np.mean(batches[:, 0]) / np.mean(batches[:, 1]), rtol=1e-14
    )


def test_independent_group_covariance_uses_squared_sample_weights() -> None:
    a = _alternating()
    x = np.stack((10.0 + a, 12.0 + 2.0 * a))
    y = np.full(x.shape, 30.0, dtype=np.float64)
    valid = np.asarray([[True] * 8, [True] * 4 + [False] * 4], dtype=np.bool_)
    result = correlated_ratio_of_means(
        x,
        y,
        policy=CorrelatedRatioPolicy(max_lag=1, minimum_draws=2, minimum_blocks=2),
        stream_ids=("a", "b"),
        dependence_ids=("independent-a", "independent-b"),
        sampling_origin_id="independent-observations",
        valid=valid,
    )
    first = np.stack((x[0], y[0]), axis=1)
    second = np.stack((x[1, :4], y[1, :4]), axis=1)
    expected = (8.0 / 12.0) ** 2 * np.cov(first, rowvar=False, ddof=1) / 8.0
    expected += (4.0 / 12.0) ** 2 * np.cov(second, rowvar=False, ddof=1) / 4.0
    assert bool(result.statistically_valid)
    np.testing.assert_allclose(result.mean_covariance, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(result.group_weights, [2.0 / 3.0, 1.0 / 3.0], rtol=1e-14)
    np.testing.assert_allclose(result.numerator_mean, 128.0 / 12.0, rtol=1e-14)


def test_group_alignment_records_discarded_unpaired_draws() -> None:
    a = _alternating()
    x = np.stack((np.append(10.0 + a, 999.0), np.append(14.0 + 2.0 * a, np.nan)))
    y = np.stack((np.append(30.0 + 0.1 * a, 30.0), np.append(50.0 + 0.3 * a, np.nan)))
    valid = np.asarray([[True] * 9, [True] * 8 + [False]], dtype=np.bool_)
    result = correlated_ratio_of_means(
        x,
        y,
        policy=CorrelatedRatioPolicy(max_lag=1),
        stream_ids=("a", "b"),
        dependence_ids=("shared", "shared"),
        sampling_origin_id="alignment",
        valid=valid,
    )
    assert bool(result.statistically_valid)
    assert int(result.discarded_records) == 1
    np.testing.assert_array_equal(result.retained[:, -1], [False, False])
    np.testing.assert_allclose(result.numerator_mean, 12.0, rtol=1e-14)


def test_masked_nan_padding_preserves_joint_covariance() -> None:
    a = _alternating()
    reference = _analyze((10.0 + a)[None, :], (20.0 + 0.2 * a)[None, :])
    x = np.append(10.0 + a, np.nan)[None, :]
    y = np.append(20.0 + 0.2 * a, np.nan)[None, :]
    result = _analyze(x, y, valid=np.asarray([[True] * 8 + [False]], dtype=np.bool_))
    assert bool(result.statistically_valid)
    np.testing.assert_allclose(
        result.mean_covariance, reference.mean_covariance, rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(result.value, reference.value, rtol=1e-14)


def test_nonfinite_active_tail_is_never_silently_dropped() -> None:
    x = np.append(10.0 + _alternating(), np.nan)[None, :]
    y = np.full(x.shape, 20.0, dtype=np.float64)
    result = _analyze(x, y, policy=CorrelatedRatioPolicy(max_lag=1, block_length=2))
    assert int(result.status) & CorrelatedRatioStatus.NONFINITE
    assert not bool(result.statistically_valid)
    assert math.isnan(float(result.value))
    assert math.isnan(float(result.standard_error))


@pytest.mark.parametrize(
    "explicit_block", [None, 1, 32], ids=["automatic", "short-explicit", "long-explicit"]
)
def test_positive_tail_at_lag_limit_refuses_even_explicit_blocks(
    explicit_block: int | None,
) -> None:
    x = (10.0 + np.linspace(-1.0, 1.0, 32, dtype=np.float64))[None, :]
    y = np.full(x.shape, 20.0, dtype=np.float64)
    result = _analyze(
        x, y, policy=CorrelatedRatioPolicy(max_lag=2, block_length=explicit_block)
    )
    assert int(result.status) & CorrelatedRatioStatus.CORRELATION_UNRESOLVED
    assert not bool(result.statistically_valid)
    assert not bool(result.diagnostic_window_closed[0, 0])
    assert math.isnan(float(result.standard_error))


def test_ratio_influence_controls_window_when_channels_look_fast() -> None:
    fast = np.asarray([2.0, -2.0] * 16, dtype=np.float64)
    slow = np.linspace(-0.1, 0.1, 32, dtype=np.float64)
    x = (10.0 + fast + slow)[None, :]
    y = (20.0 + 2.0 * fast)[None, :]
    result = _analyze(x, y)
    np.testing.assert_array_equal(result.diagnostic_window_closed[0, :2], [True, True])
    assert not bool(result.diagnostic_window_closed[0, 2])
    assert int(result.status) & CorrelatedRatioStatus.CORRELATION_UNRESOLVED
    assert not bool(result.statistically_valid)


def test_constant_stochastic_history_is_not_deterministic_evidence() -> None:
    result = _analyze(
        np.full((1, 8), 6.0, dtype=np.float64), np.full((1, 8), 3.0, dtype=np.float64)
    )
    assert int(result.status) & CorrelatedRatioStatus.ZERO_VARIATION_UNRESOLVED
    np.testing.assert_allclose(result.exploratory_value, 2.0)
    assert not bool(result.statistically_valid)
    assert math.isnan(float(result.value))
    assert math.isnan(float(result.standard_error))


def test_declared_deterministic_source_produces_singleton() -> None:
    result = _analyze(
        np.asarray([[6.0]], dtype=np.float64),
        np.asarray([[3.0]], dtype=np.float64),
        deterministic=True,
    )
    assert bool(result.statistically_valid)
    assert result.fieller_set == "singleton"
    np.testing.assert_allclose(result.value, 2.0)
    np.testing.assert_array_equal(
        result.mean_covariance, np.zeros((2, 2), dtype=np.float64)
    )
    np.testing.assert_array_equal(result.confidence_bounds, [2.0, 2.0])
    np.testing.assert_allclose(result.standard_error, 0.0)


def test_declared_exact_ratio_relation_permits_constant_singleton() -> None:
    result = _analyze(
        np.asarray([[6.0]], dtype=np.float64),
        np.asarray([[3.0]], dtype=np.float64),
        deterministic_ratio=2.0,
    )
    assert bool(result.statistically_valid)
    np.testing.assert_allclose(result.value, 2.0)
    np.testing.assert_allclose(result.ratio_covariance, [[0.0]])


def test_declared_ratio_relation_is_checked_not_fitted() -> None:
    with pytest.raises(ValueError, match="declared deterministic ratio"):
        _analyze(
            np.asarray([[6.1]], dtype=np.float64),
            np.asarray([[3.0]], dtype=np.float64),
            deterministic_ratio=2.0,
        )


def test_observed_noise_free_relation_without_declaration_is_unresolved() -> None:
    a = _alternating()
    y = (20.0 + a)[None, :]
    result = _analyze(2.0 * y, y)
    assert not bool(result.statistically_valid)
    assert int(result.status) & (
        CorrelatedRatioStatus.ZERO_VARIATION_UNRESOLVED
        | CorrelatedRatioStatus.CORRELATION_UNRESOLVED
    )
    assert math.isnan(float(result.standard_error))


def test_zero_deterministic_denominator_is_always_unsafe() -> None:
    result = _analyze(
        np.asarray([[6.0]], dtype=np.float64),
        np.asarray([[0.0]], dtype=np.float64),
        deterministic=True,
    )
    assert int(result.status) & CorrelatedRatioStatus.DENOMINATOR_UNSAFE
    assert result.fieller_set == "empty"
    assert not bool(result.statistically_valid)
    assert math.isnan(float(result.exploratory_value))


@pytest.mark.parametrize(
    ("x", "y", "covariance", "expected"),
    [
        (2.0, 3.0, [[1.0, 0.0], [0.0, 1.0]], "bounded"),
        (2.0, 1.0, [[1.0, 0.0], [0.0, 1.0]], "unbounded"),
        (2.0, 0.2, [[1.0, 0.0], [0.0, 1.0]], "disconnected"),
        (0.0, 0.0, [[1.0, 0.0], [0.0, 1.0]], "all-real"),
        (1.0, 0.0, [[0.0, 0.0], [0.0, 0.0]], "empty"),
        (2.0, 4.0, [[0.0, 0.0], [0.0, 0.0]], "singleton"),
        (2.0, 1.0, [[0.25, 0.125], [0.125, 0.0625]], "singleton"),
        (1.0, 1.0, [[4.0, 4.0], [4.0, 4.0]], "all-real"),
        (0.0, 1.0, [[0.0, 0.0], [0.0, 1.0]], "all-real"),
    ],
    ids=[
        "bounded",
        "linear-half-line",
        "disconnected",
        "negative-discriminant",
        "empty",
        "deterministic-singleton",
        "zero-discriminant-singleton",
        "zero-discriminant-negative-leading",
        "zero-polynomial",
    ],
)
def test_fieller_boundary_classes(
    x: float, y: float, covariance: list[list[float]], expected: FiellerSet
) -> None:
    joint = np.asarray(covariance, dtype=np.float64)
    kind, bounds = _fieller_set(x, y, joint, 1.0)
    assert kind == expected
    if kind in ("bounded", "disconnected", "singleton"):
        # Endpoints independently satisfy the unscaled confidence inequality.
        for boundary in bounds:
            lhs = (x - boundary * y) ** 2
            rhs = joint[0, 0] - 2.0 * boundary * joint[0, 1] + boundary**2 * joint[1, 1]
            np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-13)
    if kind == "unbounded":
        np.testing.assert_allclose(bounds, [0.75, math.inf])
    if kind == "singleton" and np.all(joint == 0.0):
        np.testing.assert_allclose(bounds, [x / y, x / y])


def test_real_denominator_confidence_crossing_zero_refuses_qualified_error() -> None:
    a = _alternating()
    result = _analyze((1.0 + 0.1 * a)[None, :], (0.1 + a)[None, :])
    assert result.fieller_set == "disconnected"
    assert int(result.status) & CorrelatedRatioStatus.DENOMINATOR_UNSAFE
    np.testing.assert_allclose(result.exploratory_value, 10.0, rtol=1e-13)
    assert not bool(result.statistically_valid)
    assert math.isnan(float(result.standard_error))


@pytest.mark.parametrize(
    "center", [0.02 + 0.01j, 20.0 + 1.0j], ids=["origin-inside", "origin-excluded"]
)
def test_singular_complex_denominator_uses_origin_exclusion_ball(center: complex) -> None:
    a = _alternating()
    x = (5.0 + 0.1 * a)[None, :]
    y = np.asarray(center + a, dtype=np.complex128)[None, :]
    result = _analyze(x, y)
    joint = np.asarray(result.mean_covariance)
    assert np.linalg.matrix_rank(joint[1:, 1:]) == 1
    expected_radius = result.policy.confidence_multiplier * math.sqrt(
        float(np.trace(joint[1:, 1:]))
    )
    np.testing.assert_allclose(
        result.denominator_confidence_radius, expected_radius, rtol=1e-13
    )
    if abs(center) < expected_radius:
        assert int(result.status) & CorrelatedRatioStatus.DENOMINATOR_UNSAFE
        assert not bool(result.statistically_valid)
        assert math.isnan(float(result.standard_error))
    else:
        assert bool(result.statistically_valid)
        np.testing.assert_allclose(result.value, 5.0 / center, rtol=1e-13)


@pytest.mark.parametrize("sign", ["positive", "negative"], ids=["positive", "negative"])
def test_real_sign_floor_is_a_statistical_gate(sign: DenominatorSign) -> None:
    policy = CorrelatedRatioPolicy(
        denominator_sign=sign, minimum_denominator_magnitude=2.0
    )
    result = _analyze(
        np.asarray([[6.0]], dtype=np.float64),
        np.asarray([[-3.0]], dtype=np.float64),
        policy=policy,
        deterministic=True,
    )
    assert bool(result.statistically_valid) == (sign == "negative")
    if sign == "positive":
        assert int(result.status) & CorrelatedRatioStatus.DENOMINATOR_UNSAFE


def test_empty_history_returns_insufficient_draws_not_a_shape_exception() -> None:
    result = _analyze(
        np.empty((1, 0), dtype=np.float64), np.empty((1, 0), dtype=np.float64)
    )
    assert int(result.status) & CorrelatedRatioStatus.INSUFFICIENT_DRAWS
    assert not bool(result.statistically_valid)
    assert math.isnan(float(result.value))


def test_minimum_draws_and_complete_blocks_are_distinct_gates() -> None:
    x = (10.0 + _alternating()[:4])[None, :]
    y = np.full(x.shape, 20.0, dtype=np.float64)
    result = _analyze(
        x, y, policy=CorrelatedRatioPolicy(max_lag=1, minimum_draws=8, minimum_blocks=5)
    )
    assert int(result.status) & CorrelatedRatioStatus.INSUFFICIENT_DRAWS
    assert int(result.status) & CorrelatedRatioStatus.INSUFFICIENT_BLOCKS
    assert not bool(result.statistically_valid)


def test_invalid_active_mask_dtype_is_refused() -> None:
    with pytest.raises(TypeError):
        correlated_ratio_of_means(
            jnp.ones((1, 8), dtype=jnp.float64),
            jnp.ones((1, 8), dtype=jnp.float64),
            policy=CorrelatedRatioPolicy(),
            stream_ids=("a",),
            dependence_ids=("a",),
            sampling_origin_id="mask-contract",
            valid=jnp.ones((1, 8), dtype=jnp.int32),
        )


def test_invalid_alignment_shape_is_refused() -> None:
    with pytest.raises(ValueError):
        _analyze(np.ones((1, 8), dtype=np.float64), np.ones((1, 7), dtype=np.float64))


def test_duplicate_stream_identity_is_refused() -> None:
    with pytest.raises(ValueError):
        correlated_ratio_of_means(
            np.ones((2, 8), dtype=np.float64),
            np.ones((2, 8), dtype=np.float64),
            policy=CorrelatedRatioPolicy(),
            stream_ids=("a", "a"),
            dependence_ids=("group", "group"),
            sampling_origin_id="identity-contract",
        )


def test_nonfloating_ratio_records_are_refused() -> None:
    with pytest.raises(TypeError):
        correlated_ratio_of_means(
            jnp.ones((1, 8), dtype=jnp.int32),
            jnp.ones((1, 8), dtype=jnp.float64),
            policy=CorrelatedRatioPolicy(),
            stream_ids=("a",),
            dependence_ids=("a",),
            sampling_origin_id="dtype-contract",
        )


def test_mask_holes_never_bridge_temporal_blocks() -> None:
    x = np.asarray([[13.0, 11.0, np.nan, 20.0, 0.0, 999.0]], dtype=np.float64)
    y = np.asarray([[40.0, 40.0, np.nan, 40.0, 40.0, 40.0]], dtype=np.float64)
    valid = np.asarray([[True, True, False, True, True, True]], dtype=np.bool_)
    result = _analyze(
        x,
        y,
        valid=valid,
        policy=CorrelatedRatioPolicy(
            max_lag=1, block_length=2, minimum_draws=2, minimum_blocks=2
        ),
    )
    np.testing.assert_array_equal(result.block_index, [[0, 0, -1, 1, 1, -1]])
    np.testing.assert_array_equal(
        result.retained, [[True, True, False, True, True, False]]
    )
    assert int(result.discarded_records) == 1
    np.testing.assert_allclose(
        result.mean_covariance, [[1.0, 0.0], [0.0, 0.0]], atol=1e-15
    )
    np.testing.assert_allclose(result.numerator_mean, 11.0)


def test_zero_variation_batch_means_do_not_certify_stochastic_ratio() -> None:
    x = (10.0 + _alternating())[None, :]
    y = np.full(x.shape, 20.0, dtype=np.float64)
    result = _analyze(x, y, policy=CorrelatedRatioPolicy(max_lag=1, block_length=2))
    assert int(result.status) & CorrelatedRatioStatus.ZERO_VARIATION_UNRESOLVED
    assert not bool(result.statistically_valid)
    assert math.isnan(float(result.standard_error))
    np.testing.assert_array_equal(
        result.mean_covariance, np.zeros((2, 2), dtype=np.float64)
    )


def test_declared_varying_noise_free_relation_retains_joint_denominator_covariance() -> (
    None
):
    y = (20.0 + _alternating())[None, :]
    result = _analyze(2.0 * y, y, deterministic_ratio=2.0)
    assert bool(result.statistically_valid)
    joint = np.cov(np.stack((2.0 * y[0], y[0])), ddof=1) / 8.0
    np.testing.assert_allclose(result.mean_covariance, joint, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(result.ratio_covariance, [[0.0]])
    np.testing.assert_allclose(result.value, 2.0)
