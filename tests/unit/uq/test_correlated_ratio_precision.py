#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict-promotion mixed-coordinate covariance and refusal regressions."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.uq import (
    correlated_ratio_of_means,
    CorrelatedRatioPolicy,
    CorrelatedRatioStatus,
)


pytestmark = pytest.mark.strict_jax


@pytest.mark.parametrize(
    "numerator_complex", [False, True], ids=["real-over-complex", "complex-over-real"]
)
def test_mixed_ratio_preserves_full_cross_covariance(
    numerator_complex: bool,
) -> None:
    amplitudes = 0.1 * np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 5.0, 6.0],
            [3.0, 5.0, 9.0, 8.0],
            [4.0, 6.0, 8.0, 16.0],
        ],
        dtype=np.float64,
    )
    deviations = np.stack((amplitudes, -amplitudes), axis=1).reshape((8, 4))
    channels = np.asarray([10.0, 3.0, 20.0, 10.0], dtype=np.float64) + deviations
    if numerator_complex:
        numerator = (channels[:, 0] + 1j * channels[:, 1])[None, :]
        denominator = channels[:, 2][None, :]
        selected = channels[:, [0, 1, 2]]
        x_mean, y_mean = 10.0 + 3.0j, 20.0 + 0.0j
        derivatives = np.asarray(
            [1.0 / y_mean, 1j / y_mean, -x_mean / y_mean**2],
            dtype=np.complex128,
        )
    else:
        numerator = channels[:, 0][None, :]
        denominator = (channels[:, 2] + 1j * channels[:, 3])[None, :]
        selected = channels[:, [0, 2, 3]]
        x_mean, y_mean = 10.0 + 0.0j, 20.0 + 10.0j
        derivative_y = -x_mean / y_mean**2
        derivatives = np.asarray(
            [1.0 / y_mean, derivative_y, 1j * derivative_y],
            dtype=np.complex128,
        )
    joint = np.cov(selected, rowvar=False, ddof=1) / 8.0
    jacobian = np.stack((derivatives.real, derivatives.imag))
    expected = jacobian @ joint @ jacobian.T
    diagonal_only = jacobian @ np.diag(np.diag(joint)) @ jacobian.T
    result = correlated_ratio_of_means(
        numerator,
        denominator,
        policy=CorrelatedRatioPolicy(max_lag=1),
        stream_ids=("mixed-stream",),
        dependence_ids=("mixed-origin",),
        sampling_origin_id="mixed-cross-covariance",
    )
    assert bool(result.statistically_valid)
    np.testing.assert_allclose(result.mean_covariance, joint, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.ratio_covariance, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.value, x_mean / y_mean, rtol=1e-14)
    np.testing.assert_allclose(
        result.standard_error, np.sqrt(np.trace(expected)), rtol=1e-12, atol=1e-15
    )
    assert not np.allclose(expected, diagonal_only, rtol=0.01, atol=1e-15)
    assert result.numerator_mean.dtype == numerator.dtype
    assert result.denominator_mean.dtype == denominator.dtype
    assert result.value.dtype == np.dtype(np.complex128)
    assert result.mean_covariance.dtype == np.dtype(np.float64)
    assert result.ratio_covariance.dtype == np.dtype(np.float64)


@pytest.mark.parametrize("numerator_complex", [False, True], ids=["real-x", "complex-x"])
@pytest.mark.parametrize(
    "denominator_complex", [False, True], ids=["real-y", "complex-y"]
)
@pytest.mark.parametrize("zero_denominator", [False, True], ids=["below-floor", "zero"])
def test_denominator_refusal_keeps_native_point_kind(
    numerator_complex: bool,
    denominator_complex: bool,
    zero_denominator: bool,
) -> None:
    numerator = np.asarray(
        [[6.0 + 2.0j]] if numerator_complex else [[6.0]],
        dtype=np.complex128 if numerator_complex else np.float64,
    )
    denominator = np.asarray(
        [[0.0]]
        if zero_denominator
        else [[3.0 + 1.0j]]
        if denominator_complex
        else [[3.0]],
        dtype=np.complex128 if denominator_complex else np.float64,
    )
    result = correlated_ratio_of_means(
        numerator,
        denominator,
        policy=CorrelatedRatioPolicy(minimum_denominator_magnitude=100.0),
        stream_ids=("refusal-stream",),
        dependence_ids=("refusal-origin",),
        sampling_origin_id="native-denominator-refusal",
        deterministic=True,
    )
    expected_status = CorrelatedRatioStatus.DENOMINATOR_UNSAFE
    if zero_denominator:
        expected_status |= CorrelatedRatioStatus.NONFINITE
        assert np.isnan(np.asarray(result.exploratory_value))
    else:
        np.testing.assert_allclose(
            result.exploratory_value, numerator[0, 0] / denominator[0, 0], rtol=1e-14
        )
    assert int(result.status) == int(expected_status)
    assert not bool(result.statistically_valid)
    assert np.isnan(np.asarray(result.value))
    assert np.all(np.isnan(np.asarray(result.ratio_covariance)))
    assert np.isnan(np.asarray(result.standard_error))
    point_dtype = np.dtype(
        np.complex128 if numerator_complex or denominator_complex else np.float64
    )
    assert result.numerator_mean.dtype == numerator.dtype
    assert result.denominator_mean.dtype == denominator.dtype
    assert result.exploratory_value.dtype == point_dtype
    assert result.value.dtype == point_dtype
    assert result.ratio_covariance.dtype == np.dtype(np.float64)
