#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.linalg import (
    banded_definiteness_certificate,
    banded_eigenvalue_brackets,
    banded_singular_value_brackets,
    banded_singular_value_certificate,
    BandedSpectrumStatus,
)


def _store(
    matrix: NDArray[np.float64] | NDArray[np.complex128], lower: int, upper: int
) -> jax.Array:
    rows, columns = matrix.shape
    bands = np.zeros((lower + upper + 1, columns), dtype=matrix.dtype)
    for column in range(columns):
        for row in range(max(0, column - upper), min(rows, column + lower + 1)):
            bands[upper + row - column, column] = matrix[row, column]
    return jnp.asarray(bands)


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("positive", [False, True])
def test_hermitian_brackets(complex_values: bool, positive: bool) -> None:
    rng = np.random.default_rng(982)
    raw = rng.normal(size=(10, 10)).astype(np.complex128)
    if complex_values:
        raw += 1j * rng.normal(size=(10, 10))
    indices = np.arange(10)
    raw[np.abs(indices[:, None] - indices[None, :]) > 2] = 0
    hermitian = (raw + raw.conj().T) / 2
    if positive:
        hermitian += 8 * np.eye(10, dtype=np.complex128)
    matrix = hermitian if complex_values else hermitian.real.copy()
    bands = _store(matrix, 2, 2)
    result = jax.jit(
        partial(banded_eigenvalue_brackets, bandwidth=2, relative_tolerance=1e-7)
    )(bands)
    reference = np.linalg.eigvalsh(matrix)
    assert float(result.minimum_lower) <= reference[0] <= float(result.minimum_upper)
    assert float(result.maximum_lower) <= reference[-1] <= float(result.maximum_upper)
    assert int(result.status) == BandedSpectrumStatus.CERTIFIED
    assert float(result.minimum_width) <= 1e-7 * abs(reference[0])
    assert float(result.maximum_width) <= 1e-7 * abs(reference[-1])


def test_ill_conditioned_brackets_and_step_limit() -> None:
    diagonal = np.geomspace(1e-10, 1, 12).astype(np.float64)
    bands = jnp.asarray(diagonal[None, :])
    result = banded_eigenvalue_brackets(bands, bandwidth=0, max_steps=100)
    assert float(result.minimum_lower) <= diagonal[0] <= float(result.minimum_upper)
    assert float(result.maximum_lower) <= diagonal[-1] <= float(result.maximum_upper)
    limited = banded_eigenvalue_brackets(bands, bandwidth=0, max_steps=1)
    assert int(limited.status) == BandedSpectrumStatus.BRACKET_NOT_CONVERGED
    assert int(limited.steps) == 1
    assert float(limited.minimum_lower) <= diagonal[0] <= float(limited.minimum_upper)
    rng = np.random.default_rng(812)
    factor = np.eye(12, dtype=np.float64)
    factor[np.arange(1, 12), np.arange(11)] = 0.1 * rng.normal(size=11)
    factor[-1, -1] = 1e-5
    matrix = factor @ factor.T
    reference = np.linalg.eigvalsh(matrix)
    ill = banded_eigenvalue_brackets(_store(matrix, 1, 1), bandwidth=1)
    assert 1e9 < reference[-1] / reference[0] < 1e11
    assert float(ill.minimum_lower) <= reference[0] <= float(ill.minimum_upper)
    assert float(ill.maximum_lower) <= reference[-1] <= float(ill.maximum_upper)


def test_shifted_definiteness_and_uncertainty() -> None:
    bands = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float64)
    assert bool(banded_definiteness_certificate(bands, bandwidth=0, shift=0.9).certified)
    assert bool(
        banded_definiteness_certificate(
            bands, bandwidth=0, shift=3.1, upper=True
        ).certified
    )
    negative = banded_definiteness_certificate(bands, bandwidth=0, shift=1.1)
    assert not bool(negative.certified)
    assert int(negative.status) == BandedSpectrumStatus.INDEFINITE_AT_SHIFT
    touching = banded_definiteness_certificate(bands, bandwidth=0, shift=1.0)
    assert not bool(touching.certified)
    assert int(touching.status) == BandedSpectrumStatus.BRACKET_NOT_CONVERGED


@pytest.mark.parametrize("rectangular", [False, True])
def test_block_banded_singular_brackets_and_threshold(rectangular: bool) -> None:
    rng = np.random.default_rng(734)
    rows, columns = (12, 8) if rectangular else (8, 8)
    matrix = np.zeros((rows, columns), dtype=np.complex128)
    for block in range(4):
        matrix[2 * block : 2 * block + 2, 2 * block : 2 * block + 2] = 3 * np.eye(
            2, dtype=np.complex128
        ) + 0.2 * rng.normal(size=(2, 2))
        if block < 3:
            matrix[2 * block : 2 * block + 2, 2 * block + 2 : 2 * block + 4] = rng.normal(
                size=(2, 2)
            ) + 0.2j * rng.normal(size=(2, 2))
    if rectangular:
        matrix[8:10, 6:8] = rng.normal(size=(2, 2))
    lower, upper = (3, 3) if rectangular else (1, 3)
    bands = _store(matrix, lower, upper)
    bracket = jax.jit(
        partial(
            banded_singular_value_brackets,
            rows=rows,
            lower_bandwidth=lower,
            upper_bandwidth=upper,
            relative_tolerance=1e-7,
        )
    )(bands)
    reference = np.linalg.svd(matrix, compute_uv=False)
    assert float(bracket.minimum_lower) <= reference[-1] <= float(bracket.minimum_upper)
    assert float(bracket.maximum_lower) <= reference[0] <= float(bracket.maximum_upper)
    assert float(bracket.condition_upper_bound) >= reference[0] / reference[-1]
    assert int(bracket.status) == BandedSpectrumStatus.CERTIFIED
    certificate = jax.jit(
        partial(
            banded_singular_value_certificate,
            rows=rows,
            lower_bandwidth=lower,
            upper_bandwidth=upper,
        )
    )
    below = certificate(bands, threshold=reference[-1] * (1 - 1e-6))
    above = certificate(bands, threshold=reference[-1] * (1 + 1e-6))
    assert bool(below.certified)
    assert not bool(above.certified)
    assert int(above.status) == BandedSpectrumStatus.INDEFINITE_AT_SHIFT


def test_default_convergence_on_cubic_collocation() -> None:
    from scipy.interpolate import BSpline

    knots = np.concatenate(
        (
            np.zeros(4, dtype=np.float64),
            np.linspace(0, 1, 30, dtype=np.float64)[1:-1],
            np.ones(4, dtype=np.float64),
        )
    )
    points = np.asarray(
        [np.mean(knots[i + 1 : i + 4]) for i in range(32)], dtype=np.float64
    )
    matrix = BSpline.design_matrix(points, knots, 3).toarray()
    result = banded_singular_value_brackets(
        _store(matrix, 3, 3), rows=32, lower_bandwidth=3, upper_bandwidth=3
    )
    reference = np.linalg.svd(matrix, compute_uv=False)
    assert int(result.status) == BandedSpectrumStatus.CERTIFIED
    assert float(result.minimum_lower) <= reference[-1] <= float(result.minimum_upper)
    assert float(result.maximum_lower) <= reference[0] <= float(result.maximum_upper)
    assert float(result.minimum_rounding_floor) > 0
    assert float(result.maximum_rounding_floor) > 0


def test_gram_off_diagonal_fault_guard() -> None:
    matrix = np.asarray(
        [[1.0, 0.8, 0.0], [0.0, 1.0, 0.8], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    bands = _store(matrix, 0, 1)
    result = banded_singular_value_brackets(
        bands, rows=3, lower_bandwidth=0, upper_bandwidth=1
    )
    reference = np.linalg.svd(matrix, compute_uv=False)
    assert float(result.minimum_lower) <= reference[-1] <= float(result.minimum_upper)
    assert float(result.maximum_lower) <= reference[0] <= float(result.maximum_upper)
    assert not bool(
        banded_singular_value_certificate(
            bands, rows=3, lower_bandwidth=0, upper_bandwidth=1, threshold=0.9
        ).certified
    )


def test_wide_and_invalid_hermitian_fail_closed() -> None:
    matrix = np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    result = banded_singular_value_brackets(
        _store(matrix, 0, 1), rows=2, lower_bandwidth=0, upper_bandwidth=1, max_steps=4
    )
    assert float(result.minimum_lower) == float(result.minimum_upper) == 0
    assert np.isinf(float(result.condition_upper_bound))
    broken = jnp.asarray([[0.0, 2.0], [1.0, 1.0], [1.0, 0.0]], dtype=jnp.float64)
    certificate = banded_definiteness_certificate(broken, bandwidth=1)
    assert not bool(certificate.certified)
    bracket = banded_eigenvalue_brackets(broken, bandwidth=1, max_steps=1)
    assert np.isneginf(float(bracket.minimum_lower))
    assert np.isposinf(float(bracket.maximum_upper))
