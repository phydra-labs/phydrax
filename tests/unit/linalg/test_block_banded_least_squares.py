"""Independent dense references for exact block-banded local QR elimination."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.linalg._block_banded_least_squares import solve_block_banded_least_squares


def _fixture() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    rng = np.random.default_rng(72)
    return (
        rng.normal(size=(4, 5, 2)),
        rng.normal(size=(4, 5)),
        rng.normal(size=(3, 2, 2)),
        np.broadcast_to(np.eye(2), (3, 2, 2)).copy(),
        rng.normal(size=(3, 2)),
    )


def _dense(
    samples: NDArray[np.float64], lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> NDArray[np.float64]:
    count, rows, width = samples.shape
    matrix = np.zeros(
        (count * rows + (count - 1) * width, count * width), dtype=np.float64
    )
    for segment in range(count):
        matrix[
            segment * rows : (segment + 1) * rows, segment * width : (segment + 1) * width
        ] = samples[segment]
    for segment in range(count - 1):
        row = count * rows + segment * width
        matrix[row : row + width, segment * width : (segment + 1) * width] = lower[
            segment
        ]
        matrix[row : row + width, (segment + 1) * width : (segment + 2) * width] = upper[
            segment
        ]
    return matrix


@pytest.mark.parametrize(
    "regularization", [0.0, 0.3], ids=["unregularized", "regularized"]
)
def test_block_banded_solution_and_singular_evidence(regularization: float) -> None:
    a, b, left, right, coupling = _fixture()
    matrix = _dense(a, left, right)
    target = np.concatenate((b.ravel(), coupling.ravel()))
    solve_matrix = np.concatenate(
        (matrix, np.sqrt(regularization) * np.eye(matrix.shape[1]))
    )
    solve_target = np.concatenate(
        (target, np.zeros((matrix.shape[1],), dtype=np.float64))
    )
    reference = np.linalg.lstsq(solve_matrix, solve_target, rcond=None)[0]
    singular = np.linalg.svd(matrix, compute_uv=False)
    result = solve_block_banded_least_squares(
        a, b, left, right, coupling, regularization=regularization
    )
    np.testing.assert_allclose(result.value.ravel(), reference, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(result.minimum_singular_value, singular[-1], rtol=1e-7)
    np.testing.assert_allclose(result.maximum_singular_value, singular[0], rtol=1e-7)
    np.testing.assert_allclose(
        result.condition_number, singular[0] / singular[-1], rtol=1e-6
    )
    assert result.successful
    assert result.rank_evidence.lower_bound == matrix.shape[1]
    np.testing.assert_allclose(
        result.residual_norm, np.linalg.norm(matrix @ reference - target), rtol=1e-11
    )


def test_block_banded_single_segment() -> None:
    result = solve_block_banded_least_squares(
        jnp.asarray([[[2.0], [1.0]]], dtype=jnp.float64),
        jnp.asarray([[1.0, 3.0]], dtype=jnp.float64),
        jnp.zeros((0, 1, 1), dtype=jnp.float64),
        jnp.zeros((0, 1, 1), dtype=jnp.float64),
        jnp.zeros((0, 1), dtype=jnp.float64),
    )
    np.testing.assert_allclose(result.value, [[1.0]], atol=1e-13)
    assert result.successful


@pytest.mark.parametrize(
    "small", [0.0, 1e-16], ids=["rank-deficient", "below-dense-cutoff"]
)
def test_block_banded_refuses_uncertified_rank(small: float) -> None:
    a = np.asarray([[[1.0, 0.0], [0.0, small]]], dtype=np.float64)
    with pytest.raises(eqx.EquinoxRuntimeError, match="dense-svd"):
        solve_block_banded_least_squares(
            a,
            np.ones((1, 2), dtype=np.float64),
            np.zeros((0, 2, 2), dtype=np.float64),
            np.zeros((0, 2, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
        )


def test_block_banded_above_dense_cutoff_keeps_small_singular_mode() -> None:
    samples = np.asarray([[[1.0, 0.0], [0.0, 1e-15]]], dtype=np.float64)
    singular = np.linalg.svd(samples[0], compute_uv=False)
    assert singular[-1] > np.finfo(np.float64).eps * 2 * singular[0]
    result = solve_block_banded_least_squares(
        samples,
        np.ones((1, 2), dtype=np.float64),
        np.zeros((0, 2, 2), dtype=np.float64),
        np.zeros((0, 2, 2), dtype=np.float64),
        np.zeros((0, 2), dtype=np.float64),
    )
    expected = np.linalg.lstsq(samples[0], np.ones((2,), dtype=np.float64), rcond=None)[0]
    np.testing.assert_allclose(result.value[0], expected, rtol=1e-12)
    assert result.rank_evidence.lower_bound == 2
    assert result.successful


def test_coupling_oracle_detects_dropped_continuity_block() -> None:
    a, b, left, right, coupling = _fixture()
    matrix = _dense(a, left, right)
    target = np.concatenate((b.ravel(), coupling.ravel()))
    expected = np.linalg.lstsq(matrix, target, rcond=None)[0]
    faulty = left.copy()
    faulty[1] = 0.0
    result = solve_block_banded_least_squares(a, b, faulty, right, coupling)
    assert np.linalg.norm(np.asarray(result.value).ravel() - expected) > 0.05
