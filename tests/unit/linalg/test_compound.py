from __future__ import annotations

from itertools import combinations
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from phydrax.linalg import compound_matrix, SmallLinearSolvePlan, solve_small_linear


def _minors(matrix: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    rows = tuple(combinations(range(matrix.shape[-2]), degree))
    columns = tuple(combinations(range(matrix.shape[-1]), degree))
    result = np.empty(matrix.shape[:-2] + (len(rows), len(columns)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            result[..., i, j] = np.linalg.det(
                matrix[
                    ...,
                    np.asarray(row, dtype=np.int32)[:, None],
                    np.asarray(column, dtype=np.int32)[None, :],
                ]
            )
    return result


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5])
def test_rectangular_cauchy_binet_and_lexicographic_order(degree: int) -> None:
    random = np.random.default_rng(17)
    left = random.normal(size=(2, 5, 6))
    right = random.normal(size=(2, 6, 4))
    actual = compound_matrix(left, degree) @ compound_matrix(right, degree)
    expected = _minors(left @ right, degree)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        compound_matrix(left, degree), _minors(left, degree), atol=1e-10, rtol=1e-10
    )
    assert actual.shape == (2, comb(5, degree), comb(4, degree) if degree <= 4 else 0)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5])
def test_inverse_transpose_and_top_degree_laws(degree: int) -> None:
    matrix = np.random.default_rng(33).normal(size=(5, 5)) + 4 * np.eye(5)
    np.testing.assert_allclose(
        compound_matrix(matrix.T, degree), compound_matrix(matrix, degree).T, atol=1e-10
    )
    np.testing.assert_allclose(
        compound_matrix(matrix, degree) @ compound_matrix(np.linalg.inv(matrix), degree),
        np.eye(comb(5, degree)),
        atol=1e-10,
    )
    if degree == 5:
        np.testing.assert_allclose(
            compound_matrix(matrix, 5)[0, 0], np.linalg.det(matrix), atol=1e-10
        )


@pytest.mark.parametrize("degree", [2, 3, 4, 5, 6])
def test_rank_one_defect_has_adjugate_not_zero_derivative(degree: int) -> None:
    diagonal = jnp.arange(1, degree + 1, dtype=jnp.float64).at[-1].set(0)
    matrix = jnp.diag(diagonal)
    tangent = jnp.zeros_like(matrix).at[-1, -1].set(1)

    def determinant(value: Array) -> Array:
        return compound_matrix(value, degree)[0, 0]

    value, derivative = jax.jvp(jax.jit(determinant), (matrix,), (tangent,))
    np.testing.assert_allclose(value, 0, atol=1e-12)
    np.testing.assert_allclose(derivative, np.prod(np.arange(1, degree)), atol=1e-10)
    expected_gradient = np.zeros((degree, degree), dtype=np.float64)
    expected_gradient[-1, -1] = np.prod(np.arange(1, degree))
    np.testing.assert_allclose(
        jax.grad(determinant)(matrix), expected_gradient, atol=1e-10
    )


def test_complex_batched_compound_and_singular_holomorphic_jvp() -> None:
    matrix = jnp.asarray([[1 + 2j, 3 - 1j], [2j, 4 + 3j]], dtype=jnp.complex128)
    batched = jnp.stack((matrix, 2 * matrix))
    np.testing.assert_allclose(
        jax.jit(lambda value: compound_matrix(value, 2))(batched)[..., 0, 0],
        np.linalg.det(np.asarray(batched)),
    )
    np.testing.assert_allclose(
        jax.vmap(lambda value: compound_matrix(value, 2))(batched),
        compound_matrix(batched, 2),
    )
    singular = jnp.diag(
        jnp.asarray([1 + 2j, 2 - 1j, 3j, 4 + 1j, 0j], dtype=jnp.complex128)
    )
    tangent = jnp.zeros_like(singular).at[-1, -1].set(2 + 3j)

    def determinant(value: Array) -> Array:
        return compound_matrix(value, 5)[0, 0]

    _, derivative = jax.jvp(determinant, (singular,), (tangent,))
    np.testing.assert_allclose(
        derivative, np.prod(np.diag(np.asarray(singular))[:-1]) * (2 + 3j), atol=1e-10
    )


def test_degree_boundaries_and_resource_refusal() -> None:
    matrix = jnp.zeros((3, 2, 4), dtype=jnp.float64)
    np.testing.assert_array_equal(
        compound_matrix(matrix, 0), np.ones((3, 1, 1), dtype=np.float64)
    )
    assert compound_matrix(matrix, 3).shape == (3, 0, 4)
    assert compound_matrix(matrix, 5).shape == (3, 0, 0)
    with pytest.raises(ValueError):
        compound_matrix(matrix, -1)
    with pytest.raises(ValueError):
        compound_matrix(matrix, 2, maximum_minor_entries=71)
    assert compound_matrix(matrix, 2, maximum_minor_entries=72).shape == (3, 1, 6)


def test_four_by_four_complex_batches_obey_strict_rank_and_dtype_contracts() -> None:
    random = np.random.default_rng(19)
    matrix = random.normal(size=(2, 3, 4, 4)) + 1j * random.normal(size=(2, 3, 4, 4))
    matrix += 7 * np.eye(4, dtype=np.complex128)
    rhs = random.normal(size=(2, 3, 4)) + 1j * random.normal(size=(2, 3, 4))
    plan = SmallLinearSolvePlan(4)
    with jax.numpy_rank_promotion("raise"), jax.numpy_dtype_promotion("strict"):
        result = jax.jit(lambda matrix, rhs: solve_small_linear(plan, matrix, rhs))(
            jnp.asarray(matrix, dtype=jnp.complex128),
            jnp.asarray(rhs, dtype=jnp.complex128),
        )
    np.testing.assert_allclose(
        result.value, np.linalg.solve(matrix, rhs[..., None])[..., 0], atol=1e-11
    )
    np.testing.assert_allclose(result.determinant, np.linalg.det(matrix), atol=1e-9)
    assert bool(jnp.all(result.successful))


def test_multiaxis_batched_compound_jvp_broadcasts_cofactor_signs_explicitly() -> None:
    random = np.random.default_rng(32)
    matrix = random.normal(size=(2, 3, 4, 4))
    tangent = random.normal(size=(2, 3, 4, 4))
    expected = np.empty((2, 3, 6, 6), dtype=np.float64)
    indices = tuple(combinations(range(4), 2))
    for i, (a, b) in enumerate(indices):
        for j, (c, d) in enumerate(indices):
            expected[..., i, j] = (
                tangent[..., a, c] * matrix[..., b, d]
                + matrix[..., a, c] * tangent[..., b, d]
                - tangent[..., a, d] * matrix[..., b, c]
                - matrix[..., a, d] * tangent[..., b, c]
            )
    with jax.numpy_rank_promotion("raise"):
        _, derivative = jax.jvp(
            jax.jit(lambda value: compound_matrix(value, 2)),
            (jnp.asarray(matrix, dtype=jnp.float64),),
            (jnp.asarray(tangent, dtype=jnp.float64),),
        )
    np.testing.assert_allclose(derivative, expected, atol=1e-11)
