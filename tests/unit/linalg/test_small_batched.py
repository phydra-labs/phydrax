from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.linalg import (
    determinant_small_linear,
    SmallLinearSolvePlan,
    solve_small_linear,
)


def _system(
    dimension: int,
    batch: tuple[int, ...],
    complex_valued: bool,
    seed: int,
    /,
) -> np.ndarray:
    random = np.random.default_rng(seed)
    shape = batch + (dimension, dimension)
    matrix = random.normal(size=shape)
    if complex_valued:
        matrix = matrix + 1j * random.normal(size=shape)
    return matrix + (dimension + 2.0) * np.eye(dimension)


def _right(
    dimension: int,
    batch: tuple[int, ...],
    columns: int | None,
    complex_valued: bool,
    seed: int,
    /,
) -> np.ndarray:
    random = np.random.default_rng(seed)
    shape = batch + (dimension,) + (() if columns is None else (columns,))
    right = random.normal(size=shape)
    if complex_valued:
        right = right + 1j * random.normal(size=shape)
    return right


@pytest.mark.parametrize("dimension", (1, 2, 3, 4))
@pytest.mark.parametrize("complex_valued", (False, True))
@pytest.mark.parametrize("columns", (None, 1, 3, 6))
def test_local_algebra_solve_matches_dense_reference_and_evidence(
    dimension: int, complex_valued: bool, columns: int | None
) -> None:
    batch = (2, 3)
    matrix = _system(dimension, batch, complex_valued, 10 * dimension)
    right = _right(dimension, batch, columns, complex_valued, 10 * dimension + 1)
    dtype = jnp.complex128 if complex_valued else jnp.float64
    plan = SmallLinearSolvePlan(dimension)
    with (
        pxconfig.context(pxconfig.NumericsConfig(rank_promotion="raise")),
        pxconfig.context(pxconfig.NumericsConfig(dtype_promotion="strict")),
    ):
        result = jax.jit(lambda a, b: solve_small_linear(plan, a, b))(
            jnp.asarray(matrix, dtype=dtype), jnp.asarray(right, dtype=dtype)
        )
        determinant = jax.jit(lambda a: determinant_small_linear(plan, a))(
            jnp.asarray(matrix, dtype=dtype)
        )

    matrix_right = right[..., None] if columns is None else right
    expected = np.linalg.solve(matrix, matrix_right)
    expected = expected[..., 0] if columns is None else expected
    value = np.asarray(result.value)
    assert value.dtype == np.dtype(dtype)
    assert value.shape == right.shape
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        result.determinant, np.linalg.det(matrix), rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(result.determinant, determinant, rtol=1e-14, atol=0.0)
    value_columns = value[..., None] if columns is None else value
    residual_bound = (
        32.0
        * np.finfo(np.float64).eps
        * (
            np.linalg.norm(matrix, axis=(-2, -1))
            * np.linalg.norm(value_columns, axis=(-2, -1))
            + np.linalg.norm(matrix_right, axis=(-2, -1))
        )
    )
    assert np.all(np.asarray(result.residual_norm) <= residual_bound)
    inverse = np.linalg.inv(matrix)
    condition = np.linalg.norm(matrix, np.inf, axis=(-2, -1)) * np.linalg.norm(
        inverse, np.inf, axis=(-2, -1)
    )
    np.testing.assert_allclose(result.condition_estimate, condition, rtol=1e-12)
    assert bool(jnp.all(result.successful))
    np.testing.assert_array_equal(result.status, np.zeros(batch, dtype=np.int32))
    np.testing.assert_array_equal(result.rank, np.full(batch, dimension, np.int32))
    assert bool(jnp.all((result.refinement_iterations >= 0)))
    assert bool(jnp.all((result.refinement_iterations <= plan.refinement_iterations)))


@pytest.mark.parametrize("dimension", (1, 2, 3, 4))
@pytest.mark.parametrize("complex_valued", (False, True))
@pytest.mark.parametrize("refinement_iterations", (0, 1))
def test_solution_and_determinant_derivatives_match_implicit_differentiation(
    dimension: int, complex_valued: bool, refinement_iterations: int
) -> None:
    batch = (4,)
    matrix = _system(dimension, batch, complex_valued, 40 + dimension)
    right = _right(dimension, batch, dimension, complex_valued, 50 + dimension)
    matrix_tangent = _system(dimension, batch, complex_valued, 60 + dimension)
    right_tangent = _right(dimension, batch, dimension, complex_valued, 70 + dimension)
    dtype = jnp.complex128 if complex_valued else jnp.float64
    plan = SmallLinearSolvePlan(dimension, refinement_iterations=refinement_iterations)

    value, derivative = jax.jit(
        lambda a, b, da, db: jax.jvp(
            lambda x, y: solve_small_linear(plan, x, y).value, (a, b), (da, db)
        )
    )(
        jnp.asarray(matrix, dtype=dtype),
        jnp.asarray(right, dtype=dtype),
        jnp.asarray(matrix_tangent, dtype=dtype),
        jnp.asarray(right_tangent, dtype=dtype),
    )
    determinant, determinant_derivative = jax.jvp(
        lambda x: determinant_small_linear(plan, x),
        (jnp.asarray(matrix, dtype=dtype),),
        (jnp.asarray(matrix_tangent, dtype=dtype),),
    )

    solution = np.linalg.solve(matrix, right)
    expected = np.linalg.solve(matrix, right_tangent - matrix_tangent @ solution)
    np.testing.assert_allclose(value, solution, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(derivative, expected, rtol=1e-11, atol=1e-12)
    expected_determinant = np.linalg.det(matrix)
    expected_determinant_derivative = expected_determinant * np.trace(
        np.linalg.solve(matrix, matrix_tangent), axis1=-2, axis2=-1
    )
    np.testing.assert_allclose(determinant, expected_determinant, rtol=1e-12)
    np.testing.assert_allclose(
        determinant_derivative, expected_determinant_derivative, rtol=1e-11
    )


@pytest.mark.parametrize("dimension", (1, 2, 3, 4))
def test_reverse_mode_is_the_transpose_of_forward_mode(dimension: int) -> None:
    batch = (3,)
    matrix = jnp.asarray(_system(dimension, batch, False, 80 + dimension))
    right = jnp.asarray(_right(dimension, batch, 2, False, 81 + dimension))
    matrix_tangent = jnp.asarray(_system(dimension, batch, False, 82 + dimension))
    right_tangent = jnp.asarray(_right(dimension, batch, 2, False, 83 + dimension))
    cotangent = jnp.asarray(_right(dimension, batch, 2, False, 84 + dimension))
    plan = SmallLinearSolvePlan(dimension)

    def value(a: jax.Array, b: jax.Array) -> jax.Array:
        return solve_small_linear(plan, a, b).value

    _, forward = jax.jvp(value, (matrix, right), (matrix_tangent, right_tangent))
    _, pullback = jax.vjp(value, matrix, right)
    matrix_cotangent, right_cotangent = jax.jit(pullback)(cotangent)

    np.testing.assert_allclose(
        jnp.sum(cotangent * forward),
        jnp.sum(matrix_cotangent * matrix_tangent)
        + jnp.sum(right_cotangent * right_tangent),
        rtol=1e-12,
    )
    solution = np.linalg.solve(np.asarray(matrix), np.asarray(right))
    adjoint = np.linalg.solve(np.swapaxes(np.asarray(matrix), -1, -2), cotangent)
    np.testing.assert_allclose(right_cotangent, adjoint, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        matrix_cotangent,
        -adjoint @ np.swapaxes(solution, -1, -2),
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("dimension", (2, 3, 4))
def test_tied_scale_entries_do_not_perturb_exact_derivatives(dimension: int) -> None:
    # Diagonal and corner entries tie at the max-magnitude scale; the tangent
    # grows two tied entries, where a max-reduction tangent is one-sided.
    matrix = 2.0 * np.eye(dimension) + 0.25 * np.tri(dimension, k=-1)
    matrix[0, -1] = -2.0
    tangent = np.zeros((dimension, dimension))
    tangent[0, 0] = 1.0
    tangent[0, -1] = -1.0
    right = np.arange(1.0, dimension + 1.0)
    plan = SmallLinearSolvePlan(dimension)

    _, derivative = jax.jvp(
        lambda a: solve_small_linear(plan, a, jnp.asarray(right)).value,
        (jnp.asarray(matrix),),
        (jnp.asarray(tangent),),
    )
    _, determinant_derivative = jax.jvp(
        lambda a: determinant_small_linear(plan, a),
        (jnp.asarray(matrix),),
        (jnp.asarray(tangent),),
    )

    solution = np.linalg.solve(matrix, right)
    np.testing.assert_allclose(
        derivative,
        -np.linalg.solve(matrix, tangent @ solution),
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        determinant_derivative,
        np.linalg.det(matrix) * np.trace(np.linalg.solve(matrix, tangent)),
        rtol=1e-12,
        atol=1e-13,
    )


def test_mixed_batch_reports_independent_status_and_conditioning() -> None:
    accepted_gap = 1e-6
    rejected_gap = 2e-12
    exact = np.asarray((1.0, -2.0, 0.5))
    lanes = np.stack(
        (
            np.asarray(((4.0, 1.0, 0.0), (1.0, 3.0, -1.0), (0.0, -1.0, 2.0))),
            np.asarray(((1.0, 1.0, 0.0), (1.0, 1.0 + accepted_gap, 0.0), (0, 0, 1.0))),
            np.asarray(((1.0, 1.0, 0.0), (1.0, 1.0 + rejected_gap, 0.0), (0, 0, 1.0))),
            np.asarray(((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 1.0, 1.0))),
            np.zeros((3, 3)),
            np.asarray(((np.nan, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
            np.eye(3),
        )
    )
    right = lanes @ exact
    right[4] = exact
    right[5] = exact
    right[6] = np.asarray((np.inf, 0.0, 0.0))
    plan = SmallLinearSolvePlan(3)

    solve = jax.jit(lambda a, b: solve_small_linear(plan, a, b))
    result = solve(jnp.asarray(lanes), jnp.asarray(right))
    first = solve(jnp.asarray(lanes[:1]), jnp.asarray(right[:1]))

    np.testing.assert_array_equal(result.status, (0, 0, 2, 1, 1, 1, 2))
    np.testing.assert_array_equal(result.successful, (1, 1, 0, 0, 0, 0, 0))
    np.testing.assert_array_equal(result.rank, (3, 3, 0, 0, 0, 0, 0))
    np.testing.assert_array_equal(result.value[2:], np.zeros((5, 3)))
    np.testing.assert_allclose(result.value[0], exact, rtol=1e-13)
    np.testing.assert_allclose(result.value[:1], first.value, rtol=1e-14, atol=0.0)
    accepted_condition = (2.0 + accepted_gap) ** 2 / accepted_gap
    rejected_condition = (2.0 + rejected_gap) ** 2 / rejected_gap
    np.testing.assert_allclose(
        result.condition_estimate[1], accepted_condition, rtol=1e-8
    )
    np.testing.assert_allclose(
        result.condition_estimate[2], rejected_condition, rtol=1e-3
    )
    assert float(result.condition_estimate[2]) > plan.maximum_condition
    np.testing.assert_allclose(
        result.value[1], exact, rtol=64.0 * accepted_condition * np.finfo(float).eps
    )
    assert float(result.residual_norm[1]) <= 32.0 * np.finfo(float).eps * float(
        np.linalg.norm(lanes[1]) * np.linalg.norm(exact) + np.linalg.norm(right[1])
    )
    assert not bool(jnp.isfinite(result.determinant[5]))


def test_refinement_reports_only_residual_reducing_corrections() -> None:
    random = np.random.default_rng(91)
    base = random.normal(size=(64, 3, 3))
    left, _, right_factor = np.linalg.svd(base)
    singular_values = np.asarray((1.0, 1e-2, 1e-7))
    matrix = (left * singular_values[None, None, :]) @ right_factor
    right = random.normal(size=(64, 3, 3))
    unrefined_plan = SmallLinearSolvePlan(3, refinement_iterations=0)
    refined_plan = SmallLinearSolvePlan(3, refinement_iterations=1)

    unrefined = jax.jit(lambda a, b: solve_small_linear(unrefined_plan, a, b))(
        jnp.asarray(matrix), jnp.asarray(right)
    )
    refined = jax.jit(lambda a, b: solve_small_linear(refined_plan, a, b))(
        jnp.asarray(matrix), jnp.asarray(right)
    )

    assert bool(jnp.all(unrefined.successful)) and bool(jnp.all(refined.successful))
    np.testing.assert_array_equal(unrefined.refinement_iterations, np.zeros(64))
    applied = np.asarray(refined.refinement_iterations) == 1
    rejected = ~applied
    slack = 1e-12 * np.asarray(unrefined.residual_norm)
    assert np.all(
        np.asarray(refined.residual_norm) <= np.asarray(unrefined.residual_norm) + slack
    )
    np.testing.assert_allclose(
        np.asarray(refined.value)[rejected],
        np.asarray(unrefined.value)[rejected],
        rtol=1e-12,
        atol=0.0,
    )
    expected = np.linalg.solve(matrix, right)
    condition = singular_values[0] / singular_values[-1]
    np.testing.assert_allclose(
        refined.value,
        expected,
        rtol=0.0,
        atol=64.0 * condition * np.finfo(float).eps * np.abs(expected).max(),
    )


def test_four_by_four_pivoting_parity_and_nonfinite_pivots() -> None:
    permutation = np.eye(4)[[2, 0, 3, 1]]
    leading_zero = np.asarray(
        (
            (0.0, 2.0, 1.0, 0.5),
            (0.0, 1.0, 3.0, 1.0),
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 0.5, 1.0, 4.0),
        )
    )
    nan_pivot = np.eye(4)
    nan_pivot[3, 0] = np.nan
    matrices = np.stack((permutation, leading_zero, nan_pivot))
    right = np.stack((np.arange(1.0, 5.0),) * 3)
    plan = SmallLinearSolvePlan(4)

    result = jax.jit(lambda a, b: solve_small_linear(plan, a, b))(
        jnp.asarray(matrices), jnp.asarray(right)
    )
    determinant = determinant_small_linear(plan, jnp.asarray(matrices))

    np.testing.assert_allclose(
        result.value[:2],
        np.linalg.solve(matrices[:2], right[:2, :, None])[..., 0],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        result.determinant[:2], np.linalg.det(matrices[:2]), rtol=1e-13
    )
    np.testing.assert_allclose(result.determinant, determinant, rtol=1e-14, atol=0.0)
    np.testing.assert_array_equal(result.status, (0, 0, 1))
    np.testing.assert_array_equal(result.value[2], np.zeros(4))
    assert not bool(jnp.isfinite(result.condition_estimate[2]))


def test_single_precision_and_integer_inputs_keep_dtype_contracts() -> None:
    matrix = _system(3, (8,), False, 101)
    right = _right(3, (8,), None, False, 102)
    plan = SmallLinearSolvePlan(3)

    single = solve_small_linear(
        plan, jnp.asarray(matrix, jnp.float32), jnp.asarray(right, jnp.float32)
    )
    integral = solve_small_linear(
        plan,
        jnp.asarray(((2, 1, 0), (1, 3, 1), (0, 1, 4)), dtype=jnp.int32),
        jnp.asarray((1, 2, 3), dtype=jnp.int32),
    )

    assert single.value.dtype == jnp.float32
    assert single.condition_estimate.dtype == jnp.float32
    np.testing.assert_allclose(
        single.value,
        np.linalg.solve(matrix, right[..., None])[..., 0],
        rtol=1e-5,
        atol=1e-5,
    )
    assert integral.value.dtype == jnp.float64
    np.testing.assert_allclose(
        integral.value,
        np.linalg.solve(
            np.asarray(((2.0, 1.0, 0.0), (1.0, 3.0, 1.0), (0.0, 1.0, 4.0))),
            np.asarray((1.0, 2.0, 3.0)),
        ),
        rtol=1e-14,
    )
