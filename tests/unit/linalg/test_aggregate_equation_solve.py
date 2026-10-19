from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.linalg import (
    AggregateLinearSolveResult,
    ArraySpace,
    FailurePolicy,
    FunctionLinearOperator,
    GMRES,
    LinearDerivativeSolvePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    plan,
    SolveResourcePolicy,
    TolerancePolicy,
)
from phydrax.linalg._constraints import OrthogonalHyperplaneConstraint
from phydrax.linalg._runtime import solve_aggregate


def _matrix(count: int) -> tuple[np.ndarray, np.ndarray]:
    mass = np.geomspace(0.003, 0.13, count, dtype=np.float64)
    incidence = np.eye(count, dtype=np.float64) - np.roll(
        np.eye(count, dtype=np.float64), 1, axis=1
    )
    stiffness = incidence.T @ (np.linspace(0.01, 0.03, count)[:, None] * incidence)
    return mass, stiffness


def test_aggregate_closed_sensitivity_and_original_residual_contract() -> None:
    mass, stiffness = _matrix(97)
    space = ArraySpace((len(mass),), dtype=np.float64)
    matrix = np.diag(mass) + 0.01 * stiffness
    operator = FunctionLinearOperator(
        lambda value: jnp.asarray(matrix) @ value,
        source=space,
        target=space,
        operator_id="closed-aggregate-transport",
    )
    initial = jnp.asarray(2.0 + np.cos(np.arange(len(mass))), dtype=jnp.float64)
    direction = jnp.sin(jnp.arange(len(mass), dtype=jnp.float64) + 1.0)

    def evolve(value: jax.Array) -> jax.Array:
        def advance(current: jax.Array, unused: None) -> tuple[jax.Array, None]:
            del unused
            result = solve_aggregate(
                LinearSystem(operator),
                jnp.asarray(mass) * current,
                jnp.ones_like(current),
            )
            return result.value, None

        return jax.lax.scan(advance, value, None, length=5)[0]

    primal, tangent = jax.jit(lambda value, vector: jax.jvp(evolve, (value,), (vector,)))(
        initial, direction
    )
    np.testing.assert_allclose(mass @ (primal - initial), 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(mass @ (tangent - direction), 0.0, atol=1e-12, rtol=0.0)
    expected = np.asarray(direction)
    for _ in range(5):
        expected = np.linalg.solve(matrix, mass * expected)
    # Each accepted linear residual bounds the propagated state error by A^-1.
    inverse_bound = np.linalg.norm(np.linalg.solve(matrix, np.eye(len(mass))), 2)
    rhs_bound = np.linalg.norm(mass * np.asarray(direction))
    per_step = inverse_bound * (1e-10 + 1e-8 * rhs_bound)
    assert np.linalg.norm(np.asarray(tangent) - expected) <= 5.0 * per_step
    single = jax.jit(
        lambda value: solve_aggregate(LinearSystem(operator), value, jnp.ones_like(value))
    )(jnp.asarray(mass) * initial)
    assert bool(single.successful)
    residual = matrix @ np.asarray(single.value) - mass * np.asarray(initial)
    np.testing.assert_allclose(
        single.full_residual_norm, np.linalg.norm(residual), atol=1e-15
    )
    assert float(single.full_residual_norm) <= float(single.full_residual_threshold)
    assert bool(single.chart_valid)
    assert np.isfinite(float(single.aggregate_row_norm))


def test_dynamic_operator_rhs_full_jvp_vjp_and_jit() -> None:
    mass, stiffness = _matrix(13)
    mass_direction = mass * np.linspace(0.1, 0.3, len(mass))
    loss = np.diag(np.linspace(0.001, 0.004, len(mass)))
    matrix = np.diag(mass) + 0.01 * stiffness + loss
    matrix_direction = np.diag(mass_direction) + 0.02 * stiffness + 0.3 * loss
    rhs = mass * (2.0 + np.cos(np.arange(len(mass))))
    rhs_direction = mass_direction + 0.001 * np.sin(np.arange(len(mass)))
    aggregate = jnp.linspace(0.5, 1.5, len(mass))
    space = ArraySpace((len(mass),), dtype=np.float64)

    def response(parameter: jax.Array, right: jax.Array) -> jax.Array:
        operator = FunctionLinearOperator(
            lambda value: (
                (jnp.asarray(matrix) + parameter * jnp.asarray(matrix_direction)) @ value
            ),
            source=space,
            target=space,
            operator_id="dynamic-open-aggregate",
        )
        return solve_aggregate(LinearSystem(operator), right, aggregate).value

    parameter = jnp.asarray(0.2, dtype=jnp.float64)
    right = jnp.asarray(rhs)
    seed = jnp.cos(jnp.arange(len(mass), dtype=jnp.float64) + 0.3)
    (value, tangent), (parameter_gradient, rhs_gradient) = jax.jit(
        lambda theta, b, cotangent: (
            jax.jvp(
                response, (theta, b), (jnp.ones_like(theta), jnp.asarray(rhs_direction))
            ),
            jax.vjp(response, theta, b)[1](cotangent),
        )
    )(parameter, right, seed)
    executed = matrix + 0.2 * matrix_direction
    expected_value = np.linalg.solve(executed, rhs)
    tangent_right = rhs_direction - matrix_direction @ np.asarray(value)
    tangent_residual = executed @ np.asarray(tangent) - tangent_right
    derivative_threshold = 1e-10 + 1e-8 * np.linalg.norm(tangent_right)
    assert np.linalg.norm(tangent_residual) <= derivative_threshold
    np.testing.assert_allclose(
        aggregate @ (executed @ tangent - tangent_right), 0.0, atol=1e-12, rtol=0.0
    )
    adjoint_residual = executed.T @ np.asarray(rhs_gradient) - np.asarray(seed)
    assert np.linalg.norm(adjoint_residual) <= 1e-10 + 1e-8 * np.linalg.norm(seed)
    inverse_bound = np.linalg.norm(np.linalg.solve(executed, np.eye(len(mass))), 2)
    np.testing.assert_allclose(
        value,
        expected_value,
        atol=inverse_bound * (1e-10 + 1e-8 * np.linalg.norm(rhs)),
        rtol=0.0,
    )
    expected_adjoint = np.linalg.solve(executed.T, seed)
    np.testing.assert_allclose(
        rhs_gradient,
        expected_adjoint,
        atol=inverse_bound * (1e-10 + 1e-8 * np.linalg.norm(seed)),
        rtol=0.0,
    )
    np.testing.assert_allclose(
        parameter_gradient,
        -np.asarray(rhs_gradient) @ matrix_direction @ np.asarray(value),
        atol=1e-12,
        rtol=0.0,
    )
    pairing_error = abs(
        float(seed @ tangent - parameter_gradient - rhs_gradient @ rhs_direction)
    )
    # Two independently inexact native solves have this residual-derived pairing bound.
    pairing_bound = np.linalg.norm(rhs_gradient) * np.linalg.norm(
        tangent_residual
    ) + np.linalg.norm(tangent) * np.linalg.norm(adjoint_residual)
    assert pairing_error <= pairing_bound + 128 * np.finfo(np.float64).eps
    changed = jax.jit(response)(jnp.asarray(0.4), right)
    np.testing.assert_allclose(
        changed,
        np.linalg.solve(matrix + 0.4 * matrix_direction, rhs),
        atol=inverse_bound * (1e-10 + 1e-8 * np.linalg.norm(rhs)),
        rtol=0.0,
    )


def test_failed_reduced_solve_retains_status_and_full_residual() -> None:
    mass, stiffness = _matrix(17)
    matrix = np.diag(mass) + stiffness
    space = ArraySpace(mass.shape, dtype=np.float64)
    operator = FunctionLinearOperator(
        lambda value: jnp.asarray(matrix) @ value,
        source=space,
        target=space,
        operator_id="failed-aggregate",
    )
    right = jnp.cos(jnp.arange(len(mass), dtype=jnp.float64))
    policy = LinearSolvePolicy(GMRES(restart=1), tolerance=TolerancePolicy(max_steps=1))
    result = jax.jit(
        lambda value: solve_aggregate(
            LinearSystem(operator), value, jnp.ones_like(value), policy=policy
        )
    )(right)
    assert not bool(result.successful)
    assert int(result.status) == int(result.reduced.status)
    assert int(result.status) != int(LinearSolveStatus.SUCCESS)
    residual = matrix @ np.asarray(result.value) - np.asarray(right)
    np.testing.assert_allclose(
        result.full_residual_norm, np.linalg.norm(residual), atol=1e-14
    )
    assert float(result.full_residual_norm) > float(result.full_residual_threshold)


def test_hyperplane_chart_dynamic_row_and_degenerate_evidence() -> None:
    space = ArraySpace((5,), dtype=np.float64)
    coordinates = jnp.asarray((0.2, -0.3, 0.7, 1.1), dtype=jnp.float64)

    def apply(row: jax.Array, right: jax.Array) -> tuple[jax.Array, jax.Array]:
        chart = OrthogonalHyperplaneConstraint(space, row, constraint_id="dynamic-row")
        expanded = chart.constraint.expand(coordinates, chart.lift(right))
        return expanded, chart.constraint.reduce_vector(
            chart.constraint.prolongation.mv(coordinates)
        )

    for row in (jnp.arange(1.0, 6.0), jnp.asarray((0.1, 0.2, -0.4, 0.7, -1.2))):
        expanded, recovered = jax.jit(apply)(row, jnp.asarray(2.3))
        np.testing.assert_allclose(row @ expanded, 2.3, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(recovered, coordinates, atol=1e-14, rtol=0.0)
    chart = OrthogonalHyperplaneConstraint(
        space, jnp.zeros((5,), dtype=jnp.float64), constraint_id="zero-row"
    )
    assert not bool(chart.valid)
    assert float(chart.row_norm) == 0.0


def test_failed_derivative_cannot_publish_an_aggregate_only_answer() -> None:
    mass, stiffness = _matrix(17)
    matrix = np.diag(mass) + stiffness
    space = ArraySpace(mass.shape, dtype=np.float64)
    operator = FunctionLinearOperator(
        lambda value: jnp.asarray(matrix) @ value,
        source=space,
        target=space,
        operator_id="failed-aggregate-derivative",
    )
    policy = LinearSolvePolicy(
        derivative_solve=LinearDerivativeSolvePolicy(maximum_steps=1)
    )

    def response(right: jax.Array) -> jax.Array:
        return solve_aggregate(
            LinearSystem(operator), right, jnp.ones_like(right), policy=policy
        ).value

    right = jnp.asarray(mass)
    direction = jnp.cos(jnp.arange(len(mass), dtype=jnp.float64))
    (primal, tangent), gradient = jax.jit(
        lambda b, vector: (
            jax.jvp(response, (b,), (vector,)),
            jax.vjp(response, b)[1](vector)[0],
        )
    )(right, direction)
    assert np.all(np.isfinite(primal))
    assert np.all(np.isnan(tangent))
    assert np.all(np.isnan(gradient))
    np.testing.assert_allclose(matrix @ primal, right, atol=1e-9, rtol=0.0)


def test_resource_refusal_retains_deficits_without_executing_a_solve() -> None:
    mass, stiffness = _matrix(17)
    matrix = np.diag(mass) + stiffness
    space = ArraySpace(mass.shape, dtype=np.float64)
    operator = FunctionLinearOperator(
        lambda value: jnp.asarray(matrix) @ value,
        source=space,
        target=space,
        operator_id="resource-refused-aggregate",
    )
    policy = LinearSolvePolicy(
        resources=SolveResourcePolicy(workspace_bytes=1, krylov_basis_bytes=64)
    )
    right = jnp.cos(jnp.arange(len(mass), dtype=jnp.float64))
    result = jax.jit(
        lambda b: solve_aggregate(
            LinearSystem(operator), b, jnp.ones_like(b), policy=policy
        )
    )(right)
    assert int(result.status) == int(LinearSolveStatus.RESOURCE_LIMIT_REACHED)
    assert not bool(result.successful)
    assert not result.reduced.provenance.prepared
    assert int(result.reduced.diagnostics.iterations) == 0
    assert int(result.reduced.diagnostics.matvec_count) == 0
    assert {name for name, _, _ in result.resource_refusals} == {
        "solve workspace",
        "Krylov basis",
    }
    assert all(
        required > available for _, required, available in result.resource_refusals
    )
    np.testing.assert_array_equal(result.value, np.zeros_like(mass))
    np.testing.assert_allclose(
        result.full_residual_norm, np.linalg.norm(right), atol=1e-14
    )
    with pytest.raises(ValueError, match="solve workspace"):
        plan(LinearSystem(operator), policy)
    error_policy = LinearSolvePolicy(
        resources=policy.resources, failure=FailurePolicy("error")
    )
    with pytest.raises(ValueError, match="solve workspace"):
        solve_aggregate(
            LinearSystem(operator), right, jnp.ones_like(right), policy=error_policy
        )


@pytest.mark.parametrize("host", (False, True))
def test_array_like_boundaries_and_dtype_refusals(host: bool) -> None:
    space = ArraySpace((2,), dtype=np.float64)
    operator = FunctionLinearOperator(
        lambda value: jnp.asarray(((2.0, 0.2), (0.1, 3.0))) @ value,
        source=space,
        target=space,
        operator_id="array-like-aggregate",
    )
    right = np.asarray((1.0, 2.0), dtype=np.float64)
    aggregate = np.asarray((1.0, -0.5), dtype=np.float64)
    result = solve_aggregate(
        LinearSystem(operator),
        right if host else jnp.asarray(right),
        aggregate if host else jnp.asarray(aggregate),
    )
    np.testing.assert_allclose(
        result.value,
        np.linalg.solve(((2.0, 0.2), (0.1, 3.0)), right),
        atol=1e-12,
        rtol=0.0,
    )
    chart = OrthogonalHyperplaneConstraint(
        space,
        aggregate if host else jnp.asarray(aggregate),
        constraint_id="array-like-row",
    )
    assert bool(chart.valid)
    for invalid in (
        right.astype(np.complex128),
        right.astype(np.float32),
    ):
        with pytest.raises(TypeError, match="dtype"):
            solve_aggregate(LinearSystem(operator), invalid, aggregate)
        with pytest.raises(TypeError, match="dtype"):
            solve_aggregate(LinearSystem(operator), right, invalid)
        with pytest.raises(TypeError, match="dtype"):
            OrthogonalHyperplaneConstraint(space, invalid, constraint_id="invalid-row")
    with pytest.raises(ValueError, match="shape"):
        solve_aggregate(LinearSystem(operator), right[:, None], aggregate)
    matrix_space = ArraySpace((1, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="rank-one"):
        solve_aggregate(
            LinearSystem(
                FunctionLinearOperator(
                    lambda value: value,
                    source=matrix_space,
                    target=matrix_space,
                    operator_id="rank-two-refusal",
                )
            ),
            right[None],
            aggregate[None],
        )


@pytest.mark.parametrize("scale", (1e-200, 1e200))
def test_scaled_chart_lift_adjoint_and_full_derivatives(scale: float) -> None:
    base = np.asarray((1.0, -2.0, 3.0, 4.0), dtype=np.float64)
    direction = np.asarray((0.2, 0.1, -0.3, 0.4), dtype=np.float64)
    beta, beta_direction = 2.3, -0.7
    space = ArraySpace(base.shape, dtype=np.float64)
    row = scale * base
    chart = OrthogonalHyperplaneConstraint(space, row, constraint_id="scaled-row")
    assert bool(chart.valid)
    np.testing.assert_allclose(
        chart.row_norm / scale, np.linalg.norm(base), atol=1e-14, rtol=0.0
    )
    coordinates = jnp.asarray((0.3, -0.4, 0.7), dtype=jnp.float64)
    complement = chart.constraint.prolongation.mv(coordinates)
    np.testing.assert_allclose(base @ complement, 0.0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(
        chart.constraint.reduce_vector(complement), coordinates, atol=1e-14, rtol=0.0
    )
    right = jnp.asarray(scale * beta, dtype=jnp.float64)
    lifted = chart.lift(right)
    expected = beta * base / (base @ base)
    np.testing.assert_allclose(lifted, expected, atol=1e-14, rtol=0.0)
    seed = jnp.asarray((0.5, -0.2, 0.4, 0.1), dtype=jnp.float64)
    adjoint = chart.transpose_lift(seed)
    _, transpose = jax.vjp(chart.lift, right)
    np.testing.assert_allclose(
        scale * transpose(seed)[0], scale * adjoint, atol=1e-14, rtol=0.0
    )
    np.testing.assert_allclose(seed @ lifted, right * adjoint, atol=1e-14, rtol=0.0)

    def response(parameter: jax.Array) -> tuple[jax.Array, jax.Array]:
        current = OrthogonalHyperplaneConstraint(
            space,
            scale * (jnp.asarray(base) + parameter * jnp.asarray(direction)),
            constraint_id="differentiated-scaled-row",
        )
        return (
            current.lift(scale * (beta + parameter * beta_direction)),
            current.row_norm / scale,
        )

    parameter = jnp.asarray(0.0, dtype=jnp.float64)
    (_, norm), (tangent, norm_tangent) = jax.jit(
        lambda theta: jax.jvp(response, (theta,), (jnp.ones_like(theta),))
    )(parameter)
    squared = base @ base
    expected_tangent = (
        beta_direction * base + beta * direction
    ) / squared - 2.0 * beta * base * (base @ direction) / squared**2
    np.testing.assert_allclose(tangent, expected_tangent, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(norm, np.linalg.norm(base), atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(
        norm_tangent, (base @ direction) / np.linalg.norm(base), atol=1e-14, rtol=0.0
    )
    gradient = jax.jit(
        lambda theta: jax.vjp(lambda p: response(p)[0], theta)[1](seed)[0]
    )(parameter)
    np.testing.assert_allclose(gradient, seed @ expected_tangent, atol=1e-14, rtol=0.0)


@pytest.mark.parametrize("scale", (1e-200, 1e200))
def test_scaled_aggregate_preserves_full_solve_and_implicit_ad(scale: float) -> None:
    space = ArraySpace((2,), dtype=np.float64)
    matrix = jnp.asarray(((2.0, 0.1), (0.2, 3.0)), dtype=jnp.float64)
    operator = FunctionLinearOperator(
        lambda value: matrix @ value,
        source=space,
        target=space,
        operator_id="scaled-aggregate-input",
    )
    right = jnp.asarray((1.2, -0.7), dtype=jnp.float64)
    vector = jnp.asarray((0.3, 0.8), dtype=jnp.float64)
    seed = jnp.asarray((-0.4, 0.2), dtype=jnp.float64)

    def response(value: jax.Array) -> jax.Array:
        return solve_aggregate(
            LinearSystem(operator),
            value,
            scale * jnp.asarray((1.0, 0.7), dtype=jnp.float64),
        ).value

    (value, tangent), gradient = jax.jit(
        lambda b: (
            jax.jvp(response, (b,), (vector,)),
            jax.vjp(response, b)[1](seed)[0],
        )
    )(right)
    np.testing.assert_allclose(
        value, np.linalg.solve(matrix, right), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        tangent, np.linalg.solve(matrix, vector), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        gradient, np.linalg.solve(matrix.T, seed), atol=1e-12, rtol=0.0
    )


def test_singleton_native_solve_and_zero_dimensional_chart() -> None:
    space = ArraySpace((1,), dtype=np.float64)
    chart = OrthogonalHyperplaneConstraint(
        space, np.asarray((2.0,), dtype=np.float64), constraint_id="singleton-chart"
    )
    assert chart.constraint.reduced_space.size == 0
    np.testing.assert_array_equal(
        chart.constraint.prolongation.mv(jnp.empty((0,), dtype=jnp.float64)),
        np.zeros((1,), dtype=np.float64),
    )
    assert chart.constraint.reduce_vector(jnp.ones((1,), dtype=jnp.float64)).shape == (0,)

    def response(parameter: jax.Array, right: jax.Array) -> jax.Array:
        operator = FunctionLinearOperator(
            lambda value: (2.0 + 0.3 * parameter) * value,
            source=space,
            target=space,
            operator_id="native-singleton",
        )
        return solve_aggregate(
            LinearSystem(operator), right, np.asarray((1.0,), dtype=np.float64)
        ).value

    theta = jnp.asarray(0.4, dtype=jnp.float64)
    right = jnp.asarray((1.7,), dtype=jnp.float64)
    (value, tangent), (theta_gradient, rhs_gradient) = jax.jit(
        lambda parameter, b: (
            jax.jvp(
                response, (parameter, b), (jnp.ones_like(parameter), jnp.ones_like(b))
            ),
            jax.vjp(response, parameter, b)[1](jnp.ones_like(b)),
        )
    )(theta, right)
    pivot = 2.0 + 0.3 * theta
    np.testing.assert_allclose(value, right / pivot, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(
        tangent, 1.0 / pivot - 0.3 * right / pivot**2, atol=1e-14, rtol=0.0
    )
    np.testing.assert_allclose(
        theta_gradient, -0.3 * right[0] / pivot**2, atol=1e-14, rtol=0.0
    )
    np.testing.assert_allclose(
        rhs_gradient, np.asarray((1.0,)) / pivot, atol=1e-14, rtol=0.0
    )


@pytest.mark.parametrize(
    "row",
    (
        np.zeros((3,), dtype=np.float64),
        np.asarray((1.0, np.inf, 2.0), dtype=np.float64),
        np.asarray((1.0, np.nan, 2.0), dtype=np.float64),
    ),
)
def test_invalid_chart_preserves_nonfinite_lift_evidence(row: np.ndarray) -> None:
    chart = OrthogonalHyperplaneConstraint(
        ArraySpace((3,), dtype=np.float64), row, constraint_id="invalid-chart"
    )
    assert not bool(chart.valid)
    assert np.all(np.isnan(chart.lift(jnp.asarray(1.0, dtype=jnp.float64))))


def _bank_reference_matrix(
    diagonal: np.ndarray, backward: np.ndarray, forward: np.ndarray
) -> np.ndarray:
    """Assemble an independent host reference for the periodic stencil."""
    count = len(diagonal)
    indices = np.arange(count)
    matrix = np.diag(diagonal)
    matrix[indices, (indices - 1) % count] += backward
    matrix[indices, (indices + 1) % count] += forward
    return matrix


def _bank_stencil_action(
    diagonal: jax.Array,
    backward: jax.Array,
    forward: jax.Array,
    value: jax.Array,
) -> jax.Array:
    return (
        diagonal * value + backward * jnp.roll(value, 1) + forward * jnp.roll(value, -1)
    )


def _bank_contraction_roundoff(
    bank: np.ndarray, matrix: np.ndarray, value: np.ndarray, right: np.ndarray
) -> np.ndarray:
    # Dimension-scaled rounding envelope for chart arithmetic and the independent
    # matvec/contractions, not the native iterative residual tolerance.
    magnitude = np.abs(matrix) @ np.abs(value) + np.abs(right)
    return 32.0 * len(right) * np.finfo(np.float64).eps * (np.abs(bank).T @ magnitude)


@pytest.mark.parametrize(
    "scales",
    ((1.0, 1.0, 1.0), (1e-200, 1e200, -3.0)),
    ids=("unit-columns", "individually-scaled-columns"),
)
def test_independent_nonorthogonal_bank_exact_rows_with_inexact_reduced_solve(
    scales: tuple[float, float, float],
) -> None:
    count = 23
    diagonal = np.linspace(1.0, 2.5, count, dtype=np.float64)
    backward = np.full((count,), -0.09, dtype=np.float64)
    forward = np.full((count,), -0.04, dtype=np.float64)
    matrix = _bank_reference_matrix(diagonal, backward, forward)
    bank = np.column_stack(
        (
            np.ones((count,), dtype=np.float64),
            np.linspace(-0.4, 1.3, count, dtype=np.float64),
            0.3 + np.cos(np.arange(count, dtype=np.float64)),
        )
    )
    scaled_bank = bank * np.asarray(scales, dtype=np.float64)[None, :]
    right = 1.1 + np.sin(np.arange(count, dtype=np.float64) + 0.2)
    space = ArraySpace((count,), dtype=np.float64)
    operator = FunctionLinearOperator(
        lambda value: _bank_stencil_action(
            jnp.asarray(diagonal), jnp.asarray(backward), jnp.asarray(forward), value
        ),
        source=space,
        target=space,
        operator_id="independent-nonorthogonal-bank",
    )
    result = jax.jit(
        lambda b: solve_aggregate(
            LinearSystem(operator),
            b,
            scaled_bank[:, 0],
            additional_aggregates=(scaled_bank[:, 1], scaled_bank[:, 2]),
        )
    )(jnp.asarray(right))
    assert bool(result.successful)
    assert int(result.status) == int(LinearSolveStatus.SUCCESS)
    assert int(result.reduced.status) == int(LinearSolveStatus.SUCCESS)
    assert bool(result.chart_valid)
    assert bool(result.aggregate_setup_performed)
    assert result.aggregate_count == 3
    assert int(result.aggregate_rank) == 3
    assert 1.0 <= float(result.aggregate_condition_estimate) < np.inf
    assert 0.0 < float(result.aggregate_rank_cutoff) < np.inf
    assert result.aggregate_row_norm.shape == (3,)
    assert np.all(np.isfinite(result.aggregate_row_norm))
    assert np.all(np.asarray(result.aggregate_row_norm) > 0.0)
    value = np.asarray(result.value)
    residual = matrix @ value - right
    rounding = _bank_contraction_roundoff(bank, matrix, value, right)
    assert np.all(np.abs(bank.T @ residual) <= rounding)
    assert result.aggregate_residual.shape == (3,)
    np.testing.assert_allclose(
        np.asarray(result.aggregate_residual) / np.asarray(scales),
        bank.T @ residual,
        atol=np.max(rounding),
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.full_residual_norm, np.linalg.norm(residual), atol=np.max(rounding)
    )
    assert float(result.full_residual_norm) <= float(result.full_residual_threshold)
    # This fixture stops at the frozen native tolerance, not at machine rounding.
    assert np.linalg.norm(residual) > 1024 * np.finfo(np.float64).eps * np.linalg.norm(
        right
    )
    inverse_bound = np.linalg.norm(np.linalg.solve(matrix, np.eye(count)), 2)
    assert np.linalg.norm(value - np.linalg.solve(matrix, right)) <= inverse_bound * (
        1e-10 + 1e-8 * np.linalg.norm(right)
    )


def test_physical_partition_inventory_rate_bank_balances_each_region() -> None:
    count = 12
    mass = np.geomspace(0.02, 0.3, count, dtype=np.float64)
    edges = np.linspace(0.08, 0.2, count, dtype=np.float64)
    dt = 0.015
    incidence = np.eye(count, dtype=np.float64) - np.roll(
        np.eye(count, dtype=np.float64), -1, axis=1
    )
    transport = incidence.T @ (edges[:, None] * incidence)
    matrix = np.eye(count, dtype=np.float64) + dt * transport / mass[None, :]
    # Two authored bulk regions and one network/reservoir inventory group.
    bank = np.zeros((count, 3), dtype=np.float64)
    bank[:5, 0] = 1.0
    bank[5:9, 1] = 1.0
    bank[9:, 2] = 1.0
    old = 1.5 + 0.2 * np.cos(np.arange(count, dtype=np.float64))
    right = mass * (0.4 + np.sin(np.arange(count, dtype=np.float64) + 0.3))
    space = ArraySpace((count,), dtype=np.float64)

    def inventory_action(rate: jax.Array) -> jax.Array:
        velocity = rate / jnp.asarray(mass)
        flux = jnp.asarray(edges) * (velocity - jnp.roll(velocity, 1))
        return rate + dt * (flux - jnp.roll(flux, -1))

    operator = FunctionLinearOperator(
        inventory_action,
        source=space,
        target=space,
        operator_id="physical-partition-inventory-rate",
    )
    result = jax.jit(
        lambda b: solve_aggregate(
            LinearSystem(operator),
            b,
            bank[:, 0],
            additional_aggregates=(bank[:, 1], bank[:, 2]),
        )
    )(jnp.asarray(right))
    assert bool(result.successful)
    assert bool(result.chart_valid)
    assert int(result.aggregate_rank) == 3
    rate = np.asarray(result.value)
    state = old + dt * rate / mass
    rounding = _bank_contraction_roundoff(bank, matrix, rate, right)
    residual = matrix @ rate - right
    assert np.all(np.abs(bank.T @ residual) <= rounding)
    expected_rate = np.linalg.solve(matrix, right)
    inverse_bound = np.linalg.norm(np.linalg.solve(matrix, np.eye(count)), 2)
    assert np.linalg.norm(rate - expected_rate) <= inverse_bound * (
        1e-10 + 1e-8 * np.linalg.norm(right)
    )
    # Inventory change includes cross-region exchange, rather than projecting
    # each region back to its old inventory.
    inventory_change = bank.T @ (mass * (state - old))
    balance = dt * (bank.T @ (right - dt * transport @ (rate / mass)))
    reconstruction_rounding = (
        8.0
        * np.finfo(np.float64).eps
        * (np.abs(bank).T @ (mass * (np.abs(state) + np.abs(old))))
    )
    assert np.all(
        np.abs(inventory_change - balance) <= dt * rounding + reconstruction_rounding
    )
    assert np.all(np.abs(inventory_change) > 100 * reconstruction_rounding)
    np.testing.assert_allclose(
        result.full_residual_norm, np.linalg.norm(residual), atol=np.max(rounding)
    )
    assert float(result.full_residual_norm) <= float(result.full_residual_threshold)


def test_nonconservative_bank_operator_rhs_jvp_vjp_match_independent_reference() -> None:
    count = 13
    diagonal = np.linspace(1.0, 2.0, count, dtype=np.float64)
    backward = np.linspace(-0.12, -0.04, count, dtype=np.float64)
    forward = np.linspace(-0.03, -0.08, count, dtype=np.float64)
    diagonal_direction = np.linspace(0.07, 0.2, count, dtype=np.float64)
    backward_direction = np.full((count,), 0.015, dtype=np.float64)
    forward_direction = np.linspace(-0.01, 0.01, count, dtype=np.float64)
    matrix = _bank_reference_matrix(diagonal, backward, forward)
    matrix_direction = _bank_reference_matrix(
        diagonal_direction, backward_direction, forward_direction
    )
    right = 0.8 + np.sin(np.arange(count, dtype=np.float64) + 0.1)
    right_direction = 0.3 * np.cos(np.arange(count, dtype=np.float64) + 0.4)
    bank = np.column_stack(
        (
            np.ones((count,), dtype=np.float64),
            np.linspace(0.2, 1.4, count, dtype=np.float64),
            np.sin(np.arange(count, dtype=np.float64) + 0.2),
        )
    )
    space = ArraySpace((count,), dtype=np.float64)

    def response(parameter: jax.Array, b: jax.Array) -> jax.Array:
        operator = FunctionLinearOperator(
            lambda value: _bank_stencil_action(
                jnp.asarray(diagonal) + parameter * jnp.asarray(diagonal_direction),
                jnp.asarray(backward) + parameter * jnp.asarray(backward_direction),
                jnp.asarray(forward) + parameter * jnp.asarray(forward_direction),
                value,
            ),
            source=space,
            target=space,
            operator_id="nonconservative-bank-derivative",
        )
        return solve_aggregate(
            LinearSystem(operator),
            b,
            bank[:, 0],
            additional_aggregates=(bank[:, 1], bank[:, 2]),
        ).value

    theta = jnp.asarray(0.3, dtype=jnp.float64)
    seed = jnp.cos(jnp.arange(count, dtype=jnp.float64) + 0.6)
    (value, tangent), (theta_gradient, rhs_gradient) = jax.jit(
        lambda parameter, b: (
            jax.jvp(
                response,
                (parameter, b),
                (jnp.ones_like(parameter), jnp.asarray(right_direction)),
            ),
            jax.vjp(response, parameter, b)[1](seed),
        )
    )(theta, jnp.asarray(right))
    executed = matrix + 0.3 * matrix_direction
    expected_value = np.linalg.solve(executed, right)
    inverse_bound = np.linalg.norm(np.linalg.solve(executed, np.eye(count)), 2)
    primal_threshold = 1e-10 + 1e-8 * np.linalg.norm(right)
    assert np.linalg.norm(np.asarray(value) - expected_value) <= (
        inverse_bound * primal_threshold
    )
    tangent_right = right_direction - matrix_direction @ np.asarray(value)
    tangent_residual = executed @ np.asarray(tangent) - tangent_right
    derivative_threshold = 1e-10 + 1e-8 * np.linalg.norm(tangent_right)
    assert np.linalg.norm(tangent_residual) <= derivative_threshold
    rounding = _bank_contraction_roundoff(
        bank, executed, np.asarray(tangent), tangent_right
    )
    assert np.all(np.abs(bank.T @ tangent_residual) <= rounding)
    expected_tangent = np.linalg.solve(
        executed, right_direction - matrix_direction @ expected_value
    )
    # The independent-reference tangent also includes propagation of primal residual.
    tangent_bound = inverse_bound * (
        derivative_threshold
        + np.linalg.norm(matrix_direction, 2) * inverse_bound * primal_threshold
    )
    assert np.linalg.norm(np.asarray(tangent) - expected_tangent) <= tangent_bound
    adjoint_residual = executed.T @ np.asarray(rhs_gradient) - np.asarray(seed)
    adjoint_threshold = 1e-10 + 1e-8 * np.linalg.norm(seed)
    assert np.linalg.norm(adjoint_residual) <= adjoint_threshold
    expected_adjoint = np.linalg.solve(executed.T, np.asarray(seed))
    assert np.linalg.norm(np.asarray(rhs_gradient) - expected_adjoint) <= (
        inverse_bound * adjoint_threshold
    )
    expected_theta_gradient = -expected_adjoint @ matrix_direction @ expected_value
    gradient_rounding = (
        128
        * np.finfo(np.float64).eps
        * np.linalg.norm(matrix_direction, 2)
        * np.linalg.norm(rhs_gradient)
        * np.linalg.norm(value)
    )
    gradient_bound = np.linalg.norm(matrix_direction, 2) * (
        inverse_bound * adjoint_threshold * np.linalg.norm(value)
        + np.linalg.norm(expected_adjoint) * inverse_bound * primal_threshold
    )
    assert abs(float(theta_gradient) - expected_theta_gradient) <= (
        gradient_bound + gradient_rounding
    )
    np.testing.assert_allclose(
        theta_gradient,
        -np.asarray(rhs_gradient) @ matrix_direction @ np.asarray(value),
        atol=gradient_rounding,
        rtol=0.0,
    )
    pairing_error = abs(
        float(seed @ tangent - theta_gradient - rhs_gradient @ right_direction)
    )
    pairing_bound = np.linalg.norm(rhs_gradient) * np.linalg.norm(
        tangent_residual
    ) + np.linalg.norm(tangent) * np.linalg.norm(adjoint_residual)
    assert pairing_error <= pairing_bound + 128 * np.finfo(np.float64).eps


def test_bank_jit_rebinds_coefficient_rhs_and_aggregate_leaves() -> None:
    count = 9
    space = ArraySpace((count,), dtype=np.float64)

    @jax.jit
    def execute(
        diagonal: jax.Array,
        backward: jax.Array,
        forward: jax.Array,
        right: jax.Array,
        primary: jax.Array,
        additional: jax.Array,
    ) -> jax.Array:
        operator = FunctionLinearOperator(
            lambda value: _bank_stencil_action(diagonal, backward, forward, value),
            source=space,
            target=space,
            operator_id="rebound-bank-coefficients",
        )
        return solve_aggregate(
            LinearSystem(operator),
            right,
            primary,
            additional_aggregates=(additional,),
        ).value

    previous = np.zeros((count,), dtype=np.float64)
    for shift in (0.0, 0.4):
        diagonal = np.linspace(1.0, 1.8, count, dtype=np.float64) + shift
        backward = np.full((count,), -0.05 - 0.02 * shift, dtype=np.float64)
        forward = np.full((count,), -0.03 + 0.01 * shift, dtype=np.float64)
        right = 0.7 + np.cos(np.arange(count, dtype=np.float64) + shift)
        bank = np.column_stack(
            (
                np.ones((count,), dtype=np.float64) + shift,
                np.linspace(-0.5, 1.0, count, dtype=np.float64) + shift,
            )
        )
        value = np.asarray(
            execute(
                jnp.asarray(diagonal),
                jnp.asarray(backward),
                jnp.asarray(forward),
                jnp.asarray(right),
                jnp.asarray(bank[:, 0]),
                jnp.asarray(bank[:, 1]),
            )
        )
        matrix = _bank_reference_matrix(diagonal, backward, forward)
        inverse_bound = np.linalg.norm(np.linalg.solve(matrix, np.eye(count)), 2)
        assert np.linalg.norm(value - np.linalg.solve(matrix, right)) <= (
            inverse_bound * (1e-10 + 1e-8 * np.linalg.norm(right))
        )
        rounding = _bank_contraction_roundoff(bank, matrix, value, right)
        assert np.all(np.abs(bank.T @ (matrix @ value - right)) <= rounding)
        assert np.linalg.norm(value - previous) > 1e-3
        previous = value


def test_failed_bank_reduction_preserves_exact_rows_and_original_failure() -> None:
    count = 17
    diagonal = np.geomspace(0.1, 4.0, count, dtype=np.float64)
    backward = np.full((count,), -0.025, dtype=np.float64)
    forward = np.full((count,), -0.01, dtype=np.float64)
    matrix = _bank_reference_matrix(diagonal, backward, forward)
    bank = np.column_stack(
        (
            np.ones((count,), dtype=np.float64),
            np.linspace(-0.3, 1.2, count, dtype=np.float64),
        )
    )
    right = np.cos(np.arange(count, dtype=np.float64) + 0.2)
    space = ArraySpace((count,), dtype=np.float64)
    operator = FunctionLinearOperator(
        lambda value: _bank_stencil_action(
            jnp.asarray(diagonal), jnp.asarray(backward), jnp.asarray(forward), value
        ),
        source=space,
        target=space,
        operator_id="failed-bank-reduction",
    )
    policy = LinearSolvePolicy(GMRES(restart=1), tolerance=TolerancePolicy(max_steps=1))
    result = jax.jit(
        lambda b: solve_aggregate(
            LinearSystem(operator),
            b,
            bank[:, 0],
            additional_aggregates=(bank[:, 1],),
            policy=policy,
        )
    )(jnp.asarray(right))
    assert not bool(result.successful)
    assert int(result.status) == int(result.reduced.status)
    assert int(result.status) != int(LinearSolveStatus.SUCCESS)
    assert bool(result.chart_valid)
    assert int(result.aggregate_rank) == 2
    residual = matrix @ np.asarray(result.value) - right
    rounding = _bank_contraction_roundoff(bank, matrix, np.asarray(result.value), right)
    assert np.all(np.abs(bank.T @ residual) <= rounding)
    np.testing.assert_allclose(
        result.aggregate_residual, bank.T @ residual, atol=np.max(rounding), rtol=0.0
    )
    np.testing.assert_allclose(
        result.full_residual_norm, np.linalg.norm(residual), atol=np.max(rounding)
    )
    assert float(result.full_residual_norm) > float(result.full_residual_threshold)


@pytest.mark.parametrize("duplicate_scale", (0.0, 2.0), ids=("zero-row", "duplicate-row"))
def test_dependent_bank_refuses_without_claiming_a_solution(
    duplicate_scale: float,
) -> None:
    count = 5
    space = ArraySpace((count,), dtype=np.float64)
    diagonal = jnp.linspace(1.0, 2.0, count, dtype=jnp.float64)
    operator = FunctionLinearOperator(
        lambda value: diagonal * value,
        source=space,
        target=space,
        operator_id="dependent-aggregate-bank",
    )
    primary = jnp.arange(1.0, count + 1.0, dtype=jnp.float64)
    result = jax.jit(
        lambda b: solve_aggregate(
            LinearSystem(operator),
            b,
            primary,
            additional_aggregates=(duplicate_scale * primary,),
        )
    )(jnp.ones((count,), dtype=jnp.float64))
    assert int(result.status) == int(LinearSolveStatus.RANK_DEFICIENT)
    assert not bool(result.successful)
    assert not bool(result.chart_valid)
    assert bool(result.aggregate_setup_performed)
    assert result.aggregate_count == 2
    assert int(result.aggregate_rank) == 1
    assert np.all(np.isnan(result.value))
    assert not bool(result.reduced.diagnostics.converged)
    assert int(result.reduced.diagnostics.iterations) == 0


def test_bank_resource_refusal_precedes_setup_and_preserves_unknown_rank() -> None:
    count = 17
    space = ArraySpace((count,), dtype=np.float64)
    operator = FunctionLinearOperator(
        lambda value: jnp.linspace(1.0, 3.0, count, dtype=jnp.float64) * value,
        source=space,
        target=space,
        operator_id="preflight-refused-aggregate-bank",
    )
    policy = LinearSolvePolicy(
        resources=SolveResourcePolicy(workspace_bytes=1, krylov_basis_bytes=64)
    )
    right = jnp.cos(jnp.arange(count, dtype=jnp.float64))
    result = jax.jit(
        lambda b: solve_aggregate(
            LinearSystem(operator),
            b,
            jnp.ones_like(b),
            additional_aggregates=(jnp.full_like(b, jnp.nan),),
            policy=policy,
        )
    )(right)
    assert int(result.status) == int(LinearSolveStatus.RESOURCE_LIMIT_REACHED)
    assert not bool(result.successful)
    assert not bool(result.aggregate_setup_performed)
    assert not bool(result.chart_valid)
    assert result.aggregate_count == 2
    assert int(result.aggregate_rank) == -1
    assert np.isnan(float(result.aggregate_condition_estimate))
    assert np.isnan(float(result.aggregate_rank_cutoff))
    assert np.all(np.isnan(result.aggregate_row_norm))
    assert not result.reduced.provenance.prepared
    assert int(result.reduced.diagnostics.iterations) == 0
    assert int(result.reduced.diagnostics.matvec_count) == 0
    assert result.resource_refusals
    assert all(
        required > available for _, required, available in result.resource_refusals
    )
    np.testing.assert_array_equal(result.value, np.zeros((count,), dtype=np.float64))
    np.testing.assert_allclose(
        result.full_residual_norm, np.linalg.norm(right), atol=1e-14, rtol=0.0
    )


def test_full_bank_exact_elimination_has_no_reduced_solve_and_full_implicit_ad() -> None:
    count = 4
    diagonal = np.asarray((1.2, 1.5, 1.8, 2.1), dtype=np.float64)
    backward = np.asarray((-0.1, -0.07, -0.04, -0.09), dtype=np.float64)
    forward = np.asarray((-0.03, -0.05, -0.08, -0.02), dtype=np.float64)
    diagonal_direction = np.asarray((0.1, -0.07, 0.04, 0.12), dtype=np.float64)
    bank = np.asarray(
        (
            (1.0, 1.0, 0.0, 0.0),
            (1.0, 2.0, 1.0, 0.0),
            (1.0, 0.0, 2.0, 1.0),
            (1.0, -1.0, 0.0, 3.0),
        ),
        dtype=np.float64,
    )
    right = np.asarray((0.4, -0.7, 1.1, 0.8), dtype=np.float64)
    right_direction = np.asarray((0.2, 0.3, -0.1, 0.4), dtype=np.float64)
    seed = jnp.asarray((0.6, -0.2, 0.5, 0.1), dtype=jnp.float64)
    space = ArraySpace((count,), dtype=np.float64)

    def execute(parameter: jax.Array, b: jax.Array) -> AggregateLinearSolveResult:
        operator = FunctionLinearOperator(
            lambda value: _bank_stencil_action(
                jnp.asarray(diagonal) + parameter * jnp.asarray(diagonal_direction),
                jnp.asarray(backward),
                jnp.asarray(forward),
                value,
            ),
            source=space,
            target=space,
            operator_id="full-bank-exact-elimination",
        )
        return solve_aggregate(
            LinearSystem(operator),
            b,
            bank[:, 0],
            additional_aggregates=(bank[:, 1], bank[:, 2], bank[:, 3]),
        )

    def response(parameter: jax.Array, b: jax.Array) -> jax.Array:
        return execute(parameter, b).value

    theta = jnp.asarray(0.3, dtype=jnp.float64)
    result = jax.jit(execute)(theta, jnp.asarray(right))
    assert bool(result.successful)
    assert bool(result.chart_valid)
    assert bool(result.aggregate_setup_performed)
    assert result.aggregate_count == count
    assert int(result.aggregate_rank) == count
    assert not result.reduced.provenance.prepared
    assert result.reduced.value.shape == (0,)
    assert int(result.reduced.diagnostics.iterations) == 0
    assert int(result.reduced.diagnostics.matvec_count) == 0
    assert int(result.reduced.diagnostics.rank) == 0
    assert int(result.reduced.diagnostics.nullity) == 0
    assert np.isnan(float(result.reduced.diagnostics.condition_estimate))
    matrix = _bank_reference_matrix(
        diagonal + 0.3 * diagonal_direction, backward, forward
    )
    value = np.asarray(result.value)
    residual = matrix @ value - right
    rounding = _bank_contraction_roundoff(bank, matrix, value, right)
    assert np.all(np.abs(bank.T @ residual) <= rounding)
    np.testing.assert_allclose(
        result.aggregate_residual, bank.T @ residual, atol=np.max(rounding), rtol=0.0
    )
    np.testing.assert_allclose(
        result.full_residual_norm, np.linalg.norm(residual), atol=np.max(rounding)
    )
    assert float(result.full_residual_norm) <= float(result.full_residual_threshold)
    inverse_bound = np.linalg.norm(np.linalg.solve(matrix, np.eye(count)), 2)
    assert np.linalg.norm(value - np.linalg.solve(matrix, right)) <= (
        inverse_bound * np.linalg.norm(rounding)
    )
    (_, tangent), (theta_gradient, rhs_gradient) = jax.jit(
        lambda parameter, b: (
            jax.jvp(
                response,
                (parameter, b),
                (jnp.ones_like(parameter), jnp.asarray(right_direction)),
            ),
            jax.vjp(response, parameter, b)[1](seed),
        )
    )(theta, jnp.asarray(right))
    tangent_right = right_direction - diagonal_direction * value
    tangent_rounding = _bank_contraction_roundoff(
        bank, matrix, np.asarray(tangent), tangent_right
    )
    assert np.linalg.norm(matrix @ np.asarray(tangent) - tangent_right) <= np.linalg.norm(
        tangent_rounding
    )
    adjoint_rounding = _bank_contraction_roundoff(
        bank, matrix.T, np.asarray(rhs_gradient), np.asarray(seed)
    )
    assert np.linalg.norm(matrix.T @ np.asarray(rhs_gradient) - np.asarray(seed)) <= (
        np.linalg.norm(adjoint_rounding)
    )
    np.testing.assert_allclose(
        theta_gradient,
        -np.asarray(rhs_gradient) @ (diagonal_direction * value),
        atol=128 * np.finfo(np.float64).eps,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("invalid", "error", "match"),
    (
        (np.ones((3,), dtype=np.float32), TypeError, "dtype"),
        (np.ones((3,), dtype=np.complex128), TypeError, "dtype"),
        (np.ones((3, 1), dtype=np.float64), ValueError, "shape"),
        (np.ones((2,), dtype=np.float64), ValueError, "shape"),
    ),
    ids=("float32", "complex", "rank-two", "wrong-extent"),
)
def test_additional_bank_rows_use_canonical_space_validation(
    invalid: np.ndarray, error: type[Exception], match: str
) -> None:
    space = ArraySpace((3,), dtype=np.float64)
    operator = FunctionLinearOperator(
        lambda value: 2.0 * value,
        source=space,
        target=space,
        operator_id="additional-aggregate-validation",
    )
    with pytest.raises(error, match=match):
        solve_aggregate(
            LinearSystem(operator),
            np.ones((3,), dtype=np.float64),
            np.ones((3,), dtype=np.float64),
            additional_aggregates=(invalid,),
        )


@pytest.mark.parametrize("invalid", (np.inf, np.nan), ids=("infinite", "nan"))
def test_nonfinite_additional_bank_row_is_not_a_valid_chart(invalid: float) -> None:
    space = ArraySpace((4,), dtype=np.float64)
    operator = FunctionLinearOperator(
        lambda value: jnp.asarray((1.0, 1.2, 1.4, 1.6), dtype=jnp.float64) * value,
        source=space,
        target=space,
        operator_id="nonfinite-additional-aggregate",
    )
    result = jax.jit(
        lambda b: solve_aggregate(
            LinearSystem(operator),
            b,
            jnp.ones_like(b),
            additional_aggregates=(
                jnp.asarray((0.0, 1.0, invalid, 0.0), dtype=jnp.float64),
            ),
        )
    )(jnp.ones((4,), dtype=jnp.float64))
    assert not bool(result.successful)
    assert not bool(result.chart_valid)
    assert bool(result.aggregate_setup_performed)
    assert int(result.status) == int(LinearSolveStatus.NONFINITE_INPUT)
    assert np.all(np.isnan(result.value))
