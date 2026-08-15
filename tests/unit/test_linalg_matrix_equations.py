#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _dense_policy():
    return la.MatrixEquationPolicy(
        linear=la.LinearSolvePolicy(la.DenseLU()),
    )


def test_generalized_matrix_equation_operator_matches_terms_and_kronecker_matrix():
    left_one = jnp.asarray([[2.0, 1.0], [-1.0, 3.0]])
    right_one = jnp.asarray([[1.0, 2.0, 0.0], [0.0, -1.0, 1.0], [0.5, 0.0, 2.0]])
    left_two = jnp.asarray([[0.5, -0.25], [1.0, 0.75]])
    right_two = jnp.asarray([[2.0, 0.0, 1.0], [-1.0, 1.0, 0.0], [0.0, 0.5, -0.5]])
    terms = (
        la.MatrixEquationTerm(left_one, right_one, coefficient=1.5),
        la.MatrixEquationTerm(left_two, right_two, coefficient=-0.25),
    )
    operator = la.MatrixEquationLinearOperator(terms)
    value = jnp.asarray([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]])
    expected = 1.5 * left_one @ value @ right_one - 0.25 * left_two @ value @ right_two
    expected_matrix = 1.5 * jnp.kron(left_one, right_one.T) - 0.25 * jnp.kron(
        left_two, right_two.T
    )

    assert jnp.allclose(operator.mv(value), expected)
    assert jnp.allclose(operator._materialize(), expected_matrix)
    assert operator.source.shape == (2, 3)


def test_complex_matrix_equation_adjoint_satisfies_frobenius_identity():
    left = jnp.asarray([[1.0 + 1.0j, 2.0], [0.5j, -1.0]])
    right = jnp.asarray([[2.0, -1.0j], [0.25 + 0.5j, 3.0]])
    operator = la.MatrixEquationLinearOperator(
        (la.MatrixEquationTerm(left, right, coefficient=0.75 - 0.2j),)
    )
    value = jnp.asarray([[1.0 - 0.5j, 2.0], [-1.0j, 0.25]])
    cotangent = jnp.asarray([[0.5, -1.0j], [2.0 + 0.25j, -0.75]])

    forward_inner = jnp.vdot(cotangent, operator.mv(value))
    adjoint_inner = jnp.vdot(operator.adjoint_mv(cotangent), value)
    assert jnp.allclose(forward_inner, adjoint_inner, rtol=1e-12, atol=1e-12)


def test_rectangular_sylvester_solve_matches_dense_reference():
    left = jnp.asarray([[2.0, 1.0], [0.0, 3.0]])
    right = jnp.asarray([[4.0, -1.0, 0.5], [0.0, 5.0, 1.0], [0.0, 0.0, 6.0]])
    forcing = jnp.asarray([[1.0, 2.0, -1.0], [3.0, 4.0, 0.5]])
    problem = la.sylvester_equation(left, right, forcing)
    result = la.solve_matrix_equation(problem, policy=_dense_policy())
    kronecker = jnp.kron(left, jnp.eye(3)) + jnp.kron(jnp.eye(2), right.T)
    expected = jnp.linalg.solve(kronecker, forcing.reshape(-1)).reshape((2, 3))

    assert result.status == int(la.MatrixEquationStatus.SUCCESS)
    assert result.successful
    assert jnp.allclose(result.value, expected, rtol=1e-11, atol=1e-12)
    assert jnp.allclose(left @ result.value + result.value @ right, forcing)
    assert jnp.isnan(result.diagnostics.self_adjoint_error)
    assert result.provenance.convention == "A X + X B = C"


def test_continuous_and_discrete_lyapunov_factories_preserve_hermitian_structure():
    continuous_operator = jnp.asarray([[-1.0 + 0.5j, 2.0], [0.0, -3.0 - 0.25j]])
    continuous_forcing = jnp.asarray([[2.0, 0.5j], [-0.5j, 1.0]])
    continuous_problem = la.continuous_lyapunov_equation(
        continuous_operator,
        continuous_forcing,
    )
    continuous = la.solve_matrix_equation(
        continuous_problem,
        policy=_dense_policy(),
    )
    continuous_residual = (
        continuous_operator @ continuous.value
        + continuous.value @ jnp.conj(continuous_operator.T)
        + continuous_forcing
    )

    discrete_operator = jnp.asarray([[0.25, 0.1], [0.0, 0.5]])
    discrete_forcing = jnp.asarray([[1.0, 0.2], [0.2, 2.0]])
    discrete_problem = la.discrete_lyapunov_equation(
        discrete_operator,
        discrete_forcing,
    )
    discrete = la.solve_matrix_equation(discrete_problem, policy=_dense_policy())
    discrete_residual = (
        discrete.value
        - discrete_operator @ discrete.value @ discrete_operator.T
        - discrete_forcing
    )

    assert continuous.status == int(la.MatrixEquationStatus.SUCCESS)
    assert discrete.status == int(la.MatrixEquationStatus.SUCCESS)
    assert jnp.linalg.norm(continuous_residual) < 1e-11
    assert jnp.linalg.norm(discrete_residual) < 1e-11
    assert jnp.allclose(continuous.value, jnp.conj(continuous.value.T), atol=1e-11)
    assert jnp.allclose(discrete.value, discrete.value.T, atol=1e-11)
    assert continuous.diagnostics.structure_satisfied
    assert discrete.diagnostics.structure_satisfied


def test_prepared_matrix_equation_is_jittable_refreshable_and_accepts_new_forcing():
    first_left = jnp.asarray([[2.0, 0.5], [0.0, 3.0]])
    second_left = jnp.asarray([[1.5, -0.25], [0.25, 2.5]])
    right = jnp.asarray([[4.0, 0.25], [0.0, 5.0]])
    first_forcing = jnp.asarray([[1.0, 2.0], [3.0, -1.0]])
    second_forcing = jnp.asarray([[0.5, -2.0], [1.0, 4.0]])
    override = jnp.asarray([[-1.0, 0.25], [2.0, 0.5]])
    first_problem = la.sylvester_equation(
        first_left,
        right,
        first_forcing,
        problem_id="refreshable-matrix-equation",
    )
    second_problem = la.sylvester_equation(
        second_left,
        right,
        second_forcing,
        problem_id="refreshable-matrix-equation",
    )
    plan = la.plan_matrix_equation(first_problem, _dense_policy())
    prepared = la.prepare_matrix_equation(first_problem, plan)
    compiled = jax.jit(la.solve_matrix_equation)

    first = compiled(prepared)
    refreshed = la.refresh_matrix_equation(prepared, second_problem)
    second = compiled(refreshed)
    overridden = la.solve_matrix_equation(refreshed, right_hand_side=override)

    assert first.status == int(la.MatrixEquationStatus.SUCCESS)
    assert second.status == int(la.MatrixEquationStatus.SUCCESS)
    assert overridden.status == int(la.MatrixEquationStatus.SUCCESS)
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == 1
    assert second.provenance.numeric_version == 1
    assert jnp.allclose(second_left @ second.value + second.value @ right, second_forcing)
    assert jnp.allclose(
        second_left @ overridden.value + overridden.value @ right, override
    )


def test_prepared_matrix_equation_derivative_with_respect_to_forcing_is_correct():
    left = jnp.asarray([[2.0, 0.5], [0.0, 3.0]])
    right = jnp.asarray([[4.0, 0.25], [0.0, 5.0]])
    forcing = jnp.asarray([[1.0, 2.0], [3.0, -1.0]])
    problem = la.sylvester_equation(left, right, forcing)
    prepared = la.prepare_matrix_equation(problem, _dense_policy())
    kronecker = jnp.kron(left, jnp.eye(2)) + jnp.kron(jnp.eye(2), right.T)

    def actual_objective(rhs):
        value = la.solve_matrix_equation(prepared, right_hand_side=rhs).value
        return jnp.sum(value**2)

    def expected_objective(rhs):
        value = jnp.linalg.solve(kronecker, rhs.reshape(-1)).reshape((2, 2))
        return jnp.sum(value**2)

    actual = jax.jit(jax.grad(actual_objective))(forcing)
    expected = jax.grad(expected_objective)(forcing)
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_matrix_equation_validates_shape_structure_and_plan_identity():
    term = la.MatrixEquationTerm(jnp.eye(2), jnp.eye(3))
    with pytest.raises(ValueError, match="shape"):
        la.MatrixEquationProblem((term,), jnp.eye(2))

    with pytest.raises(ValueError, match="self-adjoint right-hand side"):
        la.continuous_lyapunov_equation(
            -jnp.eye(2),
            jnp.asarray([[1.0, 2.0], [0.0, 1.0]]),
        )

    first = la.sylvester_equation(
        jnp.eye(2),
        2.0 * jnp.eye(2),
        jnp.eye(2),
        problem_id="first-equation",
    )
    second = la.sylvester_equation(
        jnp.eye(2),
        2.0 * jnp.eye(2),
        jnp.eye(2),
        problem_id="second-equation",
    )
    plan = la.plan_matrix_equation(first)
    with pytest.raises(ValueError, match="different symbolic"):
        la.prepare_matrix_equation(second, plan)
