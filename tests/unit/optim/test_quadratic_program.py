#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

import phydrax as phx


def test_unconstrained_and_equality_solutions_have_audited_kkt_residuals():
    unconstrained = phx.optim.QuadraticProgram(
        jnp.diag(jnp.array([2.0, 4.0])),
        jnp.array([-4.0, 8.0]),
    )
    result = phx.optim.solve_quadratic_program(unconstrained)
    np.testing.assert_allclose(result.primal, jnp.array([2.0, -2.0]), atol=1e-10)
    assert result.status == phx.optim.QP_SUCCESS
    assert result.valid
    assert result.backend == "phydrax"
    assert result.method == "dense-primal-dual"
    np.testing.assert_allclose(result.stationarity_residual, 0.0, atol=1e-10)
    assert result.kkt_residual_norm < 1e-10

    equality = phx.optim.QuadraticProgram(
        jnp.eye(2),
        jnp.array([-2.0, -5.0]),
        equality_matrix=jnp.array([[1.0, 1.0]]),
        equality_rhs=jnp.array([3.0]),
    )
    equality_result = phx.optim.solve_quadratic_program(equality)
    np.testing.assert_allclose(equality_result.primal, jnp.array([0.0, 3.0]), atol=1e-10)
    np.testing.assert_allclose(
        equality_result.equality_dual, jnp.array([2.0]), atol=1e-10
    )
    np.testing.assert_allclose(equality_result.equality_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(equality_result.stationarity_residual, 0.0, atol=1e-10)


def test_nonsymmetric_quadratic_uses_symmetric_part_in_solve_and_vjp():
    quadratic = jnp.array([[1.0, 1.0], [-1.0, 1.0]])
    linear = jnp.array([-1.0, 0.0])
    problem = phx.optim.QuadraticProgram(quadratic, linear)

    np.testing.assert_allclose(problem.quadratic, jnp.eye(2), atol=1e-10)
    result = phx.optim.solve_quadratic_program(problem)
    np.testing.assert_allclose(result.primal, jnp.array([1.0, 0.0]), atol=1e-10)
    np.testing.assert_allclose(result.objective, -0.5, atol=1e-10)
    np.testing.assert_allclose(result.stationarity_residual, 0.0, atol=1e-10)
    assert result.status == phx.optim.QP_SUCCESS

    def primal_sum(raw_quadratic):
        raw_problem = phx.optim.QuadraticProgram(raw_quadratic, linear)
        return jnp.sum(phx.optim.solve_quadratic_program_primal(raw_problem))

    np.testing.assert_allclose(
        jax.grad(primal_sum)(quadratic),
        jnp.array([[-1.0, -0.5], [-0.5, 0.0]]),
        atol=1e-10,
    )


def test_rank_revealing_direct_solve_handles_singular_and_redundant_systems():
    zero_hessian = phx.optim.QuadraticProgram(jnp.zeros((1, 1)), jnp.zeros(1))
    zero_result = phx.optim.solve_quadratic_program(zero_hessian)
    np.testing.assert_allclose(zero_result.primal, 0.0, atol=1e-10)
    assert zero_result.status == phx.optim.QP_SUCCESS
    assert zero_result.kkt_residual_norm == 0.0
    assert zero_result.iterations == 0

    redundant = phx.optim.QuadraticProgram(
        jnp.eye(1),
        jnp.zeros(1),
        equality_matrix=jnp.array([[1.0], [1.0]]),
        equality_rhs=jnp.zeros(2),
    )
    redundant_result = phx.optim.solve_quadratic_program(redundant)
    np.testing.assert_allclose(redundant_result.primal, 0.0, atol=1e-10)
    np.testing.assert_allclose(redundant_result.equality_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(redundant_result.stationarity_residual, 0.0, atol=1e-10)
    assert redundant_result.status == phx.optim.QP_SUCCESS


def test_rank_revealing_direct_solve_rejects_inconsistent_and_unbounded_systems():
    inconsistent = phx.optim.QuadraticProgram(
        jnp.eye(1),
        jnp.zeros(1),
        equality_matrix=jnp.array([[1.0], [1.0]]),
        equality_rhs=jnp.array([0.0, 1.0]),
    )
    inconsistent_result = phx.optim.solve_quadratic_program(inconsistent)
    assert inconsistent_result.status == phx.optim.QP_INFEASIBLE
    assert not inconsistent_result.valid
    assert jnp.max(jnp.abs(inconsistent_result.equality_residual)) > 0.4

    unbounded = phx.optim.QuadraticProgram(jnp.zeros((1, 1)), jnp.array([-1.0]))
    unbounded_result = phx.optim.solve_quadratic_program(unbounded)
    assert unbounded_result.status == phx.optim.QP_MAX_ITERATIONS
    assert not unbounded_result.valid
    np.testing.assert_allclose(unbounded_result.stationarity_residual, jnp.array([-1.0]))
    assert unbounded_result.kkt_residual_norm == 1.0


def test_inequality_and_mixed_solutions_identify_active_constraints():
    inequality = phx.optim.QuadraticProgram(
        jnp.eye(2),
        jnp.array([-2.0, -5.0]),
        inequality_matrix=jnp.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]),
        inequality_rhs=jnp.array([0.0, 0.0, 3.0]),
    )
    result = phx.optim.solve_quadratic_program(inequality, tolerance=1e-7)
    np.testing.assert_allclose(result.primal, jnp.array([0.0, 3.0]), atol=2e-4)
    assert result.status == phx.optim.QP_SUCCESS
    assert result.primal_residual_norm < 1e-7
    assert result.dual_residual_norm < 1e-7
    assert jnp.max(jnp.abs(result.complementarity_residual)) < 1e-7
    assert result.inequality_dual[-1] > 1.9
    assert result.inequality_slack[-1] < 1e-6

    mixed = phx.optim.QuadraticProgram(
        jnp.diag(jnp.array([2.0, 1.0])),
        jnp.array([-2.0, -4.0]),
        equality_matrix=jnp.array([[1.0, -1.0]]),
        equality_rhs=jnp.array([0.0]),
        inequality_matrix=jnp.array([[-1.0, 0.0], [1.0, 0.0]]),
        inequality_rhs=jnp.array([0.0, 1.0]),
    )
    mixed_result = phx.optim.solve_quadratic_program(mixed)
    np.testing.assert_allclose(mixed_result.primal, jnp.ones(2), atol=1e-5)
    np.testing.assert_allclose(mixed_result.equality_residual, 0.0, atol=1e-7)
    assert mixed_result.inequality_slack[1] < 1e-6
    assert mixed_result.kkt_residual_norm < 1e-7


def test_infeasible_and_nonfinite_inputs_have_distinct_explicit_statuses():
    infeasible = phx.optim.QuadraticProgram(
        jnp.eye(1),
        jnp.zeros(1),
        inequality_matrix=jnp.array([[1.0], [-1.0]]),
        inequality_rhs=jnp.array([0.0, -1.0]),
    )
    infeasible_result = phx.optim.solve_quadratic_program(infeasible)
    assert infeasible_result.status == phx.optim.QP_INFEASIBLE
    assert not infeasible_result.valid
    assert jnp.all(jnp.isfinite(infeasible_result.primal))

    inconsistent_equalities = phx.optim.QuadraticProgram(
        jnp.eye(1),
        jnp.zeros(1),
        equality_matrix=jnp.array([[1.0], [1.0]]),
        equality_rhs=jnp.array([0.0, 1.0]),
    )
    equality_result = phx.optim.solve_quadratic_program(inconsistent_equalities)
    assert equality_result.status == phx.optim.QP_INFEASIBLE
    assert not equality_result.valid

    nonfinite = phx.optim.QuadraticProgram(
        jnp.eye(1),
        jnp.array([jnp.nan]),
    )
    nonfinite_result = phx.optim.solve_quadratic_program(nonfinite)
    assert nonfinite_result.status == phx.optim.QP_NONFINITE
    assert not nonfinite_result.valid
    assert jnp.isnan(nonfinite_result.primal[0])


def test_batch_shapes_broadcast_constraints_and_preserve_per_case_status():
    problem = phx.optim.QuadraticProgram(
        jnp.broadcast_to(jnp.eye(2), (3, 2, 2)),
        jnp.array([[-1.0, -2.0], [-2.0, -4.0], [jnp.nan, 0.0]]),
        inequality_matrix=jnp.array([[-1.0, 0.0], [0.0, -1.0]]),
        inequality_rhs=jnp.zeros(2),
    )
    assert problem.batch_shape == (3,)
    assert problem.inequality_matrix.shape == (3, 2, 2)
    result = jax.jit(phx.optim.solve_quadratic_program)(problem)
    np.testing.assert_allclose(
        result.primal[:2], jnp.array([[1.0, 2.0], [2.0, 4.0]]), atol=1e-5
    )
    np.testing.assert_array_equal(
        result.status,
        jnp.array([phx.optim.QP_SUCCESS, phx.optim.QP_SUCCESS, phx.optim.QP_NONFINITE]),
    )
    np.testing.assert_array_equal(result.valid, jnp.array([True, True, False]))
    assert result.objective.shape == (3,)
    assert result.kkt_residual_norm.shape == (3,)


def test_batched_inequality_solver_preserves_early_completion_iterations():
    problem = phx.optim.QuadraticProgram(
        jnp.ones((1, 1)),
        jnp.array([-2.0]),
        inequality_matrix=jnp.array([[1.0], [-1.0]]),
        inequality_rhs=jnp.array([[0.0, -1.0], [3.0, 0.0]]),
    )
    result = jax.jit(phx.optim.solve_quadratic_program)(problem)

    np.testing.assert_array_equal(
        result.status,
        jnp.array([phx.optim.QP_INFEASIBLE, phx.optim.QP_SUCCESS]),
    )
    assert result.iterations[0] == 0
    assert result.iterations[1] > 0
    np.testing.assert_allclose(result.primal[1], jnp.array([2.0]), atol=1e-5)


def test_dense_result_matches_independent_scipy_oracle():
    quadratic = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    linear = np.array([-3.0, -2.0, -1.0])
    equality_matrix = np.array([[1.0, 1.0, 1.0]])
    equality_rhs = np.array([1.5])
    inequality_matrix = np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    inequality_rhs = np.array([0.0, 0.0, 0.0, 0.8])
    problem = phx.optim.QuadraticProgram(
        quadratic,
        linear,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality_matrix,
        inequality_rhs=inequality_rhs,
    )
    result = phx.optim.solve_quadratic_program(problem, tolerance=1e-9)

    def objective(x):
        return 0.5 * x @ quadratic @ x + linear @ x

    scipy_result = minimize(
        objective,
        np.full(3, 0.5),
        jac=lambda x: quadratic @ x + linear,
        hess=lambda _: quadratic,
        constraints=(
            LinearConstraint(equality_matrix, equality_rhs, equality_rhs),
            LinearConstraint(inequality_matrix, -np.inf, inequality_rhs),
        ),
        method="trust-constr",
        options={"gtol": 1e-12, "xtol": 1e-12, "barrier_tol": 1e-12},
    )
    assert scipy_result.success
    np.testing.assert_allclose(result.primal, scipy_result.x, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(result.objective, scipy_result.fun, atol=1e-8)
    assert result.kkt_residual_norm < 1e-9


def test_active_set_gradients_match_piecewise_analytic_sensitivities():
    quadratic = jnp.eye(2)
    inequality_matrix = -jnp.eye(2)
    inequality_rhs = jnp.zeros(2)

    def solution(linear):
        problem = phx.optim.QuadraticProgram(
            quadratic,
            linear,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
        )
        return phx.optim.solve_quadratic_program_primal(problem)

    inactive_jacobian = jax.jacrev(solution)(jnp.array([-2.0, -3.0]))
    np.testing.assert_allclose(inactive_jacobian, -jnp.eye(2), atol=1e-7)
    active_jacobian = jax.jacrev(solution)(jnp.array([2.0, -3.0]))
    np.testing.assert_allclose(
        active_jacobian,
        jnp.array([[0.0, 0.0], [0.0, -1.0]]),
        atol=1e-7,
    )

    def duplicate_active_solution(linear):
        problem = phx.optim.QuadraticProgram(
            jnp.ones((1, 1)),
            linear,
            inequality_matrix=jnp.array([[-1.0], [-1.0]]),
            inequality_rhs=jnp.zeros(2),
        )
        return phx.optim.solve_quadratic_program_primal(problem)

    duplicate_jacobian = jax.jacrev(duplicate_active_solution)(jnp.array([2.0]))
    assert jnp.all(jnp.isfinite(duplicate_jacobian))
    np.testing.assert_allclose(duplicate_jacobian, jnp.zeros((1, 1)), atol=1e-7)

    def equality_solution(rhs):
        problem = phx.optim.QuadraticProgram(
            jnp.eye(2),
            jnp.zeros(2),
            equality_matrix=jnp.array([[1.0, 1.0]]),
            equality_rhs=rhs,
        )
        return phx.optim.solve_quadratic_program_primal(problem)

    np.testing.assert_allclose(
        jax.jacrev(equality_solution)(jnp.array([2.0])),
        jnp.array([[0.5], [0.5]]),
        atol=1e-7,
    )


def test_explicit_regularization_is_recorded_and_not_hidden_in_raw_kkt_data():
    problem = phx.optim.QuadraticProgram(jnp.zeros((1, 1)), jnp.array([-1.0]))
    result = phx.optim.solve_quadratic_program(problem, regularization=2.0)
    np.testing.assert_allclose(result.primal, jnp.array([0.5]), atol=1e-10)
    np.testing.assert_allclose(result.stationarity_residual, jnp.array([-1.0]))
    np.testing.assert_allclose(result.solver_stationarity_residual, 0.0, atol=1e-10)
    assert result.regularization == 2.0
    assert result.status == phx.optim.QP_SUCCESS


def test_dense_primal_dual_solver_supports_float32_without_dtype_repair():
    problem = phx.optim.QuadraticProgram(
        jnp.eye(2, dtype=jnp.float32),
        jnp.array([-2.0, -5.0], dtype=jnp.float32),
        inequality_matrix=jnp.array(
            [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]], dtype=jnp.float32
        ),
        inequality_rhs=jnp.array([0.0, 0.0, 3.0], dtype=jnp.float32),
    )
    result = phx.optim.solve_quadratic_program(problem)
    assert result.primal.dtype == jnp.float32
    assert result.status == phx.optim.QP_SUCCESS
    assert result.kkt_residual_norm < 1e-7


@pytest.mark.parametrize(
    "constraint_data",
    [
        {
            "equality_matrix": np.array([[1.0 + 1.0j]]),
            "equality_rhs": np.array([1.0]),
        },
        {
            "equality_matrix": np.array([[1.0]]),
            "equality_rhs": np.array([1.0 + 1.0j]),
        },
        {
            "inequality_matrix": np.array([[1.0 + 1.0j]]),
            "inequality_rhs": np.array([1.0]),
        },
        {
            "inequality_matrix": np.array([[1.0]]),
            "inequality_rhs": np.array([1.0 + 1.0j]),
        },
    ],
)
def test_complex_constraint_data_is_rejected_before_target_dtype_cast(
    constraint_data,
):
    with pytest.raises(TypeError, match="real-valued"):
        phx.optim.QuadraticProgram(
            jnp.eye(1),
            jnp.zeros(1),
            **constraint_data,
        )


def test_configuration_and_shape_guards_are_explicit():
    with pytest.raises(ValueError, match="quadratic"):
        phx.optim.QuadraticProgram(jnp.ones((2, 3)), jnp.ones(2))
    with pytest.raises(ValueError, match="provided together"):
        phx.optim.QuadraticProgram(
            jnp.eye(2), jnp.ones(2), equality_matrix=jnp.ones((1, 2))
        )
    problem = phx.optim.QuadraticProgram(jnp.eye(2), jnp.ones(2))
    invalid_method: Any = "qpax-explicit"
    with pytest.raises(ValueError, match="max_dense_dimension"):
        phx.optim.solve_quadratic_program(problem, max_dense_dimension=1)
    with pytest.raises(ValueError, match="explicit differentiation"):
        phx.optim.solve_quadratic_program_primal(problem, method=invalid_method)
