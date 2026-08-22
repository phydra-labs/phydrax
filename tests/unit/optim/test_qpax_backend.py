#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _policy(method, *, tolerance=1e-7, regularization=0.0):
    return phx.optim.ConvexSolvePolicy(
        method,
        termination=phx.optim.ConvexTermination(absolute=tolerance),
        regularization=regularization,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_qpax_implicit_matches_phydrax_kkt_contract(dtype):
    problem = phx.optim.QuadraticProgram(
        jnp.array([[4.0, 1.0], [1.0, 2.0]], dtype=dtype),
        jnp.array([-1.0, -1.0], dtype=dtype),
        equality_matrix=jnp.array([[1.0, 1.0]], dtype=dtype),
        equality_rhs=jnp.array([1.0], dtype=dtype),
        inequality_matrix=jnp.array([[-1.0, 0.0], [0.0, -1.0]], dtype=dtype),
        inequality_rhs=jnp.zeros(2, dtype=dtype),
    )
    tolerance = 2e-5 if dtype == jnp.float32 else 1e-7
    expected = phx.optim.solve_quadratic_program(
        problem,
        policy=_policy(phx.optim.DensePrimalDualQP(), tolerance=tolerance),
    )
    actual = phx.optim.solve_quadratic_program(
        problem,
        policy=_policy(phx.optim.QPaxInteriorPoint(), tolerance=tolerance),
    )

    np.testing.assert_allclose(
        actual.primal,
        expected.primal,
        atol=5e-4 if dtype == jnp.float32 else 2e-6,
        rtol=5e-4 if dtype == jnp.float32 else 2e-6,
    )
    assert actual.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert actual.valid
    assert actual.backend_converged
    assert actual.backend == "qpax-0.1.4"
    assert actual.method == "qpax-implicit"
    assert actual.kkt_residual_norm <= tolerance


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_qpax_implicit_applies_requested_regularization(dtype):
    problem = phx.optim.QuadraticProgram(
        jnp.ones((1, 1), dtype=dtype),
        jnp.asarray([-1.0], dtype=dtype),
    )
    tolerance = 2e-5 if dtype == jnp.float32 else 1e-7
    dense_policy = _policy(
        phx.optim.DensePrimalDualQP(),
        tolerance=tolerance,
        regularization=2.0,
    )
    qpax_policy = _policy(
        phx.optim.QPaxInteriorPoint(),
        tolerance=tolerance,
        regularization=2.0,
    )
    expected = phx.optim.solve_quadratic_program(problem, policy=dense_policy)
    actual = phx.optim.solve_quadratic_program(problem, policy=qpax_policy)

    np.testing.assert_allclose(
        actual.primal,
        expected.primal,
        atol=5e-4 if dtype == jnp.float32 else 2e-6,
        rtol=5e-4 if dtype == jnp.float32 else 2e-6,
    )
    np.testing.assert_allclose(actual.primal, jnp.asarray([1.0 / 3.0]), atol=5e-4)
    assert actual.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert actual.solver_dual_residual_norm <= tolerance


def test_qpax_regularized_primal_has_implicit_sensitivity():
    policy = _policy(phx.optim.QPaxInteriorPoint(), regularization=2.0)
    differentiation = phx.optim.ConvexDifferentiationPolicy("backend-implicit")

    def solution(target):
        problem = phx.optim.QuadraticProgram(
            jnp.ones((1, 1)),
            -jnp.asarray([target]),
            inequality_matrix=-jnp.ones((1, 1)),
            inequality_rhs=jnp.zeros(1),
        )
        return phx.optim.solve_quadratic_program_primal(
            problem,
            policy=policy,
            differentiation=differentiation,
        )[0]

    np.testing.assert_allclose(solution(1.0), 1.0 / 3.0, atol=2e-6)
    np.testing.assert_allclose(jax.grad(solution)(1.0), 1.0 / 3.0, atol=2e-3)


def test_qpax_regularization_makes_zero_hessian_problem_finite():
    problem = phx.optim.QuadraticProgram(jnp.zeros((1, 1)), jnp.asarray([-1.0]))
    result = phx.optim.solve_quadratic_program(
        problem,
        policy=_policy(phx.optim.QPaxInteriorPoint(), regularization=2.0),
    )

    np.testing.assert_allclose(result.primal, jnp.asarray([0.5]), atol=2e-6)
    assert result.status == phx.optim.ConvexProgramStatus.OPTIMAL


def test_qpax_implicit_public_primal_api_is_differentiable_and_batched():
    quadratic = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    inequality_matrix = jnp.broadcast_to(-jnp.eye(2), (2, 2, 2))
    inequality_rhs = jnp.zeros((2, 2))
    policy = _policy(phx.optim.QPaxInteriorPoint(), tolerance=1e-6)

    def objective(linear):
        problem = phx.optim.QuadraticProgram(
            quadratic,
            linear,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
        )
        primal = phx.optim.solve_quadratic_program_primal(
            problem,
            policy=policy,
            differentiation=phx.optim.ConvexDifferentiationPolicy("backend-implicit"),
        )
        return jnp.sum(primal)

    linear = jnp.array([[-2.0, -3.0], [-1.0, -4.0]])
    np.testing.assert_allclose(
        jax.grad(objective)(linear),
        -jnp.ones_like(linear),
        atol=2e-3,
    )


def test_qpax_configuration_does_not_accept_native_step_fraction():
    with pytest.raises(TypeError):
        phx.optim.QPaxInteriorPoint(step_fraction=0.9)


def test_qpax_algorithmic_differentiation_is_rejected():
    problem = phx.optim.QuadraticProgram(jnp.eye(1), jnp.array([-1.0]))
    with pytest.raises(ValueError, match="do not expose algorithmic differentiation"):
        phx.optim.solve_quadratic_program_primal(
            problem,
            policy=_policy(phx.optim.QPaxInteriorPoint()),
            differentiation=phx.optim.ConvexDifferentiationPolicy("algorithmic"),
        )
