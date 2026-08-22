#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


pytest.importorskip("mpax")


def test_mpax_rapdhg_solves_bounded_lp_and_qp_with_independent_audit():
    policy = phx.optim.ConvexSolvePolicy(
        phx.optim.MPAXraPDHG(iteration_limit=2_000),
        termination=phx.optim.ConvexTermination(
            absolute=1e-5,
            relative=1e-5,
            maximum_steps=2_000,
        ),
    )
    lp = phx.optim.LinearProgram(
        jnp.asarray([-1.0, -0.5]),
        inequality_matrix=jnp.asarray([[1.0, 1.0]]),
        inequality_rhs=jnp.asarray([1.0]),
        bounds=phx.optim.Bounds(0.0, jnp.inf),
    )
    lp_result = phx.optim.solve_linear_program(lp, policy=policy)
    assert lp_result.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert lp_result.kkt_residual_norm < 2e-5

    qp = phx.optim.QuadraticProgram(
        jnp.eye(2),
        jnp.asarray([-1.0, -2.0]),
        bounds=phx.optim.Bounds(0.0, jnp.inf),
    )
    qp_result = phx.optim.solve_quadratic_program(qp, policy=policy)
    np.testing.assert_allclose(qp_result.primal, [1.0, 2.0], atol=2e-4)
    assert qp_result.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert qp_result.provenance.backend == "mpax"


def test_mpax_rapdhg_preserves_batch_axes():
    problem = phx.optim.QuadraticProgram(
        jnp.broadcast_to(jnp.eye(1), (2, 1, 1)),
        jnp.asarray([[-1.0], [-2.0]]),
        bounds=phx.optim.Bounds(0.0, jnp.inf),
    )
    policy = phx.optim.ConvexSolvePolicy(
        phx.optim.MPAXraPDHG(iteration_limit=2_000),
        termination=phx.optim.ConvexTermination(
            absolute=1e-5,
            relative=1e-5,
            maximum_steps=2_000,
        ),
    )
    result = phx.optim.solve_quadratic_program(problem, policy=policy)

    assert result.primal.shape == (2, 1)
    np.testing.assert_allclose(result.primal[:, 0], [1.0, 2.0], atol=2e-4)
    assert jnp.all(result.status == int(phx.optim.ConvexProgramStatus.OPTIMAL))


def test_mpax_algorithmic_differentiation_requires_explicit_unrolling():
    problem = phx.optim.QuadraticProgram(
        jnp.eye(1),
        jnp.asarray([-1.0]),
        bounds=phx.optim.Bounds(0.0, jnp.inf),
    )
    implicit_request = phx.optim.ConvexDifferentiationPolicy("algorithmic")
    policy = phx.optim.ConvexSolvePolicy(phx.optim.MPAXraPDHG(unroll=False))
    with pytest.raises(ValueError, match="requires an unrolled method"):
        phx.optim.solve_quadratic_program_primal(
            problem,
            policy=policy,
            differentiation=implicit_request,
        )

    unrolled = phx.optim.ConvexSolvePolicy(
        phx.optim.MPAXraPDHG(unroll=True, iteration_limit=200),
        termination=phx.optim.ConvexTermination(
            absolute=1e-4,
            relative=1e-4,
            maximum_steps=200,
        ),
    )

    def solution(target):
        candidate = phx.optim.QuadraticProgram(
            jnp.eye(1),
            -jnp.asarray([target]),
            bounds=phx.optim.Bounds(0.0, jnp.inf),
        )
        return phx.optim.solve_quadratic_program_primal(
            candidate,
            policy=unrolled,
            differentiation=implicit_request,
        )[0]

    derivative = jax.grad(solution)(1.0)
    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(derivative, 1.0, atol=5e-2)
