#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_linear_program_scenario_1() -> None:
    problem = phx.optim.LinearProgram(
        jnp.asarray([-1.0, -0.5]),
        inequality_matrix=jnp.asarray([[1.0, 1.0]]),
        inequality_rhs=jnp.asarray([1.0]),
        bounds=phx.optim.Bounds(0.0, jnp.inf),
    )

    result = phx.optim.solve_linear_program(problem)
    assert problem.canonical.quadratic is None
    np.testing.assert_allclose(result.primal, jnp.asarray([1.0, 0.0]), atol=2e-5)
    assert result.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert result.successful
    assert result.kkt_residual_norm < 1e-7
    assert not result.certificate.primal_ray_valid
    problem = phx.optim.LinearProgram(
        jnp.asarray([-1.0, -0.5]),
        inequality_matrix=jnp.asarray([[1.0, 1.0]]),
        inequality_rhs=jnp.asarray([1.0]),
        bounds=phx.optim.Bounds(0.0, jnp.inf),
    )
    cold = phx.optim.solve_linear_program(problem)
    warm_start = phx.optim.ConvexWarmStart.from_result(cold)
    warm = phx.optim.solve_convex_program(problem, warm_start=warm_start).result

    np.testing.assert_allclose(warm.primal, cold.primal, atol=2e-6)
    assert warm.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert warm.iterations <= cold.iterations
    infeasible = phx.optim.LinearProgram(
        jnp.zeros(1),
        inequality_matrix=jnp.asarray([[1.0], [-1.0]]),
        inequality_rhs=jnp.asarray([0.0, -1.0]),
    )
    infeasible_result = phx.optim.solve_linear_program(infeasible)
    assert infeasible_result.status == phx.optim.ConvexProgramStatus.PRIMAL_INFEASIBLE
    assert infeasible_result.certificate.dual_ray_valid
    assert infeasible_result.certificate.dual_ray_residual_norm < 1e-7

    unbounded = phx.optim.LinearProgram(jnp.asarray([-1.0]))
    unbounded_result = phx.optim.solve_linear_program(unbounded)
    assert unbounded_result.status == phx.optim.ConvexProgramStatus.DUAL_INFEASIBLE
    assert unbounded_result.certificate.primal_ray_valid
    assert unbounded_result.certificate.primal_ray_objective < 0.0
    problem = phx.optim.LinearProgram(
        jnp.asarray([1.0, 1.0, -2.0]),
        bounds=phx.optim.Bounds(
            jnp.asarray([2.0, 0.0, -jnp.inf]),
            jnp.asarray([2.0, jnp.inf, 3.0]),
        ),
    )
    result = phx.optim.solve_linear_program(problem)

    np.testing.assert_allclose(result.primal, jnp.asarray([2.0, 0.0, 3.0]), atol=2e-5)
    assert result.lower_bound_dual[0] > 0.0
    assert result.lower_bound_dual[1] > 0.0
    assert result.upper_bound_dual[2] > 0.0
    assert result.status == phx.optim.ConvexProgramStatus.OPTIMAL


def test_bound_roles_must_be_shared_across_program_batch() -> None:
    with pytest.raises(ValueError, match="shared finite/fixed role pattern"):
        phx.optim.LinearProgram(
            jnp.asarray([[-1.0], [-1.0]]),
            bounds=phx.optim.Bounds(
                jnp.asarray([[0.0], [-jnp.inf]]),
                jnp.ones((2, 1)),
            ),
        ).as_quadratic_program()


def test_linear_program_is_jittable_for_static_bound_topology() -> None:
    bounds = phx.optim.Bounds(jnp.zeros(2), jnp.ones(2))

    @jax.jit
    def solve(linear: Any) -> Any:
        return phx.optim.solve_linear_program(
            phx.optim.LinearProgram(linear, bounds=bounds)
        ).primal

    np.testing.assert_allclose(solve(jnp.asarray([-1.0, 1.0])), [1.0, 0.0], atol=2e-5)
