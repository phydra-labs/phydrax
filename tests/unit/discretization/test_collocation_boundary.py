#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_chebyshev_collocation_differentiates_polynomials_and_solves_poisson():
    collocation = phx.discretization.ChebyshevCollocation(17)
    nodes = collocation.nodes

    first = collocation.derivative(nodes**4, 1)
    second = collocation.derivative(nodes**4, 2)
    solution = collocation.solve_helmholtz_dirichlet(
        jnp.full((17,), 2.0),
        lower_value=0.0,
        upper_value=0.0,
    )

    assert jnp.allclose(first, 4.0 * nodes**3, atol=1e-9)
    assert jnp.allclose(second, 12.0 * nodes**2, atol=1e-8)
    assert jnp.allclose(solution, nodes**2 - 1.0, atol=1e-8)


def test_low_rank_boundary_correction_enforces_constraints_exactly():
    operator = jnp.asarray(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]
    )
    plan = phx.linalg.LowRankBoundaryCorrectionPlan(operator, [0, 2])

    solution = plan.solve(jnp.zeros((3,)), jnp.asarray([1.0, 3.0]))

    assert jnp.allclose(solution, jnp.asarray([1.0, 2.0, 3.0]), atol=1e-7)


def test_collocation_and_boundary_correction_refuse_unsafe_dense_budgets():
    with pytest.raises(ValueError, match="maximum_dimension"):
        phx.discretization.ChebyshevCollocation(33, maximum_dimension=16)

    with pytest.raises(ValueError, match="budget"):
        phx.linalg.LowRankBoundaryCorrectionPlan(
            jnp.eye(32),
            jnp.arange(16),
            maximum_construction_bytes=64,
        )
