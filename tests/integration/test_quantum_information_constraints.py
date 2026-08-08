#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_infidelity_residual_runs_through_functional_solver():
    time = phx.domain.TimeInterval(0.0, 1.0)
    target = time.Function()(jnp.asarray([1.0, 1.0], dtype=complex) / jnp.sqrt(2.0))

    @time.Function("t")
    def exact_state(t):
        del t
        return target.func()

    @time.Function("t")
    def orthogonal_state(t):
        del t
        return jnp.asarray([1.0, -1.0], dtype=complex) / jnp.sqrt(2.0)

    condition = phx.conditions.Residual(
        "psi",
        time.component(),
        lambda state: 1.0 - phx.operators.state_fidelity(state, target),
        label="target infidelity",
    )
    constraint = phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(
            phx.integration.mean_over(condition.on),
            phx.integration.MonteCarloPlan(32),
        ),
    )
    exact = phx.solver.FunctionalSolver(functions={"psi": exact_state}, terms=[constraint], )
    orthogonal = phx.solver.FunctionalSolver(functions={"psi": orthogonal_state}, terms=[constraint], )

    loss = eqx.filter_jit(lambda solver, key: solver.loss(key=key))
    exact_loss = loss(exact, jr.key(0))
    orthogonal_loss = loss(orthogonal, jr.key(1))

    assert jnp.isrealobj(exact_loss)
    assert jnp.isrealobj(orthogonal_loss)
    assert exact_loss < 1e-20
    assert jnp.allclose(orthogonal_loss, 1.0, atol=1e-12)
