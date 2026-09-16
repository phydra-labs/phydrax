#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _plan():
    return phx.solver.SHAKERATTLEPlan(
        jnp.ones((2,)),
        lambda configuration, _: configuration,
        lambda configuration, _: jnp.asarray(
            [jnp.vdot(configuration, configuration) - 1.0]
        ),
        maximum_projection_steps=8,
        constraint_tolerance=1.0e-10,
    )


def test_shake_rattle_preserves_position_and_velocity_constraints() -> None:
    state = phx.solver.ConstrainedMechanicalState(
        jnp.asarray((1.0, 0.0)), jnp.asarray((0.0, 1.0))
    )
    plan = _plan()

    result = jax.jit(lambda value: plan.step(value, 0.05))(state)

    assert bool(result.accepted)
    assert result.position_residual < 1.0e-10
    assert result.velocity_residual < 1.0e-10
    assert jnp.allclose(jnp.vdot(result.state.configuration, result.state.configuration), 1.0)
    assert jnp.allclose(
        jnp.vdot(result.state.configuration, result.state.momentum), 0.0, atol=1.0e-10
    )


def test_shake_rattle_rejects_nonpositive_step_without_committing() -> None:
    state = phx.solver.ConstrainedMechanicalState(
        jnp.asarray((1.0, 0.0)), jnp.asarray((0.0, 1.0))
    )

    result = _plan().step(state, 0.0)

    assert not bool(result.accepted)
    assert jnp.array_equal(result.state.configuration, state.configuration)
    assert jnp.array_equal(result.state.momentum, state.momentum)
