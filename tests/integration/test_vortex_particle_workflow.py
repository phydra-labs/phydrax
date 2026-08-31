#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_vortex_particle_diffrax_rollout_preserves_pair_circulation_and_is_differentiable():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2),
        jnp.ones((2,)),
        ambient_dimension=2,
    ).prepare()
    properties = phx.discretization.VortexParticleProperties(
        jnp.full((2,), 0.1),
        jnp.ones((2,)),
    )
    method = phx.discretization.VortexParticleMethodPlan(
        phx.operators.GaussianDirectVortexPlan2D(maximum_sources=2)
    )
    compiled = phx.equations.compile_vortex_particle_flow(
        phx.equations.VortexParticleFlowProblem("integration-pair", 2),
        particles,
        properties,
        method,
    )
    position = jnp.asarray(((-0.5, 0.0), (0.5, 0.0)))
    circulation = jnp.ones((2,))
    problem = compiled.as_differential_problem(
        position,
        circulation,
        t0=0.0,
        t1=0.01,
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray((0.0, 0.01)),
        solver=phx.solver.SSPRK33(),
        dt0=1e-3,
        max_steps=32,
    )
    final = compiled.dynamics.state_layout.unpack(solution.states[-1])
    initial_radius = jnp.linalg.norm(position[1] - position[0])
    final_radius = jnp.linalg.norm(final.position[1] - final.position[0])
    gradient = jax.grad(
        lambda gamma: jnp.sum(
            compiled.dynamics(0.0, compiled.initialize_state(position, gamma), None) ** 2
        )
    )(circulation)

    np.testing.assert_allclose(final.strength, circulation, atol=1e-12)
    np.testing.assert_allclose(final_radius, initial_radius, rtol=2e-6, atol=2e-8)
    assert jnp.all(jnp.isfinite(gradient))
    assert bool(solution.backend_successful)
