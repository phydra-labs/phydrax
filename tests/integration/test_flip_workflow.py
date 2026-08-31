#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_fixed_population_flip_workflow_is_jittable_and_transactional():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(8) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(mac).prepare()
    projection = phx.solver.MACFreeSurfaceProjectionPlan(
        mac, boundaries=boundaries, tolerance=1.0e-7
    )
    position = jnp.asarray(
        [[0.25, 0.25], [0.40, 0.25], [0.25, 0.40], [0.40, 0.40]]
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), jnp.ones((4,)), ambient_dimension=2
    ).prepare()
    transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(particles)
    compiled = phx.equations.compile_flip_problem(
        phx.equations.FLIPProblemIR("workflow", 1.0, jnp.asarray([0.0, -0.1])),
        transfer,
        projection,
        phx.discretization.flip.FLIPMethodPlan(
            0.05, liquid_fraction_threshold=0.01
        ),
    )
    state = compiled.initialize_state(position, jnp.zeros_like(position))
    apply = jax.jit(lambda value, dt: compiled.step_detailed(value, dt))
    result = apply(state, jnp.asarray(1.0e-4))
    assert result.successful
    assert result.diagnostics.mass_balance_defect < 1.0e-12
    assert result.diagnostics.momentum_balance_defect < 1.0e-12
    assert result.diagnostics.divergence_norm < 1.0e-6
    assert result.accepted_state.particles.position.shape == position.shape
