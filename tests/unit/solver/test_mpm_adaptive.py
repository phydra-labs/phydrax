#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _case():
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(12, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), jnp.full((4,), 0.01), ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "adaptive-test",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    position = jnp.asarray([[0.28, 0.31], [0.42, 0.36], [0.34, 0.49], [0.48, 0.52]])
    velocity = jnp.broadcast_to(jnp.asarray((0.03, -0.01)), position.shape)
    initial = compiled.initialize_state(
        position, velocity, jnp.full((4,), 0.01), arguments
    )
    return compiled, arguments, initial


def test_adaptive_mpm_retries_transactionally_and_builds_realized_mesh():
    compiled, arguments, initial = _case()
    plan = phx.solver.AdaptiveMPMRolloutPlan(
        compiled.dynamics,
        phx.solver.MPMAdaptivePolicy(
            16,
            maximum_retries=8,
            minimum_step_size=1e-5,
            maximum_step_size=1.0,
        ),
        final_time=0.02,
        initial_step_size=1.0,
    )
    result = jax.jit(lambda state: plan.rollout(state, arguments))(initial)

    assert bool(result.completed)
    assert int(result.status) == int(phx.solver.MPMAdaptiveStatus.SUCCESS)
    assert int(result.journal.attempt_count) > int(result.journal.accepted_count)
    assert int(result.realized_mesh.count) == int(result.journal.accepted_count)
    assert result.final_state.time == 0.02
    first_rejected = jnp.argmax(~result.journal.accepted & result.journal.attempted)
    assert result.journal.requested_step_sizes[first_rejected] > 0.0
    assert result.journal.stable_step_limits[first_rejected] > 0.0

    replay = phx.solver.ScheduledMPMRolloutPlan.from_realized(
        compiled.dynamics,
        result.realized_mesh,
        replay=phx.solver.MPMReplayPolicy("block", block_size=2),
    ).rollout(initial, arguments)
    for adaptive, scheduled in zip(
        jax.tree.leaves(result.final_state.particles),
        jax.tree.leaves(replay.final_state.particles),
        strict=True,
    ):
        np.testing.assert_allclose(adaptive, scheduled, rtol=1e-11, atol=1e-11)


def test_adaptive_mpm_reports_step_capacity_without_partial_failure():
    compiled, arguments, initial = _case()
    plan = phx.solver.AdaptiveMPMRolloutPlan(
        compiled.dynamics,
        phx.solver.MPMAdaptivePolicy(1, maximum_retries=0),
        final_time=0.02,
        initial_step_size=0.001,
    )
    result = plan.rollout(initial, arguments)

    assert not bool(result.completed)
    assert int(result.status) == int(phx.solver.MPMAdaptiveStatus.STEP_CAPACITY_REACHED)
    assert int(result.realized_mesh.count) == 1
    assert int(result.final_state.accepted_step) == 1
