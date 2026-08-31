#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _case(*, geometry_ad="piecewise"):
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
        execution=phx.discretization.SplatExecutionPolicy(
            accumulation="deterministic", geometry_ad=geometry_ad
        ),
    ).prepare(particles)
    domain = phx.discretization.MPMParticleDomainPlan(
        jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        periodic=(True, True),
        support_margin=0.0,
    )
    problem = phx.equations.MaterialPointProblemIR(
        "rollout-solid",
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
    )
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        domain,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    position = jnp.asarray([[0.28, 0.31], [0.42, 0.36], [0.34, 0.49], [0.48, 0.52]])
    velocity = jnp.broadcast_to(jnp.asarray((0.03, -0.01)), position.shape)
    initial = compiled.initialize_state(
        position,
        velocity,
        jnp.full((4,), 0.01),
        arguments,
    )
    mesh = phx.discretization.TemporalMesh.uniform(0.0, 0.005, 5, role="internal")
    return compiled, arguments, initial, mesh


def test_mpm_retention_modes_have_identical_final_state():
    compiled, arguments, initial, mesh = _case()
    trajectory = phx.solver.ScheduledMPMRolloutPlan(
        compiled.dynamics, mesh, retention="trajectory"
    ).rollout(initial, arguments)
    checkpoints = phx.solver.ScheduledMPMRolloutPlan(
        compiled.dynamics, mesh, retention="checkpoints", checkpoint_stride=2
    ).rollout(initial, arguments)
    final = phx.solver.ScheduledMPMRolloutPlan(
        compiled.dynamics, mesh, retention="final"
    ).rollout(initial, arguments)

    assert trajectory.retained.times.shape == (5,)
    assert checkpoints.retained.times.shape == (2,)
    assert final.retained.times.shape == (1,)
    assert jnp.all(trajectory.accepted)
    for reference, candidate in (
        (trajectory.final_state.particles, checkpoints.final_state.particles),
        (trajectory.final_state.particles, final.final_state.particles),
    ):
        for left, right in zip(
            jax.tree.leaves(reference), jax.tree.leaves(candidate), strict=True
        ):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


def test_full_step_and_block_replay_match_primal_and_gradients():
    compiled, arguments, initial, mesh = _case()
    policies = (
        phx.solver.MPMReplayPolicy("full"),
        phx.solver.MPMReplayPolicy("step"),
        phx.solver.MPMReplayPolicy("block", block_size=3),
    )

    def objective(initial_velocity, policy):
        particles = phx.discretization.MPMParticleState(
            initial.particles.position,
            initial_velocity,
            initial.particles.deformation_gradient,
            initial.particles.affine_velocity,
            initial.particles.reference_volume,
            initial.particles.first_piola,
            initial.particles.reference_energy_density,
            initial.particles.maximum_wave_speed,
            initial.particles.material_state,
        )
        runtime = phx.discretization.MPMRuntimeState(
            particles, initial.time, initial.accepted_step, initial.last_status
        )
        result = phx.solver.ScheduledMPMRolloutPlan(
            compiled.dynamics,
            mesh,
            replay=policy,
        ).rollout(runtime, arguments)
        return jnp.sum(result.final_state.particles.position**2)

    values = [objective(initial.particles.velocity, policy) for policy in policies]
    gradients = [
        jax.grad(lambda velocity: objective(velocity, policy))(initial.particles.velocity)
        for policy in policies
    ]
    for value in values[1:]:
        np.testing.assert_allclose(value, values[0], rtol=1e-12, atol=1e-12)
    for gradient in gradients[1:]:
        np.testing.assert_allclose(gradient, gradients[0], rtol=1e-10, atol=1e-10)


def _directions(initial):
    particle_direction = jax.tree.map(jnp.zeros_like, initial.particles)
    particle_direction = phx.discretization.MPMParticleState(
        particle_direction.position,
        jnp.full_like(initial.particles.velocity, 0.02),
        particle_direction.deformation_gradient,
        particle_direction.affine_velocity,
        particle_direction.reference_volume,
        particle_direction.first_piola,
        particle_direction.reference_energy_density,
        particle_direction.maximum_wave_speed,
        particle_direction.material_state,
    )
    argument_direction = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters(0.1, 0.2)
    )
    return particle_direction, argument_direction


def test_piecewise_gradient_report_matches_jvp_vjp_and_finite_difference():
    compiled, arguments, initial, mesh = _case(geometry_ad="piecewise")
    plan = phx.solver.ScheduledMPMRolloutPlan(
        compiled.dynamics,
        mesh,
        replay=phx.solver.MPMReplayPolicy("step"),
    )
    particle_direction, argument_direction = _directions(initial)
    report = plan.gradient_report(
        lambda final, _: jnp.sum(final.particles.position**2),
        initial,
        arguments,
        particle_direction,
        argument_direction,
        epsilon=1e-5,
    )

    assert report.gradient_kind == "piecewise-discrete"
    assert bool(report.branch_matched)
    assert report.jvp_vjp_residual < 1e-9
    assert report.finite_difference_residual < 1e-6


def test_frozen_gradient_report_does_not_claim_ordinary_finite_difference():
    compiled, arguments, initial, mesh = _case(geometry_ad="frozen")
    plan = phx.solver.ScheduledMPMRolloutPlan(compiled.dynamics, mesh)
    particle_direction, argument_direction = _directions(initial)
    report = plan.gradient_report(
        lambda final, _: jnp.sum(final.particles.velocity**2),
        initial,
        arguments,
        particle_direction,
        argument_direction,
    )

    assert report.gradient_kind == "frozen-surrogate"
    assert not bool(report.branch_matched)
    assert jnp.isnan(report.finite_difference_derivative)
    assert jnp.isnan(report.finite_difference_residual)
    assert report.jvp_vjp_residual < 1e-9
