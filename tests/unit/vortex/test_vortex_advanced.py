#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_reformulated_vpm_and_relaxation_report_their_invariants():
    strength = jnp.asarray(((1.0, 0.2, 0.0), (0.0, 1.0, 0.1)))
    stretching = jnp.asarray(((0.2, -0.1, 0.0), (0.1, 0.3, -0.1)))
    core = jnp.asarray((0.2, 0.3))
    rate = phx.discretization.ReformulatedVPMPlan3D().rate(
        strength,
        stretching,
        core,
    )
    vorticity = jnp.asarray(((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    relaxed = phx.discretization.PedrizzettiRelaxationPlan3D(
        0.5,
        preserve_magnitude=True,
    ).apply(strength, vorticity)

    assert jnp.abs(rate.conservation_residual) < 1e-14
    assert bool(rate.finite & relaxed.finite)
    assert jnp.max(relaxed.alignment_after - relaxed.alignment_before) >= 0.0
    assert relaxed.magnitude_residual < 1e-12


def test_barnes_hut_backend_matches_direct_for_small_cloud_and_detects_staleness():
    position = jnp.asarray(((-0.6, -0.2), (-0.2, 0.3), (0.3, -0.4), (0.7, 0.2)))
    circulation = jnp.asarray((0.5, -0.3, 0.8, -0.4))
    core = jnp.full((4,), 0.1)
    target = jnp.asarray(((2.0, 1.5), (-2.0, -1.5)))
    accelerated = phx.operators.BarnesHutVortexPlan2D(
        position,
        leaf_size=2,
        opening_angle=0.6,
        maximum_reference_displacement=0.2,
    ).evaluate(position, circulation, core, target)
    direct = (
        phx.operators.GaussianDirectVortexPlan2D(
            maximum_sources=4,
            maximum_targets=2,
        )
        .prepare(source_capacity=4, target_capacity=2)
        .evaluate(
            position,
            circulation,
            core,
            targets=target,
        )
    )
    stale = phx.operators.BarnesHutVortexPlan2D(
        position,
        maximum_reference_displacement=0.01,
    ).evaluate(position + 0.1, circulation, core, target)

    np.testing.assert_allclose(
        accelerated.velocity, direct.velocity, rtol=0.15, atol=2e-3
    )
    assert bool(accelerated.successful)
    assert not bool(stale.successful)
    assert bool(stale.diagnostics.backend_diagnostics.stale_topology)


def test_actuator_sources_and_passive_probes_are_distinct():
    filament = phx.applications.vortex_flow.actuator_line_sources(
        jnp.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0))),
        jnp.asarray((1.0, 0.5)),
        0.05,
    )
    probes = phx.applications.vortex_flow.PassiveVortexProbes(
        jnp.asarray(((1.0, 0.5, 0.0),))
    )
    velocity = phx.operators.PreparedFilamentVelocity3D(filament).evaluate(
        probes.position
    )

    assert filament.topology.segment_capacity == 2
    assert probes.position.shape == (1, 3)
    assert jnp.linalg.norm(velocity.velocity) > 0.0


def test_random_diffusion_builds_named_diagonal_wiener_term():
    layout = phx.discretization.VortexParticleStateLayout(3, 2)
    term = phx.applications.vortex_flow.RandomVortexDiffusion(0.01).wiener_term(layout)
    coefficient = term.coefficient_array(
        0.0,
        jnp.zeros((layout.state_size,)),
        None,
    )

    assert term.representation == "diagonal"
    assert coefficient.shape == (layout.state_size,)
    assert jnp.all(coefficient[: layout.position_size] > 0.0)
    np.testing.assert_allclose(coefficient[layout.position_size :], 0.0)


def test_learned_vorticity_workflow_requires_and_uses_real_callbacks():
    trainer = lambda samples, weights, args: jnp.sum(weights) / samples.shape[0]
    evaluator = lambda model, targets: model * jnp.ones((targets.shape[0],))
    reconstruction = lambda vorticity, targets, args: (
        jnp.stack((-targets[:, 1], targets[:, 0]), axis=-1) * vorticity[:, None]
    )
    workflow = phx.applications.vortex_flow.LearnedVorticityWorkflow(
        trainer,
        evaluator,
        reconstruction,
        workflow_id="tiny-vorticity-model",
    )
    result = workflow.fit_and_reconstruct(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0))),
        jnp.asarray((1.0, 2.0)),
        jnp.asarray(((0.25, 0.5),)),
    )

    assert result.vorticity.shape == (1,)
    assert result.velocity.shape == (1, 2)
    assert bool(result.finite)


def test_nonlinear_vortex_step_closure_solves_polar_circulation_root():
    span = jnp.linspace(-1.0, 1.0, 4)
    leading = jnp.stack(
        (jnp.zeros_like(span), span, jnp.zeros_like(span)),
        axis=-1,
    )
    surface = phx.discretization.LiftingSurfacePlan(
        leading,
        leading + jnp.asarray((1.0, 0.0, 0.0)),
    ).prepare()
    lattice = phx.solver.SteadyVortexLatticePlan(
        surface,
        jnp.asarray((1.0, 0.0, 0.0)),
        wake_length=20.0,
        core_radius=0.03,
    )
    polar_angle = jnp.deg2rad(jnp.asarray((-15.0, 0.0, 15.0)))
    polar = phx.solver.SampledAirfoilPolar(
        polar_angle,
        2.0 * jnp.pi * polar_angle,
        jnp.asarray((0.05, 0.01, 0.05)),
    )
    alpha = jnp.deg2rad(4.0)
    result = phx.solver.VortexStepPlan(lattice, polar).solve(
        jnp.asarray((jnp.cos(alpha), 0.0, jnp.sin(alpha)))
    )

    assert result.residual_norm < 1e-6
    assert jnp.all(jnp.isfinite(result.panel_force))
    assert bool(result.successful)
