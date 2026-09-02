#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
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
    source = phx.discretization.VortexSourceState(
        position,
        circulation,
        core_radius=core,
    )
    targets = phx.discretization.VortexTargetState(target)
    accelerated = phx.operators.FixedClusterVortexPlan2D(
        position,
        leaf_size=2,
        opening_angle=0.6,
        maximum_reference_displacement=0.2,
    ).evaluate(source, targets)
    direct = (
        phx.operators.GaussianDirectVortexPlan2D(
            maximum_sources=4,
            maximum_targets=2,
        )
        .prepare(
            source_capacity=4,
            target_capacity=2,
            target_topology="arbitrary-targets",
        )
        .evaluate(source, targets)
    )
    stale_source = phx.discretization.VortexSourceState(
        position + 0.1,
        circulation,
        core_radius=core,
    )
    stale = phx.operators.FixedClusterVortexPlan2D(
        position,
        maximum_reference_displacement=0.01,
    ).evaluate(stale_source, targets)

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


def test_random_vortex_solver_uses_named_antithetic_realizations():
    direct = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=2,
    ).prepare(source_capacity=2, target_capacity=2)
    source = phx.discretization.VortexSourceState(
        jnp.asarray(((-0.5, 0.0), (0.5, 0.0))),
        jnp.asarray((1.0, -1.0)),
        core_radius=jnp.full((2,), 0.1),
        volume=jnp.ones((2,)),
    )
    solver = phx.applications.vortex_flow.RandomVortexSolverPlan(
        direct,
        0.01,
        2,
        antithetic=True,
    )
    result = solver.step(
        solver.initialize(source),
        jax.random.key(1),
        0.01,
    )

    assert result.evidence.antithetic
    assert result.evidence.weak_moment_residual < 1e-12
    assert bool(result.successful)


def test_native_learned_vorticity_reconstruction_is_divergence_free():
    count = 8
    coordinates = jnp.arange(count) / count
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    vorticity = jnp.sin(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * yy)
    result = phx.applications.vortex_flow.PeriodicVorticityReconstructionPlan(
        (count, count),
        (1.0, 1.0),
    ).reconstruct(vorticity, velocity_gradient=True)

    assert result.velocity.shape == (count, count, 2)
    assert result.velocity_gradient.shape == (count, count, 2, 2)
    assert result.divergence_norm < 1e-10
    assert bool(result.successful)


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


def test_equilibrium_wall_closure_and_bounded_load_recovery_report_evidence():
    wall = phx.discretization.EquilibriumWallVortexClosurePlan(
        1.0,
        1.0e-3,
        jnp.asarray((0.1,)),
        jnp.asarray((1.0,)),
        y_plus_envelope=(0.0, 1.0e5),
    )
    wall_result = wall.evaluate(
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        jnp.asarray(((0.0, 1.0),)),
        jnp.asarray(0.1),
    )
    assert bool(wall_result.evidence.successful[0])
    assert jnp.allclose(wall_result.traction, 0.0)
    assert jnp.allclose(wall_result.vortex_strength_increment, 0.0)

    recovery = phx.discretization.VortexLoadRecoveryPlan(
        jnp.asarray((4.0, 2.0, 1.0)), 2.0
    )
    errors = 0.01 * jnp.asarray((16.0, 4.0, 1.0))
    loads = jnp.stack((1.0 + errors, 2.0 - errors), axis=-1)
    recovered = recovery.evaluate(
        loads,
        loads,
        topology_correspondence=jnp.asarray(True),
        circulation_defect=jnp.asarray(0.0),
        impulse_defect=jnp.asarray(0.0),
        time_stencil_defect=jnp.asarray(0.0),
        panel_residual=jnp.asarray(0.0),
    )
    assert bool(recovered.recoverable)
    assert jnp.allclose(recovered.estimate, jnp.asarray((1.0, 2.0)))
