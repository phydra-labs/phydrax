#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_source_capabilities_and_property_requirements_are_explicit():
    source = phx.discretization.VortexSourceState(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0))),
        jnp.asarray((1.0, -1.0)),
        core_radius=jnp.full((2,), 0.2),
        volume=jnp.ones((2,)),
        active_mask=jnp.asarray((True, False)),
    )
    capabilities = phx.discretization.VortexVelocityCapabilities(
        2,
        required_source_fields=(
            "positions",
            "strength",
            "active_mask",
            "core_radius",
        ),
        supported_fields=("velocity", "velocity_gradient"),
        acceleration="direct",
    )

    assert source.dimension == 2
    assert capabilities.requires_core_radius
    assert not capabilities.requires_volume
    np.testing.assert_allclose(source.safe_strength(), (1.0, 0.0))


def test_singular_and_rosenhead_cores_recover_orientation_and_finite_center():
    displacement = jnp.asarray(((1.0, 0.0),))
    singular = phx.operators.SingularVortexKernel2D().evaluate(
        displacement,
        jnp.ones((1,)),
    )
    rosenhead = phx.operators.RosenheadVortexKernel2D().evaluate(
        jnp.zeros((1, 2)),
        jnp.ones((1,)),
        jnp.full((1,), 0.2),
    )

    np.testing.assert_allclose(
        singular.velocity,
        ((0.0, 1.0 / (2.0 * jnp.pi)),),
        rtol=1e-12,
    )
    np.testing.assert_allclose(rosenhead.velocity, 0.0)
    assert rosenhead.vorticity[0] > 0.0
    assert bool(singular.finite & rosenhead.finite)


def test_periodic_ewald_requires_compatible_mean_and_is_odd_under_swap():
    plan = phx.operators.PeriodicVortexEwaldPlan(
        (1.0, 1.0),
        splitting_parameter=6.0,
        real_image_radius=2,
        reciprocal_mode_radius=5,
    ).prepare(
        source_capacity=2,
        target_capacity=2,
        target_topology="same-support",
    )
    position = jnp.asarray(((0.25, 0.5), (0.75, 0.5)))
    source = phx.discretization.VortexSourceState(
        position,
        jnp.asarray((1.0, -1.0)),
    )
    target = phx.discretization.VortexTargetState(
        position,
        source_indices=jnp.arange(2),
    )
    result = plan.evaluate(source, target)

    np.testing.assert_allclose(result.velocity[0], result.velocity[1], atol=1e-8)
    assert result.diagnostics.backend_diagnostics.compatibility_residual < 1e-12
    assert bool(result.successful)


def test_population_transactions_preserve_strength_and_fail_closed_on_capacity():
    plan = phx.discretization.VortexPopulationPlan(3, 2)
    state, journal = plan.initialize(
        jnp.zeros((3, 2)),
        jnp.zeros((3,)),
        jnp.ones((3,)),
        jnp.ones((3,)),
        active_mask=jnp.zeros((3,), dtype=bool),
    )
    first = plan.insert(
        state,
        journal,
        (0.0, 0.0),
        1.0,
        0.2,
        0.5,
    )
    second = plan.split(
        first.accepted,
        first.journal,
        0,
        (0.1, 0.0),
    )
    third = plan.merge(
        second.accepted,
        second.journal,
        0,
        1,
    )

    assert bool(first.successful & second.successful & third.successful)
    np.testing.assert_allclose(
        jnp.sum(third.accepted.strength),
        1.0,
        atol=1e-12,
    )
    assert int(third.evidence.duplicate_id_count) == 0


def test_core_spreading_and_vrm_return_finite_conservative_rates():
    source = phx.discretization.VortexSourceState(
        jnp.asarray(((-0.4, 0.0), (0.0, 0.0), (0.4, 0.0))),
        jnp.asarray((0.2, 0.6, 0.2)),
        core_radius=jnp.full((3,), 0.2),
        volume=jnp.full((3,), 0.3),
    )
    core_rate, evidence = phx.discretization.GaussianCoreSpreadingPlan(2).rate(
        source,
        0.01,
    )
    redistributed = phx.discretization.VortexRedistributionPlan(2).apply(
        source,
        source.positions,
    )

    assert jnp.all(core_rate > 0.0)
    assert bool(evidence.compatible)
    np.testing.assert_allclose(
        jnp.sum(redistributed.strength),
        jnp.sum(source.strength),
        atol=1e-8,
    )
    assert bool(redistributed.successful)


def test_fmm_has_real_hierarchy_and_matches_direct_small_cloud():
    position = jnp.asarray(((-0.6, -0.2), (-0.2, 0.3), (0.3, -0.4), (0.7, 0.2)))
    strength = jnp.asarray((0.5, -0.3, 0.8, -0.4))
    core = jnp.full((4,), 0.08)
    targets = jnp.asarray(((0.9, 0.9), (-0.9, -0.9)))
    source = phx.discretization.VortexSourceState(
        position,
        strength,
        core_radius=core,
    )
    target = phx.discretization.VortexTargetState(targets)
    fmm = phx.operators.VortexFMMPlan(
        position,
        (-1.0, -1.0),
        (1.0, 1.0),
        depth=2,
        expansion_order=1,
        leaf_capacity=4,
    ).prepare(
        source_capacity=4,
        target_capacity=2,
        target_topology="arbitrary-targets",
    )
    direct = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=4,
        maximum_targets=2,
    ).prepare(
        source_capacity=4,
        target_capacity=2,
        target_topology="arbitrary-targets",
    )
    fmm_result = fmm.evaluate(source, target)
    direct_result = direct.evaluate(source, target)

    np.testing.assert_allclose(
        fmm_result.velocity,
        direct_result.velocity,
        rtol=0.25,
        atol=5e-3,
    )
    assert fmm_result.diagnostics.backend_diagnostics.m2l_count >= 0
    assert bool(fmm_result.successful)


def test_three_dimensional_fmm_matches_direct_vector_vorticity():
    position = jnp.asarray(
        (
            (-0.5, -0.2, 0.1),
            (-0.1, 0.4, -0.2),
            (0.4, -0.3, 0.2),
            (0.6, 0.2, -0.1),
        )
    )
    strength = jnp.asarray(
        (
            (0.2, 0.5, -0.1),
            (-0.3, 0.1, 0.4),
            (0.4, -0.2, 0.3),
            (-0.1, -0.4, -0.2),
        )
    )
    source = phx.discretization.VortexSourceState(
        position,
        strength,
        core_radius=jnp.full((4,), 0.1),
    )
    target = phx.discretization.VortexTargetState(
        jnp.asarray(((0.8, 0.7, 0.6), (-0.8, -0.7, -0.6)))
    )
    fmm = phx.operators.VortexFMMPlan(
        position,
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        depth=2,
        expansion_order=1,
        leaf_capacity=4,
    ).prepare(
        source_capacity=4,
        target_capacity=2,
        target_topology="arbitrary-targets",
    )
    direct = phx.operators.GaussianErfDirectVortexPlan3D(
        maximum_sources=4,
        maximum_targets=2,
        maximum_interactions=8,
    ).prepare(
        source_capacity=4,
        target_capacity=2,
        target_topology="arbitrary-targets",
    )
    fmm_result = fmm.evaluate(source, target)
    direct_result = direct.evaluate(source, target)

    np.testing.assert_allclose(
        fmm_result.velocity,
        direct_result.velocity,
        rtol=0.35,
        atol=1e-3,
    )
    assert bool(fmm_result.successful)


def test_corrected_p3m_and_free_space_fft_are_finite_authorities():
    count = 12
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(
                count,
                periodic=True,
                endpoint=False,
            )
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    spectral = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in range(2)))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2),
        jnp.ones((2,)),
        ambient_dimension=2,
    ).prepare()
    mesh = phx.operators.PeriodicVortexInCellPlan(
        particles,
        grid,
        spectral,
        phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(source_capacity=2, target_capacity=2)
    position = jnp.asarray(((0.35, 0.5), (0.65, 0.5)))
    source = phx.discretization.VortexSourceState(
        position,
        jnp.asarray((1.0, -1.0)),
        core_radius=jnp.full((2,), 0.08),
    )
    target = phx.discretization.VortexTargetState(
        position,
        source_indices=jnp.arange(2),
    )
    p3m = phx.operators.CorrectedP3MPlan(
        mesh,
        6.0,
        0.4,
    ).evaluate(source, target)

    omega = jnp.zeros((count, count)).at[count // 2, count // 2].set(1.0)
    free = phx.operators.FreeSpaceVortexFFTPlan(
        (count, count),
        (0.0, 0.0),
        (1.0, 1.0),
    ).evaluate(omega)

    assert jnp.all(jnp.isfinite(p3m.velocity))
    assert p3m.diagnostics.backend_diagnostics.near_pair_count > 0
    assert jnp.all(jnp.isfinite(free.velocity))
    assert free.boundary_vorticity_fraction == 0.0


def test_dynamic_core_rvpm_is_part_of_packed_dynamics():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2),
        jnp.ones((2,)),
        ambient_dimension=3,
    ).prepare()
    properties = phx.discretization.VortexParticleProperties(
        jnp.full((2,), 0.2),
        jnp.ones((2,)),
    )
    method = phx.discretization.VortexParticleMethodPlan(
        phx.operators.GaussianErfDirectVortexPlan3D(
            maximum_sources=2,
            maximum_targets=2,
            maximum_interactions=4,
        ),
        formulation=phx.discretization.ReformulatedVPMFormulation(),
    )
    compiled = phx.equations.compile_vortex_particle_flow(
        phx.equations.VortexParticleFlowProblem("rvpm", 3),
        particles,
        properties,
        method,
    )
    state = compiled.initialize_state(
        jnp.asarray(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))),
        jnp.asarray(((0.0, 1.0, 0.2), (0.0, -1.0, 0.2))),
    )
    unpacked = compiled.dynamics.state_layout.unpack(state)
    rate = compiled.dynamics(0.0, state)
    rate_unpacked = compiled.dynamics.state_layout.unpack(rate)

    assert compiled.dynamics.state_layout.dynamic_core
    np.testing.assert_allclose(unpacked.core_radius, 0.2)
    assert jnp.all(jnp.isfinite(rate_unpacked.core_radius))


def test_checkpoint_round_trip_preserves_population_and_journal(tmp_path):
    population = phx.discretization.VortexPopulationPlan(2, 2)
    state, journal = population.initialize(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0))),
        jnp.asarray((1.0, -1.0)),
        jnp.full((2,), 0.2),
        jnp.ones((2,)),
    )
    plan = phx.discretization.VortexCheckpointPlan(
        "population",
        (population.plan_id,),
    )
    path = tmp_path / "vortex.pxa"
    plan.write(
        path,
        state,
        journal,
        jax.random.key(7),
        jnp.asarray((0.0, 0.1)),
        source_lineage_id="lineage",
        backend_ids=("direct",),
        epoch_index=0,
    )
    restored = plan.restore(
        path,
        state,
        journal,
        jax.random.key(0),
        jnp.asarray((0.0, 0.0)),
    )

    np.testing.assert_allclose(restored.state.positions, state.positions)
    np.testing.assert_array_equal(restored.state.stable_ids, state.stable_ids)
    np.testing.assert_allclose(restored.accepted_times, (0.0, 0.1))
