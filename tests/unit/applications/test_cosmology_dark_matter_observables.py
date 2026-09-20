from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology._dark_matter_observables import (
    ComponentForceWorkLedgerPlan,
    DarkMatterSpatialContract,
    DarkMatterSurfaceDensityProduct,
    find_dark_matter_halos,
    MixedComponentSpectrumPlan,
    project_dark_matter_lensing_plane,
    select_dark_matter_periodic_radial_shells,
    sidm_angular_moments,
    sidm_collision_observables,
    WaveDarkMatterObservablePlan,
    weighted_particle_statistics,
    weighted_sidm_collision_observables,
    weighted_sidm_packet_observables,
)
from phydrax.applications.cosmology._sidm import SIDMCollisionDiagnostics
from phydrax.applications.cosmology._sidm_weighted import (
    WeightedSIDMCollisionDiagnostics,
    WeightedSIDMPacketState,
)
from phydrax.applications.cosmology._simulation_products import (
    ParticleSimulationSnapshot,
    ParticleSnapshotEvidence,
)
from phydrax.artifacts import ScientificArtifactEnvelope


jax.config.update("jax_enable_x64", True)


def _wave_observable_plan(shape=(8, 8)):
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for count in shape),
        axis_names=("x", "y"),
        field_name="psi",
    ).prepare(tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in shape))
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    wave = phx.applications.cosmology.WaveDarkMatterPlan(
        2.0,
        jnp.asarray((0.5, 0.50001)),
        gravitational_constant=0.01,
        reduced_planck_constant=0.25,
    ).prepare(space, background)
    shells = phx.discretization.PeriodicFourierShellPlan(
        shape,
        (1.0, 1.0),
        jnp.asarray((0.0, 5.0, 8.0, 12.0, 20.0, 40.0)),
        source_id="wave-observables-test",
    )
    plan = WaveDarkMatterObservablePlan(
        wave,
        shells,
        jnp.linspace(0.0, np.sqrt(0.5), 7),
    )
    return space, wave, plan


def test_wave_known_density_current_and_spectra_are_exact():
    space, wave, plan = _wave_observable_plan()
    x, _ = jnp.meshgrid(*(axis.nodes for axis in space.axes), indexing="ij")
    psi = jnp.exp(2.0j * jnp.pi * x)
    state = wave.initialize(psi)

    product = plan.evaluate(state, source_product_id="plane-wave")

    np.testing.assert_allclose(product.density, 2.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        product.peculiar_mass_current[..., 0],
        jnp.pi,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(product.peculiar_mass_current[..., 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(product.density_spectrum.power, 0.0, atol=1e-12)
    np.testing.assert_allclose(product.current_spectrum.power, 0.0, atol=1e-12)
    assert bool(product.density_spectrum.successful)
    vector = product.density_spectrum.as_theory_vector()
    assert vector.values.shape == (len(product.density_spectrum.valid_shell_indices),)
    assert not bool(product.core_profile.center_unique)
    assert not bool(product.core_profile.successful)
    assert not bool(product.vortices.topology_present)


def test_wave_soliton_profile_and_vortex_winding_carry_nondifferentiable_evidence():
    space, wave, plan = _wave_observable_plan()
    x, y = jnp.meshgrid(*(axis.nodes for axis in space.axes), indexing="ij")
    dx = jnp.minimum(jnp.abs(x - 0.5), 1.0 - jnp.abs(x - 0.5))
    dy = jnp.minimum(jnp.abs(y - 0.5), 1.0 - jnp.abs(y - 0.5))
    soliton = jnp.exp(-(dx * dx + dy * dy) / (2.0 * 0.13**2)).astype(jnp.complex128)
    profile = plan.evaluate(
        wave.initialize(soliton), source_product_id="soliton"
    ).core_profile
    assert bool(profile.center_unique)
    assert bool(profile.core_identified)
    assert not profile.topology_differentiable
    assert float(profile.core_radius) > 0.0
    assert np.all(np.diff(np.asarray(profile.enclosed_mass)) >= -1e-12)

    vortex_pair = jnp.sin(2.0 * jnp.pi * (x + 0.03)) + 1.0j * jnp.sin(
        2.0 * jnp.pi * (y + 0.03)
    )
    vortices = plan.evaluate(
        wave.initialize(vortex_pair), source_product_id="vortex-pair"
    ).vortices
    assert not vortices.topology_differentiable
    assert bool(vortices.topology_present)
    assert int(vortices.total_absolute_winding) > 0
    np.testing.assert_allclose(
        vortices.circulation,
        2.0
        * jnp.pi
        * wave.reduced_planck_constant
        / (wave.boson_mass * 0.5)
        * vortices.winding_number,
        rtol=1e-12,
        atol=1e-12,
    )


def test_weighted_particle_and_scattering_statistics_have_exact_effective_samples():
    result = weighted_particle_statistics(
        jnp.asarray(((0.0, 2.0), (2.0, 4.0), (jnp.nan, jnp.nan))),
        jnp.asarray((1.0, 3.0, jnp.nan)),
        jnp.asarray((True, True, False)),
    )
    np.testing.assert_allclose(result.mean, jnp.asarray((1.5, 3.5)))
    np.testing.assert_allclose(
        result.covariance,
        jnp.asarray(((0.75, 0.75), (0.75, 0.75))),
    )
    np.testing.assert_allclose(result.effective_sample_size, 1.6)
    assert int(result.active_count) == 2
    assert bool(result.successful)

    angular = sidm_angular_moments(
        jnp.asarray((-1.0, 1.0, jnp.nan)),
        jnp.asarray((1.0, 1.0, jnp.nan)),
        jnp.asarray((True, True, False)),
        maximum_order=3,
    )
    np.testing.assert_allclose(
        angular.legendre_moments,
        jnp.asarray((1.0, 0.0, 1.0, 0.0)),
        atol=1e-12,
    )
    np.testing.assert_allclose(angular.effective_sample_size, 2.0)
    assert bool(angular.successful)


def test_zero_rate_sidm_diagnostics_report_explicit_collisionless_evidence():
    zero = jnp.zeros((2,))
    false = jnp.zeros((2,), dtype="bool")
    true = jnp.asarray(True)
    rare = SIDMCollisionDiagnostics(
        physical_time_step=jnp.asarray(1.0),
        smoothing_length_comoving=jnp.ones((2,)),
        density_comoving=jnp.ones((2,)),
        density_physical=jnp.ones((2,)),
        kernel_weight_comoving=zero,
        kernel_weight_physical=zero,
        relative_speed_physical=zero,
        kernel_total_cross_section=zero,
        kernel_total_cross_section_per_mass=zero,
        kernel_supported=true,
        sampled_cosine=zero,
        sampled_azimuth=zero,
        angular_sample_successful=jnp.ones((2,), dtype="bool"),
        pair_probability=zero,
        particle_aggregate_probability=zero,
        random_uniform=jnp.ones((2,)),
        proposed_pairs=false,
        selected_pairs=false,
        accepted_pairs=false,
        event_count=jnp.asarray(0, dtype=jnp.int32),
        mean_free_path_physical=jnp.full((2,), jnp.inf),
        support_radius_physical=jnp.ones((2,)),
        knudsen_number=jnp.full((2,), jnp.inf),
        pair_momentum_defect=jnp.zeros((2, 3)),
        pair_kinetic_energy_defect=zero,
        total_momentum_defect=jnp.zeros((3,)),
        total_kinetic_energy_defect=jnp.asarray(0.0),
        neighborhood_successful=true,
        smoothing_converged=true,
        smoothing_within_bounds=true,
        equal_active_mass=true,
        probability_valid=true,
        aggregate_probability_valid=true,
        capacity_valid=true,
        knudsen_valid=true,
        endpoint_disjoint=true,
        inactive_preserved=true,
        conservative=true,
        finite=true,
        successful=true,
    )
    rare_product = sidm_collision_observables(rare)
    assert bool(rare_product.collisionless)
    assert bool(jnp.isposinf(rare_product.minimum_knudsen_number))
    assert bool(rare_product.finite)
    assert bool(rare_product.successful)

    weighted = WeightedSIDMCollisionDiagnostics(
        physical_time_step=jnp.asarray(1.0),
        smoothing_length_comoving=jnp.ones((2,)),
        kernel_weight_physical=zero,
        relative_speed_physical=zero,
        total_cross_section=zero,
        pair_probability=zero,
        particle_aggregate_probability=zero,
        exchanged_weight=zero,
        proposed_pairs=false,
        selected_pairs=false,
        accepted_pairs=false,
        child_required=jnp.asarray(0, dtype=jnp.int32),
        child_slots_used=jnp.asarray(0, dtype=jnp.int32),
        event_count=jnp.asarray(0, dtype=jnp.int32),
        number_density_physical=jnp.ones((2,)),
        mean_free_path_physical=jnp.full((2,), jnp.inf),
        support_radius_physical=jnp.ones((2,)),
        knudsen_number=jnp.full((2,), jnp.inf),
        mass_defect=jnp.asarray(0.0),
        momentum_defect=jnp.zeros((3,)),
        kinetic_energy_defect=jnp.asarray(0.0),
        neighborhood_successful=true,
        kernel_supported=true,
        probability_valid=true,
        aggregate_probability_valid=true,
        capacity_valid=true,
        knudsen_valid=true,
        endpoint_disjoint=true,
        mass_relation_valid=true,
        angular_split_successful=true,
        angular_sampling_residual=jnp.asarray(0.0),
        angular_sampling_valid=true,
        lineage_valid=true,
        conservative=true,
        finite=true,
        successful=true,
    )
    weighted_product = weighted_sidm_collision_observables(weighted)
    assert bool(weighted_product.collisionless)
    assert bool(jnp.isposinf(weighted_product.minimum_mean_free_path_physical))
    np.testing.assert_allclose(weighted_product.effective_exchange_event_count, 0.0)
    assert bool(weighted_product.finite)
    assert bool(weighted_product.successful)


def test_weighted_sidm_packets_keep_multiplicity_and_gravitational_mass_evidence():
    positions = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    microscopic_mass = jnp.asarray((2.0, 2.0, 0.0))
    multiplicity = jnp.asarray((1.0, 3.0, 0.0))
    gravitational_mass = microscopic_mass * multiplicity
    active = jnp.asarray((True, True, False))
    velocity = jnp.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    scale_factor = jnp.asarray(0.5)
    state = WeightedSIDMPacketState(
        positions,
        microscopic_mass,
        multiplicity,
        gravitational_mass,
        gravitational_mass[:, None] * scale_factor * velocity,
        active,
        jnp.asarray((10, 11, 0)),
        jnp.asarray((-1, -1, 0)),
        jnp.asarray((0, 0, 0)),
        scale_factor,
    )

    product = weighted_sidm_packet_observables(state)

    np.testing.assert_allclose(
        product.multiplicity_statistics.mean, jnp.asarray((1.5, 0.0, 0.0))
    )
    np.testing.assert_allclose(product.represented_particle_count, 4.0)
    np.testing.assert_allclose(product.gravitational_mass, 8.0)
    np.testing.assert_allclose(product.multiplicity_effective_sample_size, 1.6)
    assert bool(product.mass_relation_valid)
    assert bool(product.stable_identity_valid)
    assert bool(product.successful)


def test_particle_density_bulk_velocity_and_dispersion_use_native_mass_transfer():
    positions = jnp.asarray(((0.25, 0.25, 0.25), (0.75, 0.75, 0.75)))
    masses = jnp.asarray((0.25, 0.75))
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((10, 11)), masses, ambient_dimension=3
    ).prepare()
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))))
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    from phydrax.applications.cosmology._dark_matter_observables import (
        ParticleDarkMatterObservablePlan,
    )
    from phydrax.applications.cosmology._particles import CosmologicalParticleState

    plan = ParticleDarkMatterObservablePlan(
        transfer,
        jnp.asarray((0.5, 0.5, 0.5)),
        jnp.asarray((0.0, 0.2, 0.4, 0.5)),
    )
    velocity = jnp.asarray((1.0, -2.0, 0.5))
    scale_factor = jnp.asarray(0.5)
    state = CosmologicalParticleState(
        positions,
        masses[:, None] * scale_factor * velocity,
        scale_factor,
    )

    product = plan.evaluate(state, source_product_id="two-particle-state")

    np.testing.assert_allclose(jnp.sum(product.mass_content), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        product.bulk_velocity,
        jnp.where(product.populated_cells[..., None], velocity, 0.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(product.velocity_dispersion_tensor, 0.0, atol=1e-12)
    np.testing.assert_allclose(product.global_statistics.mean, velocity, atol=1e-12)
    assert bool(product.finite)

    counter_streaming = jnp.asarray(((0.0, 1.0, 0.0), (0.0, -1.0 / 3.0, 0.0)))
    angular_product = plan.evaluate(
        CosmologicalParticleState(
            positions,
            masses[:, None] * scale_factor * counter_streaming,
            scale_factor,
        ),
        source_product_id="counter-streaming-state",
    )
    relative = positions - jnp.asarray((0.5, 0.5, 0.5))
    expected_specific_angular = jnp.sum(
        masses[:, None] * jnp.cross(scale_factor * relative, counter_streaming),
        axis=0,
    )
    np.testing.assert_allclose(
        angular_product.angular_moments.mean_specific_angular_momentum,
        expected_specific_angular,
        atol=1e-12,
    )


def test_mixed_component_cross_power_reconstructs_direct_total_power():
    shape = (8, 8)
    shells = phx.discretization.PeriodicFourierShellPlan(
        shape,
        (1.0, 1.0),
        jnp.asarray((0.0, 5.0, 8.0, 12.0, 20.0, 40.0)),
        source_id="mixed-spectrum-test",
    )
    x = jnp.arange(shape[0], dtype="float64")[:, None] / shape[0]
    mode = jnp.broadcast_to(jnp.sin(2.0 * jnp.pi * x), shape)
    plan = MixedComponentSpectrumPlan(
        shells,
        ("wave", "particles"),
        normalization="additive-field",
        closure_absolute_tolerance=1e-12,
        closure_relative_tolerance=1e-12,
    )

    product = plan.evaluate(
        jnp.stack((mode, 2.0 * mode)),
        jnp.asarray(0.5),
        source_product_ids=("wave-product", "particle-product"),
    )

    resolved_shell = int(jnp.argmax(product.auto_power[0]))
    resolved_power = float(product.auto_power[0, resolved_shell])
    assert resolved_power > 0.0
    roundoff = 64.0 * np.finfo(np.float64).eps * resolved_power
    np.testing.assert_allclose(
        product.auto_power[1, resolved_shell],
        4.0 * product.auto_power[0, resolved_shell],
        rtol=2e-13,
        atol=roundoff,
    )
    np.testing.assert_allclose(
        product.cross_power[0, resolved_shell],
        2.0 * product.auto_power[0, resolved_shell],
        rtol=2e-13,
        atol=roundoff,
    )
    np.testing.assert_allclose(
        product.direct_total_power[resolved_shell],
        9.0 * product.auto_power[0, resolved_shell],
        rtol=2e-13,
        atol=roundoff,
    )
    np.testing.assert_allclose(
        product.reconstructed_total_power,
        product.direct_total_power,
        rtol=1e-13,
        atol=1e-12,
    )
    assert float(product.closure_residual) <= 1e-12
    assert bool(product.successful)


def test_component_force_and_work_ledger_closes_against_independent_totals():
    plan = ComponentForceWorkLedgerPlan(
        ("wave", "particles"),
        force_time_level="interval-midpoint",
        displacement_time_level="accepted-end-minus-start",
        force_unit="code-mass*code-length/code-time^2",
        work_unit="code-mass*code-length^2/code-time^2",
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    force = jnp.asarray(
        (
            ((1.0, 0.0), (-0.25, 2.0)),
            ((0.0, 3.0), (2.0, -1.0)),
        )
    )
    displacement = jnp.asarray(
        (
            ((2.0, 1.0), (4.0, -1.0)),
            ((1.0, 2.0), (-1.0, 3.0)),
        )
    )
    system_force = jnp.sum(force, axis=1)
    system_work = jnp.sum(force * displacement, axis=(1, 2))

    ledger = plan.evaluate(
        jnp.asarray((0.5, 0.6)),
        jnp.asarray((0.6, 0.7)),
        force,
        displacement,
        system_force,
        system_work,
    )

    np.testing.assert_allclose(ledger.force_closure_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(ledger.work_closure_residual, 0.0, atol=1e-12)
    assert bool(ledger.successful)


def test_lensing_composition_binds_spatial_units_and_source_artifact():
    artifact = ScientificArtifactEnvelope(
        artifact_kind="dark-matter-surface-density",
        content_digest="surface-density-digest",
        producer="native-test",
        producer_version="1",
        build_id="build",
        license_id="internal",
        resource_id="surface-density-resource",
        status="complete",
    )
    spatial = DarkMatterSpatialContract(
        (1.0, 1.0),
        axis_names=("y", "x"),
        geometry_kind="flat-sky-cartesian",
        frame_id="flat-periodic-plane",
        physics_id="projected-density-law",
        scale_id="code-cosmology",
        coordinate_time_level="a=0.5",
        length_unit_id="code-length",
        length_coordinate_kind="physical",
    )
    source = DarkMatterSurfaceDensityProduct(
        jnp.ones((4, 4)),
        jnp.asarray(2.0),
        spatial,
        artifact,
        density_unit_id="code-mass/code-length^2",
        density_coordinate_kind="physical",
        source_product_id="surface-density-product",
    )
    result = project_dark_matter_lensing_plane(
        phx.applications.cosmology.LensingPlanePlan(0.25),
        source,
    )

    np.testing.assert_allclose(result.convergence, 0.5)
    np.testing.assert_allclose(result.first_shear, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.second_shear, 0.0, atol=1e-12)
    assert result.source.product_id == source.product_id
    assert result.source.spatial.contract_id == spatial.contract_id
    assert bool(result.successful)
    with pytest.raises(ValueError, match="raster shape"):
        project_dark_matter_lensing_plane(
            phx.applications.cosmology.LensingPlanePlan(0.2),
            source,
        )

    with pytest.raises(ValueError, match="coordinate kinds"):
        DarkMatterSurfaceDensityProduct(
            jnp.ones((4, 4)),
            jnp.asarray(2.0),
            spatial,
            artifact,
            density_unit_id="code-mass/code-length^2",
            density_coordinate_kind="comoving",
            source_product_id="mismatched-surface-density",
        )


def test_halo_and_periodic_shell_composition_bind_geometry_and_active_padding():
    ids = jnp.asarray((10, 11, 12))
    active = jnp.asarray((True, False, False))
    masses = jnp.asarray((1.0, 0.0, 0.0))
    particles = phx.discretization.ParticleSetPlan(
        ids,
        masses,
        ambient_dimension=3,
        active_mask=active,
    ).prepare()
    initial_positions = jnp.asarray(((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    initial_momenta = jnp.zeros((3, 3))
    kinematics = phx.applications.cosmology.CosmologicalKDKPlan(
        particles,
        (1.0, 1.0, 1.0),
    )
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)
    )
    grid = phx.discretization.TensorGridPlan(
        axes,
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))))
    system = phx.equations.EulerSystem(3)
    discretization = phx.discretization.FiniteVolumePlan(
        grid,
        component_names=system.component_names,
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "dark-matter-observable-test",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    gravity = phx.solver.ParticleMeshGravityPlan(
        phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
            phx.solver.prepare_balance_law_transport(runtime)
        ),
        phx.discretization.ParticleGridSplatPlan(grid).prepare(particles),
    )
    producer_plan = phx.applications.cosmology.CosmologicalParticleMeshPlan(
        kinematics,
        gravity,
        (0.5, 0.5001),
    )
    producer_result = producer_plan.rollout(
        phx.applications.cosmology.FLRWBackground(1.0, 1.0),
        kinematics.initialize(initial_positions, initial_momenta, 0.5),
    )
    producer_state = producer_result.state
    evidence = ParticleSnapshotEvidence(
        producer_result,
        particles,
        producer_plan=producer_plan,
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="particle-snapshot",
        content_digest="particle-snapshot-digest",
        producer="native-test",
        producer_version="1",
        build_id="build",
        license_id="internal",
        resource_id="particle-snapshot-resource",
        status="complete",
    )

    def snapshot(snapshot_masses):
        return ParticleSimulationSnapshot(
            ids,
            producer_state.positions,
            producer_state.canonical_momenta,
            active,
            snapshot_masses,
            jnp.ones((3,)),
            jnp.zeros((3,), dtype=jnp.int32),
            -jnp.ones((3, 2), dtype=jnp.int64),
            producer_state.scale_factor,
            evidence,
            artifact,
            producer_result,
            particles,
            support_id=particles.prepared_id,
            interaction_id=evidence.interaction_id,
            physics_id=evidence.producer_result_id,
            scale_id=evidence.producer_scale_id,
            coordinate_time_level="accepted-end-scale-factor",
            inactive_reference_positions=initial_positions,
            inactive_reference_momenta=initial_momenta,
            inactive_reference_masses=masses,
            inactive_reference_weights=jnp.ones((3,)),
        )

    source = snapshot(jnp.asarray((1.0, 0.0, 0.0)))
    spatial = DarkMatterSpatialContract(
        (1.0, 1.0, 1.0),
        axis_names=("x", "y", "z"),
        geometry_kind="flat-periodic-cartesian",
        frame_id="flat-periodic-box",
        physics_id=source.physics_id,
        scale_id=source.scale_id,
        coordinate_time_level=source.coordinate_time_level,
        length_unit_id="code-length",
        length_coordinate_kind="comoving",
    )
    halos = find_dark_matter_halos(
        phx.applications.cosmology.PeriodicFoFFinderPlan(
            (1.0, 1.0, 1.0),
            0.2,
            3,
        ),
        source,
        particles,
        spatial,
    )
    assert halos.snapshot.snapshot_id == source.snapshot_id
    assert halos.spatial.contract_id == spatial.contract_id
    assert bool(halos.successful)

    radial_shells = select_dark_matter_periodic_radial_shells(
        phx.applications.cosmology.LightConePlan(
            jnp.asarray((0.2, 0.4)),
            3,
        ),
        source,
        spatial,
        jnp.zeros((3,)),
    )
    assert int(radial_shells.selected_count) == 1
    np.testing.assert_array_equal(
        radial_shells.selected_active,
        jnp.asarray((True, False, False)),
    )
    assert int(radial_shells.selected_stable_ids[0]) == 10
    assert bool(radial_shells.successful)

    with pytest.raises(ValueError, match="typed result|support"):
        snapshot(jnp.asarray((-1.0, 0.0, 0.0)))
