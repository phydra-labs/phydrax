#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.lattice_field._continuum_study import (
    ContinuumStudyPlan,
    ContinuumSystematicVariation,
    CorrelatedContinuumData,
    RenormalizationCondition,
    run_continuum_study,
    ScaleSettingCondition,
)
from phydrax.applications.lattice_field._qcd_ensembles import (
    EnsembleManifest,
    EnsembleSegment,
    measurement_work_items,
    MeasurementSchedule,
    merge_ensemble_segments,
)
from phydrax.applications.lattice_field._qcd_observables import (
    baryon_correlator,
    HypercubicGaugeObservablePlan,
    measure_hypercubic_gauge_observables,
    meson_correlator,
    point_source_vectors,
    prepare_propagator_solve,
    PropagatorSolvePlan,
    run_wilson_flow,
    solve_propagators,
    stochastic_source_plan_for_measurement,
    WilsonFlowPlan,
)
from phydrax.applications.lattice_field._qcd_recipes import (
    build_su3_gauge_geometry,
    prepare_quenched_su3_recipe,
    prepare_staggered_hisq_style_rhmc_recipe,
    prepare_wilson_clover_nf2_recipe,
    QuenchedSU3Recipe,
    StaggeredHisqStyleRHMCRecipe,
    WilsonCloverNf2Recipe,
)
from phydrax.discretization._cell_complex import polygonal_cell_complex
from phydrax.discretization._lattice_boundary import LatticeBoundaryPhasePlan
from phydrax.discretization._oriented_path import prepare_cell_boundary_paths
from phydrax.discretization._topology import TensorTopology
from phydrax.graph._matrix_gauge import MatrixGaugeLinkSpace
from phydrax.metrix._complex_matrix_manifold import SpecialUnitaryGroup


def _su3_square_geometry():
    shape = (2, 2)
    boundary = LatticeBoundaryPhasePlan(
        TensorTopology(("x", "t"), shape, periodic=(True, True)),
        jnp.ones((2,), dtype=jnp.complex128),
        maximum_displacement=3,
    )
    topology = polygonal_cell_complex(
        None,
        jnp.asarray([[0, 1, 3, 2]], dtype=jnp.int32),
        4,
    )
    link_space = MatrixGaugeLinkSpace(topology, SpecialUnitaryGroup(3))
    plaquettes = prepare_cell_boundary_paths(topology)
    coordinates = np.stack(
        np.unravel_index(np.arange(4), boundary.topology.axis_sizes), axis=-1
    )
    tails = np.asarray(link_space.tail_vertices)
    heads = np.asarray(link_space.head_vertices)
    forward_sites = np.empty((4, 2), dtype=np.int32)
    forward_edges = np.empty((4, 2), dtype=np.int32)
    forward_orientations = np.empty((4, 2), dtype=np.int32)
    for site in range(4):
        for axis, size in enumerate(shape):
            neighbor_coordinate = coordinates[site].copy()
            neighbor_coordinate[axis] = (neighbor_coordinate[axis] + 1) % size
            neighbor = int(np.ravel_multi_index(neighbor_coordinate, shape))
            forward_sites[site, axis] = neighbor
            direct = np.flatnonzero((tails == site) & (heads == neighbor))
            reverse = np.flatnonzero((tails == neighbor) & (heads == site))
            if direct.size:
                forward_edges[site, axis] = int(direct[0])
                forward_orientations[site, axis] = 1
            else:
                forward_edges[site, axis] = int(reverse[0])
                forward_orientations[site, axis] = -1
    return build_su3_gauge_geometry(
        link_space,
        plaquettes,
        boundary,
        forward_sites,
        forward_edges,
        forward_orientations,
    )


def test_gauge_normalization_topology_and_flow_identities():
    plan = HypercubicGaugeObservablePlan(
        (2, 2, 2, 2),
        lattice_spacing=0.125,
        color_components=3,
    )
    identity = jnp.eye(3, dtype=jnp.complex128)
    links = jnp.broadcast_to(identity, plan.configuration_shape)
    measured = measure_hypercubic_gauge_observables(plan, links)

    np.testing.assert_allclose(measured.mean_plaquette, 1.0, atol=1.0e-13)
    np.testing.assert_allclose(measured.wilson_action_density, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(measured.topological_charge, 0.0, atol=1.0e-13)
    assert measured.topology_status == "measured"
    assert measured.topology_id == plan.topology_id

    geometry = _su3_square_geometry()
    action_links = geometry.link_space.identity()
    quenched = QuenchedSU3Recipe(
        geometry,
        MeasurementSchedule(
            8,
            thermalization_trajectories=2,
            measurement_interval=2,
        ),
        beta=5.7,
        step_size=0.01,
        leapfrog_steps=2,
    )
    prepared = prepare_quenched_su3_recipe(quenched, action_links)
    flow_plan = WilsonFlowPlan(prepared.gauge_action, step_size=0.01, num_steps=2)
    flowed = run_wilson_flow(flow_plan, action_links)
    assert flowed.flow_id == flow_plan.flow_id
    assert flowed.topology_id == geometry.link_space.topology.topology_id
    assert bool(jnp.all(flowed.valid))
    np.testing.assert_allclose(flowed.mean_plaquette, 1.0, atol=1.0e-12)


def test_meson_and_baryon_contractions_obey_declared_spatial_normalization():
    spin, color = 2, 3
    identity = jnp.eye(spin * color, dtype=jnp.complex128).reshape(
        (spin, color, spin, color)
    )
    amplitudes = jnp.asarray([1.0, 2.0])
    propagator = jnp.broadcast_to(identity, (3, 2) + identity.shape)
    propagator = propagator * amplitudes[None, :, None, None, None, None]
    gamma = jnp.eye(spin, dtype=jnp.complex128)

    meson_sum = meson_correlator(
        propagator,
        gamma,
        gamma,
        time_axis=1,
        spatial_normalization="sum",
    )
    meson_mean = meson_correlator(
        propagator,
        gamma,
        gamma,
        time_axis=1,
        spatial_normalization="mean",
    )
    np.testing.assert_allclose(meson_sum, 3 * spin * color * amplitudes**2)
    np.testing.assert_allclose(meson_mean, spin * color * amplitudes**2)

    spin_one = jnp.eye(3, dtype=jnp.complex128).reshape((1, 3, 1, 3))
    baryon_propagator = jnp.broadcast_to(spin_one, (3, 2) + spin_one.shape)
    spin_tensor = jnp.ones((1, 1, 1), dtype=jnp.complex128)
    baryon = baryon_correlator(
        baryon_propagator,
        baryon_propagator,
        baryon_propagator,
        spin_tensor,
        spin_tensor,
        time_axis=1,
        spatial_normalization="sum",
        normalize_color=True,
    )
    np.testing.assert_allclose(baryon, jnp.full((2,), 3.0))


def test_measurement_streams_and_segment_merge_exclude_thermalization():
    schedule = MeasurementSchedule(
        12,
        thermalization_trajectories=4,
        measurement_interval=2,
        sources_per_configuration=2,
    )
    manifest = EnsembleManifest(
        schedule,
        ensemble_id="ensemble",
        recipe_id="recipe",
        topology_id="topology",
        field_space_id="links",
        chain_id="chain-0",
        update_randomness_id="updates-root",
        measurement_randomness_id="measurements-root",
    )
    work = measurement_work_items(manifest)
    assert schedule.trajectory_indices == (4, 6, 8, 10)
    assert all(item.trajectory_index >= 4 for item in work)
    assert all(item.update_randomness_id != item.source_randomness_id for item in work)
    assert len({item.source_randomness_id for item in work}) == len(work)
    source_plan = stochastic_source_plan_for_measurement(
        work[0], (4, 2, 3), noise_kind="z4"
    )
    assert source_plan.configuration_id == work[0].configuration_id
    assert source_plan.randomness_id == work[0].source_randomness_id
    assert source_plan.randomness_id != work[0].update_randomness_id

    first = EnsembleSegment(
        manifest.manifest_id,
        0,
        7,
        jnp.asarray((4, 6)),
        ("configuration-4", "configuration-6"),
        initial_checkpoint_id="checkpoint-0",
        terminal_checkpoint_id="checkpoint-7",
    )
    second = EnsembleSegment(
        manifest.manifest_id,
        7,
        12,
        jnp.asarray((8, 10)),
        ("configuration-8", "configuration-10"),
        initial_checkpoint_id="checkpoint-7",
        terminal_checkpoint_id="checkpoint-12",
    )
    merged = merge_ensemble_segments(manifest, (second, first))
    np.testing.assert_array_equal(merged.trajectory_indices, (4, 6, 8, 10))
    assert merged.evidence.non_overlapping
    assert merged.evidence.thermalization_excluded
    assert merged.evidence.complete_measurement_coverage

    overlap = EnsembleSegment(
        manifest.manifest_id,
        6,
        9,
        jnp.asarray((6, 8)),
        ("other-6", "other-8"),
        initial_checkpoint_id="checkpoint-6",
        terminal_checkpoint_id="checkpoint-9",
    )
    with pytest.raises(ValueError, match="must not overlap"):
        merge_ensemble_segments(
            manifest,
            (first, overlap),
            require_complete=False,
            require_checkpoint_chain=False,
        )


def _continuum_data(spacings, site_counts):
    spacings = np.asarray(spacings, dtype=float)
    sites = np.asarray(site_counts, dtype=int)
    extents = spacings * sites
    values = 1.25 + 0.7 * spacings**2 + 0.4 / extents
    count = spacings.size
    observable_covariance = 2.5e-5 * (0.8 * np.eye(count) + 0.2 * np.ones((count, count)))
    covariance = np.zeros((3 * count, 3 * count))
    covariance[:count, :count] = observable_covariance
    covariance[count : 2 * count, count : 2 * count] = 1.0e-4 * np.eye(count)
    covariance[2 * count :, 2 * count :] = 1.0e-6 * np.eye(count)
    return CorrelatedContinuumData(
        values,
        1.0 / spacings,
        np.ones(count),
        sites,
        covariance,
        datum_ids=tuple(f"datum-{index}" for index in range(count)),
        covariance_id="joint-jackknife-covariance",
    )


def _continuum_plan():
    variations = (
        ContinuumSystematicVariation(
            cutoff_power=2.0,
            finite_volume_power=1.0,
            variation_id="a2-plus-inverse-L",
        ),
        ContinuumSystematicVariation(
            cutoff_power=1.0,
            finite_volume_power=1.0,
            variation_id="a-plus-inverse-L",
        ),
    )
    return ContinuumStudyPlan(
        ScaleSettingCondition(
            1.0,
            scheme="reference-gradient-flow-scale",
            physical_reference_uncertainty=0.001,
        ),
        RenormalizationCondition(scheme="nonperturbative", scale="2-GeV"),
        variations,
        minimum_distinct_spacings=3,
        minimum_distinct_volumes=2,
    )


def test_correlated_continuum_fit_and_resolution_volume_abstention():
    data = _continuum_data(
        (0.12, 0.12, 0.09, 0.09, 0.06, 0.06),
        (24, 32, 32, 40, 48, 64),
    )
    result = run_continuum_study(_continuum_plan(), data)
    assert result.status == "complete"
    assert len(result.fits) == 2
    np.testing.assert_allclose(result.fits[0].continuum_value, 1.25, atol=2.0e-3)
    assert float(result.total_standard_error) >= float(result.statistical_standard_error)

    resolution_failure = run_continuum_study(
        _continuum_plan(),
        _continuum_data(
            (0.12, 0.12, 0.08, 0.08),
            (24, 32, 36, 48),
        ),
    )
    assert resolution_failure.status == "abstained"
    assert "insufficient-distinct-lattice-spacings" in (
        resolution_failure.abstention_reasons
    )

    volume_failure = run_continuum_study(
        _continuum_plan(),
        _continuum_data(
            (0.12, 0.10, 0.08, 0.06),
            (32, 32, 32, 32),
        ),
    )
    assert volume_failure.status == "abstained"
    assert "insufficient-distinct-spatial-volumes" in volume_failure.abstention_reasons


def test_small_volume_dynamical_recipes_lower_to_native_contracts():
    geometry = _su3_square_geometry()
    schedule = MeasurementSchedule(
        6,
        thermalization_trajectories=2,
        measurement_interval=2,
    )
    links = geometry.link_space.identity()
    nf2 = WilsonCloverNf2Recipe(
        geometry,
        schedule,
        beta=5.6,
        mass=0.2,
        spectral_lower=0.01,
        spectral_upper=100.0,
        step_size=0.005,
        trajectory_steps=1,
    )
    prepared_nf2 = prepare_wilson_clover_nf2_recipe(nf2, links)
    assert prepared_nf2.pseudofermion.determinant_power == 1.0
    assert prepared_nf2.kernel.registry.registry_id == prepared_nf2.registry.registry_id
    sources, source_ids = point_source_vectors(
        prepared_nf2.dirac, 0, source_id_prefix="origin"
    )
    propagator = solve_propagators(
        prepare_propagator_solve(
            PropagatorSolvePlan(
                prepared_nf2.dirac,
                sources,
                source_ids,
                relative_tolerance=1.0e-6,
                absolute_tolerance=1.0e-8,
                maximum_steps=128,
            )
        )
    )
    assert bool(jnp.all(propagator.status == 0))
    assert float(jnp.max(propagator.relative_residual)) < 1.0e-6
    clover = WilsonCloverNf2Recipe(
        geometry,
        schedule,
        beta=5.6,
        mass=0.2,
        variant="clover",
        clover_coefficient=1.1,
        spectral_lower=0.01,
        spectral_upper=100.0,
        step_size=0.005,
        trajectory_steps=1,
    )
    prepared_clover = prepare_wilson_clover_nf2_recipe(clover, links)
    assert prepared_clover.dirac.inverse_evidence.storage_bytes > 0
    assert bool(jnp.all(prepared_clover.dirac.inverse_evidence.finite))

    staggered = StaggeredHisqStyleRHMCRecipe(
        geometry,
        schedule,
        beta=6.0,
        mass=0.15,
        flavor_count=2,
        link_improvement_id="native-stout-improved-links",
        spectral_lower=0.1,
        spectral_upper=10.0,
        num_poles=2,
        verification_points=129,
        rational_tolerance=10.0,
        step_size=0.005,
        trajectory_steps=1,
    )
    prepared_staggered = prepare_staggered_hisq_style_rhmc_recipe(staggered, links)
    assert prepared_staggered.pseudofermion.determinant_power == 0.25
    assert bool(prepared_staggered.action_approximation.successful)
    assert bool(prepared_staggered.refresh_approximation.successful)
