import hashlib

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.applications.dark_matter._profiles import LayeredTerrestrialProfile
from phydrax.applications.dark_matter._rates import (
    CrossingStatus,
    observation_flux,
    spherical_surface_crossings,
)
from phydrax.applications.dark_matter._scattering import (
    BoundedThermalMarkSamplerPlan,
    elastic_scatter_velocity,
    ElasticScatteringTable,
)
from phydrax.applications.dark_matter._terrestrial import (
    HazardQuadraturePlan,
    layered_analytic_optical_depth,
    quadrature_optical_depth,
    sample_target_from_partial_rates,
)
from phydrax.applications.dark_matter._transport import (
    ProfiledElasticJumpProcess,
    transparent_state,
    transport_path_evidence,
    TransportNumericalStatus,
)
from phydrax.integration import WeightedSampleBatch
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.stochastic import JUMP_INVALID_INTENSITY, JUMP_MAX_EVENTS, JUMP_SUCCESS


def _manifest(name, *, commercial=False, thermal_sigma_cutoff=6.0):
    payload = name.encode()
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-Test-Only",
        commercial_use_permitted=commercial,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="test-only",
        nondimensionalization={
            "length_m": 1.0,
            "thermal_sigma_cutoff": thermal_sigma_cutoff,
            "maxwellian_tail_probability_bound": 7.488376948795484e-08,
        },
        uncertainty={"relative": 0.0},
        lineage_ids=(f"synthetic:{name}",),
    )


def _profile():
    return LayeredTerrestrialProfile(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((1.0, 2.0)),
        jnp.asarray(((2.0, 3.0), (5.0, 7.0))),
        jnp.asarray((100.0, 300.0)),
        ("A", "B"),
        _manifest("synthetic-layer-profile"),
        frame_id="synthetic-body-frame",
    )


def _homogeneous_profile():
    return LayeredTerrestrialProfile(
        jnp.asarray((2.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(((0.2, 0.3),)),
        jnp.asarray((200.0,)),
        ("A", "B"),
        _manifest("synthetic-homogeneous-profile"),
        frame_id="synthetic-body-frame",
    )


def _scattering():
    speeds = jnp.asarray((0.0, 10.0))
    cross_sections = jnp.asarray(((0.25, 0.25), (0.5, 0.5)))
    rates = cross_sections[:, None, :] * speeds[None, None, :]
    rates = jnp.broadcast_to(rates, (2, 2, 2))
    return ElasticScatteringTable(
        ("A", "B"),
        jnp.asarray((1.0, 4.0)),
        speeds,
        cross_sections,
        _manifest("synthetic-elastic-table"),
        temperatures_K=jnp.asarray((100.0, 300.0)),
        rate_coefficients_m3_s=rates,
        mark_sampler=BoundedThermalMarkSamplerPlan(maximum_proposals=64),
    )


def test_requested_use_rights_are_enforced_and_bound_into_identity():
    manifest = _manifest("synthetic-rights-profile", commercial=True)
    arguments = (
        jnp.asarray((2.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(((1.0,),)),
        jnp.asarray((200.0,)),
        ("A",),
        manifest,
    )
    default = LayeredTerrestrialProfile(*arguments, frame_id="rights-frame")
    commercial = LayeredTerrestrialProfile(
        *arguments, frame_id="rights-frame", commercial_use=True
    )
    assert default.profile_id != commercial.profile_id
    with pytest.raises(PermissionError):
        LayeredTerrestrialProfile(
            *arguments,
            frame_id="rights-frame",
            redistribution=True,
        )
    speeds = jnp.asarray((0.0, 10.0))
    coefficients = jnp.asarray((((0.0, 1.0), (0.0, 1.0)),))
    sampler = BoundedThermalMarkSamplerPlan(maximum_proposals=8)
    default_table = ElasticScatteringTable(
        ("A",),
        jnp.asarray((1.0,)),
        speeds,
        jnp.asarray(((0.1, 0.1),)),
        manifest,
        temperatures_K=jnp.asarray((100.0, 300.0)),
        rate_coefficients_m3_s=coefficients,
        mark_sampler=sampler,
    )
    commercial_table = ElasticScatteringTable(
        ("A",),
        jnp.asarray((1.0,)),
        speeds,
        jnp.asarray(((0.1, 0.1),)),
        manifest,
        temperatures_K=jnp.asarray((100.0, 300.0)),
        rate_coefficients_m3_s=coefficients,
        mark_sampler=sampler,
        commercial_use=True,
    )
    assert default_table.table_id != commercial_table.table_id
    with pytest.raises(PermissionError):
        ElasticScatteringTable(
            ("A",),
            jnp.asarray((1.0,)),
            speeds,
            jnp.asarray(((0.1, 0.1),)),
            manifest,
            temperatures_K=jnp.asarray((100.0, 300.0)),
            rate_coefficients_m3_s=coefficients,
            mark_sampler=sampler,
            redistribution=True,
        )
    with pytest.raises(ValueError, match="bind the mark cutoff"):
        ElasticScatteringTable(
            ("A",),
            jnp.asarray((1.0,)),
            speeds,
            jnp.asarray(((0.1, 0.1),)),
            _manifest(
                "synthetic-mismatched-truncation",
                thermal_sigma_cutoff=5.0,
            ),
            temperatures_K=jnp.asarray((100.0, 300.0)),
            rate_coefficients_m3_s=coefficients,
            mark_sampler=sampler,
        )


def test_transparent_path_and_layer_transitions_are_geometrically_explicit():
    state = jnp.asarray((-3.0, 0.0, 0.0, 2.0, 0.0, 0.0))
    propagated = transparent_state(state, 1.25)
    evaluated = _profile().evaluate(
        jnp.asarray(((0.5, 0.0, 0.0), (1.5, 0.0, 0.0), (2.5, 0.0, 0.0)))
    )

    assert jnp.allclose(propagated, jnp.asarray((-0.5, 0.0, 0.0, 2.0, 0.0, 0.0)))
    assert evaluated.layer_index.tolist() == [0, 1, -1]
    assert evaluated.inside.tolist() == [True, True, False]
    assert jnp.allclose(
        _profile().boundaries[0].signed_distance(jnp.asarray(((1.0, 0.0, 0.0),))),
        0.0,
    )


def test_layered_analytic_hazard_recovers_homogeneous_exponential_optical_depth():
    result = layered_analytic_optical_depth(
        _homogeneous_profile(),
        _scattering(),
        jnp.asarray((-3.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        6.0,
        4.0,
    )

    assert result.successful
    assert jnp.allclose(result.partial_optical_depths, jnp.asarray((0.2, 0.6)))
    assert jnp.allclose(result.total_optical_depth, 0.8)
    assert jnp.allclose(result.interaction_cdf, 1.0 - jnp.exp(-0.8))
    thresholds = jr.exponential(jr.key(9), (32768,))
    empirical = jnp.mean(thresholds <= result.total_optical_depth)
    assert jnp.abs(empirical - result.interaction_cdf) < 0.01


def test_generic_hazard_quadrature_converges_to_layered_realization():
    profile = _profile()
    scattering = _scattering()
    exact = layered_analytic_optical_depth(
        profile,
        scattering,
        jnp.asarray((-3.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        6.0,
        4.0,
    )
    quadrature = quadrature_optical_depth(
        profile,
        scattering,
        jnp.asarray((-3.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        6.0,
        4.0,
        quadrature=HazardQuadraturePlan(128),
    )

    assert quadrature.successful
    assert jnp.allclose(
        quadrature.partial_optical_depths,
        exact.partial_optical_depths,
        rtol=1.0e-10,
    )


def test_vacuum_layers_mask_out_of_support_rate_coefficients():
    vacuum = LayeredTerrestrialProfile(
        jnp.asarray((2.0,)),
        jnp.zeros((1,)),
        jnp.zeros((1, 2)),
        jnp.asarray((200.0,)),
        ("A", "B"),
        _manifest("synthetic-vacuum-layer"),
        frame_id="synthetic-body-frame",
    )
    arguments = (
        vacuum,
        _scattering(),
        jnp.asarray((-3.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        6.0,
        10.0,
    )
    analytic = layered_analytic_optical_depth(*arguments)
    quadrature = quadrature_optical_depth(*arguments, quadrature=HazardQuadraturePlan(16))

    assert analytic.successful
    assert quadrature.successful
    assert analytic.total_optical_depth == 0.0
    assert quadrature.total_optical_depth == 0.0


def test_partial_hazards_select_only_supported_targets():
    rates = jnp.asarray((0.0, 4.0, 0.0))
    selected = jnp.stack(
        tuple(
            sample_target_from_partial_rates(jr.fold_in(jr.key(0), index), rates)
            for index in range(32)
        )
    )
    assert jnp.all(selected == 1)
    assert sample_target_from_partial_rates(jr.key(1), jnp.zeros((3,))) == -1


def test_profiled_marked_jump_process_exposes_partial_target_rates_and_marks():
    process = ProfiledElasticJumpProcess(_profile(), _scattering(), 2.0)
    state = jnp.asarray((0.5, 0.0, 0.0, 4.0, 0.0, 0.0))
    rates = process.intensities(0.0, state)
    mark = process.sample_mark(jr.key(5), 0.0, state, 1)
    collision = process.collision_evidence(state, 1, mark)
    after = process.jump(state, 1, mark)

    assert jnp.allclose(rates, jnp.asarray((2.0, 6.0)))
    assert mark.shape == (6,)
    assert collision.successful
    assert jnp.array_equal(after[:3], state[:3])
    assert jnp.array_equal(after[3:], collision.projectile_velocity_m_s)


def test_rate_table_interpolation_and_elastic_invariants_are_exact():
    scattering = _scattering()
    coefficient = scattering.rate_coefficients(5.5, 200.0)
    collision = elastic_scatter_velocity(
        jnp.asarray((3.0, -2.0, 1.0)),
        jnp.asarray((-1.0, 0.5, 2.0)),
        jnp.asarray((0.0, 1.0, 0.0)),
        2.0,
        5.0,
    )
    invalid_mark = elastic_scatter_velocity(
        jnp.asarray((3.0, -2.0, 1.0)),
        jnp.asarray((-1.0, 0.5, 2.0)),
        jnp.zeros((3,)),
        2.0,
        5.0,
    )
    mixed_support = ElasticScatteringTable(
        ("present", "absent"),
        jnp.asarray((1.0, 1.0e-27)),
        jnp.asarray((0.0, 10.0)),
        jnp.ones((2, 2)),
        _manifest("synthetic-mixed-mark-support"),
        temperatures_K=jnp.asarray((100.0, 300.0)),
        rate_coefficients_m3_s=jnp.ones((2, 2, 2)),
        mark_sampler=BoundedThermalMarkSamplerPlan(maximum_proposals=8),
    )
    mixed_rates = mixed_support.partial_rates(
        jnp.asarray((1.0, 0.0)),
        jnp.asarray((5.0, 0.0, 0.0)),
        300.0,
    )

    assert jnp.allclose(coefficient, jnp.asarray((1.375, 2.75)))
    assert not jnp.any(scattering.support(11.0, 200.0))
    assert jnp.all(jnp.isnan(scattering.rate_coefficients(11.0, 200.0)))
    assert not jnp.any(scattering.support(10.0, 200.0))
    assert jnp.all(jnp.isnan(scattering.rate_coefficients(10.0, 200.0)))
    assert jnp.isfinite(mixed_rates[0])
    assert mixed_rates[1] == 0.0
    assert collision.successful
    assert collision.scattered
    assert collision.mass_valid
    assert collision.direction_valid
    assert collision.conservative
    assert jnp.allclose(collision.momentum_before_kg_m_s, collision.momentum_after_kg_m_s)
    assert jnp.allclose(
        collision.kinetic_energy_before_J, collision.kinetic_energy_after_J
    )
    assert not invalid_mark.direction_valid
    assert not invalid_mark.successful
    assert jnp.array_equal(
        invalid_mark.projectile_velocity_m_s, jnp.asarray((3.0, -2.0, 1.0))
    )


def test_weighted_surface_crossing_preserves_flux_and_applies_density_jacobian():
    source = WeightedSampleBatch(
        jnp.asarray(((0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))),
        jnp.log(jnp.asarray((2.0, 3.0))),
        ancestry_ids=jnp.asarray((71, 93)),
        stratum_ids=jnp.asarray((3, 4)),
        pair_ids=jnp.asarray((10, 11)),
        replicate_ids=jnp.asarray((20, 21)),
        sample_axes=0,
        provenance="synthetic-incoming-flux",
        independent=True,
    )
    crossing_states = jnp.asarray(
        ((1.0, 0.0, 0.0, 2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 4.0, 0.0, 0.0))
    )
    crossing = spherical_surface_crossings(
        crossing_states[:, None, :],
        jnp.ones((2, 1), dtype=bool),
        source,
        1.0,
        direction="outward",
    )

    flux_weights = jnp.exp(crossing.flux.log_weights[crossing.flux.mask])
    density_weights = jnp.exp(crossing.density.log_weights[crossing.density.mask])
    assert jnp.allclose(flux_weights, jnp.asarray((2.0, 3.0)))
    assert jnp.allclose(density_weights, jnp.asarray((1.0, 0.75)))
    assert jnp.allclose(
        crossing.initial_to_crossing_density_jacobian, jnp.asarray((0.5, 0.25))
    )
    assert jnp.allclose(crossing.initial_speeds_m_s, 1.0)
    assert jnp.allclose(crossing.crossing_speeds_m_s, jnp.asarray((2.0, 4.0)))
    assert jnp.allclose(crossing.diagnostics.effective_sample_size, 25.0 / 13.0)
    assert jnp.array_equal(crossing.flux.ancestry_ids, jnp.asarray((71, 93)))
    assert jnp.array_equal(crossing.flux.stratum_ids, jnp.asarray((3, 4)))
    assert jnp.array_equal(crossing.flux.pair_ids, jnp.asarray((10, 11)))
    assert jnp.array_equal(crossing.flux.replicate_ids, jnp.asarray((20, 21)))
    oblique_source = WeightedSampleBatch(
        jnp.asarray(((0.0, 0.0, 0.0, 2.0, 0.0, 0.0),)),
        jnp.zeros((1,)),
        ancestry_ids=jnp.asarray((42,)),
        sample_axes=0,
        provenance="synthetic-oblique-source",
    )
    oblique_states = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 1.0, 1.0, 0.0)),
        (1, 2, 6),
    )
    oblique = spherical_surface_crossings(
        oblique_states,
        jnp.ones((1, 2), dtype=bool),
        oblique_source,
        1.0,
        direction="outward",
    )
    assert jnp.allclose(oblique.initial_to_crossing_density_jacobian, 2.0)
    assert not jnp.allclose(
        oblique.initial_to_crossing_density_jacobian,
        oblique.initial_speeds_m_s / oblique.crossing_speeds_m_s,
    )
    assert jnp.array_equal(oblique.flux.ancestry_ids, jnp.asarray((42, 42)))
    unsupported = WeightedSampleBatch(
        source.samples,
        source.log_weights,
        support_valid=jnp.asarray(False),
        sample_axes=0,
        provenance="synthetic-unsupported-flux",
    )
    unsupported_crossing = spherical_surface_crossings(
        crossing_states[:, None, :],
        jnp.ones((2, 1), dtype=bool),
        unsupported,
        1.0,
    )
    assert unsupported_crossing.diagnostics.status == int(CrossingStatus.INVALID_INPUT)


def test_crossing_and_observation_reductions_accept_traced_evidence():
    states = jnp.asarray((1.0, 0.0, 0.0, 2.0, 0.0, 0.0))[None, :]

    @jax.jit
    def evaluate(support_valid, exposure):
        source = WeightedSampleBatch(
            states,
            jnp.zeros((1,)),
            support_valid=support_valid,
            sample_axes=0,
            provenance="jitted-crossing",
        )
        crossing = spherical_surface_crossings(
            states[:, None, :],
            jnp.ones((1, 1), dtype=bool),
            source,
            jnp.asarray(1.0),
            direction="outward",
        )
        flux = observation_flux(crossing, exposure)
        return crossing.diagnostics.status, flux.flux_m2_s, flux.valid

    status, flux, valid = evaluate(jnp.asarray(True), jnp.asarray(2.0))
    invalid_status, _, invalid = evaluate(jnp.asarray(False), jnp.asarray(2.0))
    assert status == int(CrossingStatus.SUCCESS)
    assert jnp.isfinite(flux)
    assert valid
    assert invalid_status == int(CrossingStatus.INVALID_INPUT)
    assert not invalid


def test_numerical_failures_do_not_alias_physical_outcomes():
    evidence = transport_path_evidence(
        jnp.asarray(
            (JUMP_SUCCESS, JUMP_INVALID_INTENSITY, JUMP_MAX_EVENTS, JUMP_SUCCESS)
        ),
        jnp.asarray((1, 1, 1, 1)),
        jnp.asarray((False, False, False, True)),
        jnp.asarray((True, True, True, True)),
    )

    assert evidence.numerical_status.tolist() == [
        int(TransportNumericalStatus.SUCCESS),
        int(TransportNumericalStatus.INVALID_RATE),
        int(TransportNumericalStatus.JUMP_EVENT_CAPACITY),
        int(TransportNumericalStatus.GUARD_EVENT_CAPACITY),
    ]
    assert evidence.successful.tolist() == [True, False, False, False]
    invalid_initial = transport_path_evidence(
        jnp.asarray((JUMP_SUCCESS,)),
        jnp.asarray((0,)),
        jnp.asarray((False,)),
        jnp.asarray((True,)),
        initial_valid=jnp.asarray((False,)),
    )
    assert invalid_initial.numerical_status[0] == int(
        TransportNumericalStatus.INVALID_INITIAL_STATE
    )
    solver_failure = transport_path_evidence(
        jnp.asarray((JUMP_SUCCESS,)),
        jnp.asarray((0,)),
        jnp.asarray((False,)),
        jnp.asarray((True,)),
        solver_successful=jnp.asarray((False,)),
    )
    assert solver_failure.numerical_status[0] == int(
        TransportNumericalStatus.DIFFERENTIAL_SOLVER_FAILURE
    )
    collision_failure = transport_path_evidence(
        jnp.asarray((JUMP_SUCCESS,)),
        jnp.asarray((0,)),
        jnp.asarray((False,)),
        jnp.asarray((True,)),
        collision_invariants_valid=jnp.asarray((False,)),
    )
    assert collision_failure.numerical_status[0] == int(
        TransportNumericalStatus.COLLISION_INVARIANT_FAILURE
    )
