import hashlib

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.applications.astrodynamics import (
    AstrodynamicsContext,
    AstrodynamicsScaleContract,
    FrameDefinition,
    JulianDate,
    ReferenceEpoch,
    TimeInstant,
)
from phydrax.applications.dark_matter._profiles import SmoothStellarRadialProfile
from phydrax.applications.dark_matter._rates import (
    observation_flux,
    spherical_surface_crossings,
)
from phydrax.applications.dark_matter._scattering import (
    BoundedThermalMarkSamplerPlan,
    ElasticScatteringTable,
)
from phydrax.applications.dark_matter._solar import (
    classify_solar_outcomes,
    gravitational_focusing_speed,
    kepler_specific_energy,
    propagate_exterior_kepler,
    SolarTransportPlan,
    stellar_specific_energy_evidence,
)
from phydrax.applications.dark_matter._transport import (
    BodyFrameTransportState,
    TransportOutcome,
)
from phydrax.integration import WeightedSampleBatch
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.stochastic import PoissonClockRealization
from phydrax.units import CENTIMETER, KILOGRAM, METER, SECOND


def _manifest(name):
    payload = name.encode()
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-Test-Only",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="test-only",
        nondimensionalization={
            "length_m": 1.0,
            "thermal_sigma_cutoff": 6.0,
            "maxwellian_tail_probability_bound": 7.488376948795484e-08,
        },
        uncertainty={"relative": 0.0},
        lineage_ids=(f"synthetic:{name}",),
    )


def _context(*, length_unit=METER, pseudo_inertial=True):
    frame = FrameDefinition("synthetic-star", "ICRF", pseudo_inertial=pseudo_inertial)
    epoch = ReferenceEpoch(TimeInstant(JulianDate(2451545.0), "TDB"))
    scale = AstrodynamicsScaleContract(length_unit, KILOGRAM, SECOND)
    return AstrodynamicsContext(scale, epoch, frame)


def _profile(context):
    return SmoothStellarRadialProfile(
        jnp.asarray((0.0, 5.0e6, 1.0e7)),
        jnp.asarray((2.0e3, 1.0e3, 0.0)),
        jnp.asarray(((4.0e29,), (2.0e29,), (0.0,))),
        jnp.asarray((1.0e6, 5.0e5, 1.0e4)),
        jnp.asarray((0.0, 4.0e21, 1.0e22)),
        ("H",),
        _manifest("synthetic-stellar-profile"),
        frame_id=context.frame.frame_id,
    )


def _zero_scattering():
    return ElasticScatteringTable(
        ("H",),
        jnp.asarray((1.6735575e-27,)),
        jnp.asarray((0.0, 1.0e6)),
        jnp.zeros((1, 2)),
        _manifest("synthetic-zero-scattering"),
        temperatures_K=jnp.asarray((1.0e4, 1.0e6)),
        rate_coefficients_m3_s=jnp.zeros((1, 2, 2)),
        mark_sampler=BoundedThermalMarkSamplerPlan(maximum_proposals=64),
    )


def _thermal_scattering():
    speeds = jnp.asarray((0.0, 1.0e6))
    cross_sections = 1.0e-30 * jnp.ones((1, 2))
    rates = jnp.broadcast_to(
        cross_sections[:, None, :] * speeds[None, None, :], (1, 2, 2)
    )
    return ElasticScatteringTable(
        ("H",),
        jnp.asarray((1.6735575e-27,)),
        speeds,
        cross_sections,
        _manifest("synthetic-thermal-scattering"),
        temperatures_K=jnp.asarray((1.0e4, 1.0e6)),
        rate_coefficients_m3_s=rates,
        mark_sampler=BoundedThermalMarkSamplerPlan(maximum_proposals=64),
    )


def test_stellar_enclosed_mass_interpolation_is_center_regular():
    profile = _profile(_context())
    evaluated = profile.evaluate(
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0e3, 0.0, 0.0),
                (2.0e3, 0.0, 0.0),
            )
        )
    )
    energy = stellar_specific_energy_evidence(
        profile,
        jnp.asarray((2.0e3, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )

    assert evaluated.gravitational_acceleration_m_s2[0] == 0.0
    assert jnp.allclose(
        evaluated.gravitational_acceleration_m_s2[2],
        2.0 * evaluated.gravitational_acceleration_m_s2[1],
        rtol=1.0e-3,
    )
    assert energy.finite
    assert energy.quadrature_error_m2_s2 >= 0.0
    assert energy.sign_qualified


def test_solar_transport_rejects_non_si_or_non_inertial_contexts():
    contexts = (
        _context(length_unit=CENTIMETER),
        _context(pseudo_inertial=False),
    )
    for context in contexts:
        profile = _profile(context)
        with pytest.raises(ValueError, match="continuous, pseudo-inertial physical SI"):
            SolarTransportPlan(
                profile,
                _zero_scattering(),
                1.0,
                context,
                observation_radius_m=2.0e7,
            )
        state = BodyFrameTransportState(
            jnp.asarray((2.0e7, 0.0, 0.0)),
            jnp.asarray((0.0, 300.0, 0.0)),
            frame_id=context.frame.frame_id,
        )
        with pytest.raises(ValueError, match="continuous, pseudo-inertial physical SI"):
            propagate_exterior_kepler(state, 1.0, 1.0e22, context)


def test_exterior_kepler_propagation_preserves_specific_energy():
    context = _context()
    state = BodyFrameTransportState(
        jnp.asarray((2.0e7, 0.0, 0.0)),
        jnp.asarray((0.0, 300.0, 0.0)),
        frame_id=context.frame.frame_id,
    )
    result = propagate_exterior_kepler(state, 1000.0, 1.0e22, context)
    compiled_energy = jax.jit(
        lambda elapsed, mass: (
            propagate_exterior_kepler(
                state, elapsed, mass, context
            ).specific_energy_after_m2_s2
        )
    )(jnp.asarray(1000.0), jnp.asarray(1.0e22))

    assert result.valid
    assert jnp.allclose(
        result.specific_energy_before_m2_s2,
        result.specific_energy_after_m2_s2,
        rtol=1.0e-8,
        atol=1.0e-4,
    )
    assert jnp.allclose(
        kepler_specific_energy(result.state, 1.0e22),
        result.specific_energy_after_m2_s2,
    )
    assert jnp.allclose(compiled_energy, result.specific_energy_after_m2_s2)


def test_zero_cross_section_focusing_is_fixed_by_kepler_energy():
    context = _context()
    profile = _profile(context)
    observation_radius = 2.0e7
    observation_speed = 300.0
    surface_speed = gravitational_focusing_speed(
        observation_speed,
        observation_radius,
        profile.radius_m,
        profile.total_mass_kg,
    )
    compiled_surface_speed = jax.jit(gravitational_focusing_speed)(
        jnp.asarray(observation_speed),
        jnp.asarray(observation_radius),
        profile.radius_m,
        profile.total_mass_kg,
    )
    observation_energy = (
        0.5 * observation_speed**2
        - 6.67430e-11 * profile.total_mass_kg / observation_radius
    )
    surface_energy = (
        0.5 * surface_speed**2 - 6.67430e-11 * profile.total_mass_kg / profile.radius_m
    )
    zero_scattering = _zero_scattering()
    endpoint_rate = zero_scattering.partial_rates(
        jnp.asarray((1.0,)),
        jnp.asarray((1.0e6, 0.0, 0.0)),
        1.0e6,
    )
    plan = SolarTransportPlan(
        profile,
        zero_scattering,
        1.0,
        context,
        observation_radius_m=observation_radius,
    )
    with pytest.raises(ValueError, match="exactly one reachable terminal guard"):
        SolarTransportPlan(
            profile,
            zero_scattering,
            1.0,
            context,
            observation_radius_m=observation_radius,
            maximum_guard_events=2,
        )
    bound_surface = jnp.asarray(((profile.radius_m, 0.0, 0.0, 50.0, 0.0, 0.0),))
    _, bound_observed, bound_final, bound_valid = plan._observation_crossing(
        bound_surface,
        jnp.asarray((1.0e5,)),
        jnp.asarray((True,)),
    )
    bound_energy = kepler_specific_energy(bound_final, profile.total_mass_kg)
    bound_outcome = classify_solar_outcomes(
        bound_energy,
        jnp.asarray((0,)),
        bound_observed,
        jnp.asarray((True,)),
        bound_valid,
        energy_sign_qualified=jnp.asarray((True,)),
    )

    assert jnp.all(zero_scattering.cross_sections_m2 == 0.0)
    assert endpoint_rate[0] == 0.0
    assert jnp.allclose(surface_energy, observation_energy)
    assert jnp.allclose(compiled_surface_speed, surface_speed)
    assert surface_speed > observation_speed
    assert not bound_observed[0]
    assert bound_valid[0]
    assert jnp.array_equal(bound_final[0], bound_surface[0])
    assert bound_outcome.outcomes[0] == int(TransportOutcome.CAPTURED)


def test_solar_simulation_uses_guarded_interior_and_analytic_exterior():
    context = _context()
    profile = _profile(context)
    observation_radius = 2.0e7
    surface_speed = gravitational_focusing_speed(
        300.0,
        observation_radius,
        profile.radius_m,
        profile.total_mass_kg,
    )
    plan = SolarTransportPlan(
        profile,
        _zero_scattering(),
        1.0,
        context,
        observation_radius_m=observation_radius,
        maximum_jump_events=4,
        maximum_guard_events=1,
    )
    transverse_speed = 50.0
    radial_speed = jnp.sqrt(surface_speed**2 - transverse_speed**2)
    states = jnp.broadcast_to(
        jnp.asarray((profile.radius_m, 0.0, 0.0, -radial_speed, transverse_speed, 0.0)),
        (2, 6),
    )
    states = states.at[1].set(jnp.nan)
    paths = WeightedSampleBatch(
        states,
        jnp.zeros((2,)),
        support_valid=jnp.asarray(True),
        mask=jnp.asarray((True, False)),
        sample_axes=0,
        provenance="synthetic-solar-surface-injection",
    )
    clocks = PoissonClockRealization(
        jr.key(17),
        1,
        support=(0.0, 1.2e5),
        max_events_per_channel=4,
        sample_shape=(2,),
        process_id=plan.process.process_id,
    )
    result = plan.simulate(paths, clocks, jnp.asarray((0.0, 1.2e5)))

    assert result.solution.terminal[0]
    assert result.evidence.successful.tolist() == [True, False]
    assert result.outcomes.tolist() == [
        int(TransportOutcome.ESCAPED),
        int(TransportOutcome.UNRESOLVED),
    ]
    assert result.observation_crossings.diagnostics.crossing_count == 1
    assert jnp.sqrt(jnp.sum(result.final_states[0, :3] ** 2)) >= observation_radius


def test_thermal_target_marks_follow_rate_weighted_bounded_law():
    table = _thermal_scattering()
    temperature = 1.0e6
    projectile = jnp.asarray((2.0e5, 0.0, 0.0))
    keys = jr.split(jr.key(4), 2048)
    marks = jax.vmap(lambda key: table.sample_mark(key, 0, projectile, temperature))(keys)

    assert jnp.all(jnp.isfinite(marks))
    assert jnp.mean(marks[:, 0]) < -1.0e4
    assert jnp.allclose(jnp.mean(marks[:, 1:3], axis=0), 0.0, atol=5.0e3)
    assert jnp.allclose(jnp.sum(marks[:, 3:] ** 2, axis=-1), 1.0, atol=1.0e-6)
    assert table.mark_sampler.maxwellian_tail_probability_bound < 1.0e-7


def test_observation_radius_flux_uses_spherical_area_and_exposure():
    radius = 2.0e7
    state = jnp.asarray((radius, 0.0, 0.0, 300.0, 0.0, 0.0))
    source = WeightedSampleBatch(
        state[None, :],
        jnp.log(jnp.asarray((2.0,))),
        sample_axes=0,
        provenance="synthetic-observation-flux",
    )
    crossings = spherical_surface_crossings(
        state[None, None, :],
        jnp.ones((1, 1), dtype=bool),
        source,
        radius,
        direction="outward",
    )
    flux = observation_flux(crossings, 2.0)

    assert flux.valid
    assert jnp.allclose(flux.total_crossing_weight, 2.0)
    assert jnp.allclose(flux.flux_m2_s, 1.0 / (4.0 * jnp.pi * radius**2))


def test_solar_outcomes_are_exclusive_and_numerical_failure_is_unresolved():
    classified = classify_solar_outcomes(
        jnp.asarray((1.0, -1.0, 1.0, 1.0, -1.0e-15)),
        jnp.asarray((0, 3, 2, 4, 1)),
        jnp.asarray((True, False, False, True, False)),
        jnp.asarray((True, False, True, True, False)),
        jnp.asarray((True, True, True, False, True)),
        energy_sign_qualified=jnp.asarray((True, True, True, True, False)),
    )

    assert classified.outcomes.tolist() == [
        int(TransportOutcome.ESCAPED),
        int(TransportOutcome.CAPTURED),
        int(TransportOutcome.REFLECTED),
        int(TransportOutcome.UNRESOLVED),
        int(TransportOutcome.UNRESOLVED),
    ]
    assert jnp.all(jnp.sum(classified.one_hot, axis=-1) == 1)
