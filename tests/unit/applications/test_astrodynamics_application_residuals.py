import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


astro = phx.applications.astrodynamics


def _context():
    return astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0), "TT")),
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
    )


def _record(mean_motion, eccentricity=0.1):
    return astro.TleRecord(
        "synthetic-line-1",
        "synthetic-line-2",
        42,
        "U",
        "synthetic",
        astro.TimeInstant(astro.JulianDate(2451545.0), "UTC"),
        0.0,
        0.0,
        0.0,
        np.deg2rad(63.4),
        0.1,
        eccentricity,
        0.2,
        0.3,
        mean_motion,
        1,
    )


def _event_vector_field(time, state, args):
    del time, args
    return jnp.zeros_like(state)


def test_single_pair_regularization_crosses_close_orbit_and_rolls_back_ambiguity():
    context = _context()
    radius = 0.1
    speed = np.sqrt(2.0 / radius)
    positions = jnp.asarray([[-0.5 * radius, 0.0, 0.0], [0.5 * radius, 0.0, 0.0]])
    velocities = jnp.asarray([[0.0, -0.5 * speed, 0.0], [0.0, 0.5 * speed, 0.0]])
    policy = astro.CloseEncounterPolicy(0.2, 1.0e-4)
    encounter = astro.detect_close_encounter(
        positions, policy, regularization_prepared=True
    )
    plan = astro.CloseEncounterRegularizationPlan(
        0.2,
        1.0e-4,
        maximum_fictitious_steps=48,
        physical_time_tolerance=1.0e-9,
    )
    prepared = plan.prepare(jnp.ones(2), positions, velocities, encounter, context)
    period = 2.0 * np.pi * np.sqrt(radius**3 / 2.0)
    result = jax.jit(prepared.propagate)(0.25 * period)
    assert bool(result.successful)
    assert result.pair.tolist() == [0, 1]
    assert result.time_residual <= 1.0e-9
    assert result.energy_residual <= 1.0e-6
    assert result.angular_momentum_residual <= 1.0e-6
    np.testing.assert_allclose(jnp.sum(result.positions, axis=0), 0.0, atol=1.0e-8)

    crowded = jnp.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]])
    crowded_velocity = jnp.zeros_like(crowded)
    ambiguous = astro.detect_close_encounter(crowded, policy)
    rejected = plan.prepare(
        jnp.ones(3), crowded, crowded_velocity, ambiguous, context
    ).propagate(0.01)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.positions, crowded)
    np.testing.assert_array_equal(rejected.velocities, crowded_velocity)


def test_tle_static_regimes_resonances_and_range_failure():
    record = _record(10.0)
    with pytest.raises(ValueError, match="continuous solver epoch"):
        astro.ReferenceEpoch(record.epoch)
    native_context = astro.TLEPropagationPlan.native_context(record)
    si_teme_context = astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        native_context.epoch,
        native_context.frame,
    )
    km_icrf_context = astro.AstrodynamicsContext(
        native_context.scale,
        native_context.epoch,
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
    )
    with pytest.raises(ValueError, match="contexts are incompatible"):
        astro.TLEPropagationPlan(record, si_teme_context, maximum_minutes=1440.0)
    with pytest.raises(ValueError, match="contexts are incompatible"):
        astro.TLEPropagationPlan(record, km_icrf_context, maximum_minutes=1440.0)
    near = astro.TLEPropagationPlan(record, native_context, maximum_minutes=1440.0)
    assert near.regime == "near-earth"
    near_result = jax.jit(near.propagate)(60.0)
    assert bool(near_result.valid)
    assert near_result.frame == "TEME"
    assert near_result.position_unit == "km"
    assert near_result.velocity_unit == "km/s"
    assert near_result.state.context.context_id == native_context.context_id
    assert near_result.state.context.frame.origin_id == "earth"
    assert near_result.state.context.frame.orientation_id == "TEME"
    assert near_result.state.context.scale.length_to_reference == 1000.0
    assert near_result.state.context.epoch.instant.instant_id == record.epoch.instant_id
    assert not near_result.state.context.epoch.continuous
    assert near_result.epoch.scale == "UTC"
    np.testing.assert_allclose(near_result.epoch.offset_seconds, 3600.0)
    np.testing.assert_allclose(near_result.epoch.julian_date_high, 2451545.0)
    np.testing.assert_allclose(near_result.epoch.julian_date_low, 60.0 / 1440.0)
    np.testing.assert_allclose(near_result.epoch.julian_date, 2451545.0 + 60.0 / 1440.0)

    boundary = astro.TLEPropagationPlan(_record(6.39))
    below_boundary = astro.TLEPropagationPlan(_record(6.41))
    assert boundary.regime == "deep-space"
    assert below_boundary.regime == "near-earth"

    synchronous = astro.TLEPropagationPlan(
        _record(1.0), maximum_minutes=4320.0, resonance_step_minutes=360.0
    )
    sync_result = jax.jit(synchronous.propagate)(720.0)
    assert synchronous.regime == "deep-space"
    assert synchronous.resonance_kind == "synchronous"
    assert sync_result.resonance_steps == 2
    assert bool(sync_result.range_valid)

    twelve_hour = astro.TLEPropagationPlan(
        _record(2.0, eccentricity=0.7), maximum_minutes=2880.0
    )
    assert twelve_hour.resonance_kind == "twelve-hour"
    assert bool(twelve_hour.propagate(-720.0).range_valid)
    overflow = twelve_hour.propagate(3000.0)
    assert not bool(overflow.valid)
    assert not bool(overflow.range_valid)
    assert int(overflow.status) == int(astro.AstrodynamicsStatus.CAPACITY_EXCEEDED)
    nonfinite = twelve_hour.propagate(jnp.nan)
    assert not bool(nonfinite.valid)
    assert int(nonfinite.status) == int(astro.AstrodynamicsStatus.NONFINITE_INPUT)


def test_tle_matches_vallado_near_earth_and_resonant_deep_space_vectors():
    cases = (
        (
            "near-earth",
            "none",
            "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
            "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
            (
                (
                    0.0,
                    (7022.46529266, -1400.08296755, 0.03995155),
                    (1.893841015, 6.405893759, 4.534807250),
                ),
                (
                    360.0,
                    (-7154.03120202, -3783.17682504, -3536.19412294),
                    (4.741887409, -4.151817765, -2.093935425),
                ),
            ),
        ),
        (
            "deep-space",
            "twelve-hour",
            "1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0   813",
            "2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383225656",
            (
                (
                    0.0,
                    (2349.89483350, -14785.93811562, 0.02119378),
                    (2.721488096, -3.256811655, 4.498416672),
                ),
                (
                    720.0,
                    (2622.13222207, -15125.15464924, 474.51048398),
                    (2.688287199, -3.078426664, 4.494979530),
                ),
            ),
        ),
        (
            "deep-space",
            "synchronous",
            "1 24208U 96044A   06177.04061740 -.00000094  00000-0  10000-3 0  1600",
            "2 24208   3.8536  80.0121 0026640 311.0977  48.3000  1.00778054 36119",
            (
                (
                    0.0,
                    (7534.10987189, 41266.39266843, -0.10801028),
                    (-3.027168008, 0.558848996, 0.207982755),
                ),
                (
                    1440.0,
                    (5501.08137100, 41590.27784405, 138.32522930),
                    (-3.050691874, 0.409203052, 0.207958133),
                ),
            ),
        ),
    )
    for regime, resonance, line1, line2, vectors in cases:
        record = astro.parse_tle(line1, line2)
        plan = astro.TLEPropagationPlan(record, maximum_minutes=2880.0)
        assert plan.regime == regime
        assert plan.resonance_kind == resonance
        for minutes, expected_position, expected_velocity in vectors:
            result = plan.propagate(minutes)
            assert bool(result.valid)
            assert int(result.status) == int(astro.AstrodynamicsStatus.SUCCESS)
            assert result.residual_indicator <= 1.0e-10
            assert result.frame == "TEME"
            assert result.position_unit == "km"
            assert result.velocity_unit == "km/s"
            assert result.state.context.frame.orientation_id == result.frame
            assert result.state.context.scale.length_unit == result.position_unit
            assert result.state.context.scale.velocity_unit == result.velocity_unit
            expected_epoch = (
                record.epoch.julian_date.high
                + record.epoch.julian_date.low
                + minutes / 1440.0
            )
            np.testing.assert_allclose(result.epoch.julian_date, expected_epoch)
            np.testing.assert_allclose(
                result.state.position, expected_position, rtol=0.0, atol=2.0e-5
            )
            np.testing.assert_allclose(
                result.state.velocity, expected_velocity, rtol=0.0, atol=2.0e-8
            )


def test_astrodynamics_event_ids_include_guard_and_reset_parameters():
    context = _context()

    def make(radius, delta_velocity):
        return astro.AstrodynamicsEventPlan(
            astro.RadiusGuard(radius),
            astro.ImpulsiveVelocityReset(delta_velocity),
            _event_vector_field,
            _event_vector_field,
            context,
            event_kind="radius-crossing",
        )

    reference = make(7000.0, (0.0, 0.01, 0.0))
    duplicate = make(7000.0, (0.0, 0.01, 0.0))
    different_radius = make(7100.0, (0.0, 0.01, 0.0))
    different_reset = make(7000.0, (0.0, 0.02, 0.0))
    assert reference.event_id == duplicate.event_id
    assert reference.hybrid.plan_id == duplicate.hybrid.plan_id
    assert reference.event_id != different_radius.event_id
    assert reference.event_id != different_reset.event_id
    assert reference.hybrid.plan_id != different_radius.hybrid.plan_id
    assert reference.hybrid.plan_id != different_reset.hybrid.plan_id


def test_bundled_astronomy_assets_are_typed_bounded_and_offline():
    context = _context()
    store = astro.bundled_astronomy_data_store()
    assert set(astro.ASTRONOMY_ASSET_MANIFESTS) == {
        "leap_seconds.json",
        "eop_cip_2024.json",
        "earth_gravity_degree4.json",
        "sun_earth_moon_chebyshev.json",
        "iau_precession_nutation.json",
    }
    leap = astro.load_bundled_leap_seconds(store)
    assert leap.tai_minus_utc[-1] == 37
    eop = astro.load_bundled_earth_orientation(store)
    assert bool(eop.evaluate(0.0).valid)
    assert not bool(eop.evaluate(6.0 * 86400.0).valid)
    gravity = astro.load_bundled_earth_gravity(context, store)
    assert gravity.maximum_degree == 4
    ephemeris = astro.load_bundled_sun_earth_moon_ephemeris(context, store)
    assert bool(ephemeris.evaluate(0.0, 1).valid)
    assert not bool(ephemeris.evaluate(2.0 * 86400.0, 1).valid)
    iau = astro.load_bundled_iau_coefficients(store)
    assert iau.coefficient("epsilon_0").shape == ()
    assert iau.coefficient("psib").shape == (5,)
