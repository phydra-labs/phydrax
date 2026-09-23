import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _context():
    astro = phx.applications.astrodynamics
    return astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0), "TT")),
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
    )


def _provenance(context):
    return phx.applications.astrodynamics.AstrodynamicsDataProvenance(
        producer="test",
        producer_version="1",
        source_id="stations",
        checksum="sha256:test",
        license_id="test",
        frame_id=context.frame.frame_id,
        epoch_id=context.epoch.epoch_id,
        scale_id=context.scale.scale_id,
        differentiability="constant",
    )


def test_coupled_vehicle_and_tracking_closure():
    astro = phx.applications.astrodynamics
    context = _context()
    configuration = astro.VehicleConfiguration(
        10.0,
        jnp.eye(3),
        jnp.asarray([[0.0, 0.0, 0.0]]),
        jnp.asarray([1.0]),
        jnp.empty((0, 3)),
        jnp.empty((0,)),
        context,
    )

    def zero_effector(time, state, command):
        del time, command
        return astro.VehicleEffectorEvaluation(
            jnp.zeros(3),
            jnp.zeros(3),
            jnp.zeros_like(state.tank_masses),
            jnp.zeros_like(state.wheel_momentum),
            jnp.asarray(True),
        )

    state = astro.VehicleState(
        jnp.zeros(3),
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        jnp.zeros(3),
        jnp.asarray([1.0]),
        jnp.empty((0,)),
    )
    result = astro.CoupledVehiclePlan(
        configuration,
        (zero_effector,),
        jnp.asarray([0.0, 1.0]),
        effector_ids=("zero",),
    ).rollout(state, lambda time: jnp.empty((0,)))
    assert bool(result.successful)
    np.testing.assert_allclose(result.states.position[-1], jnp.asarray([1.0, 0.0, 0.0]))

    stations = astro.TrackingStationCatalog(
        ("station",),
        jnp.zeros((1, 3)),
        jnp.zeros((1, 3)),
        jnp.asarray([-jnp.pi / 2.0]),
        context,
        _provenance(context),
    )
    schedule = astro.ObservationSchedule(
        jnp.asarray([0.0]),
        jnp.asarray([0]),
        jnp.asarray([0]),
        jnp.zeros((1, 2)),
        jnp.eye(2)[None],
        jnp.asarray([False]),
        ("range", "range_rate", "azimuth_elevation", "right_ascension_declination"),
    )
    observed = astro.TrackingObservationPlan(stations, schedule).evaluate(
        jnp.asarray([[2.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    )
    assert bool(observed.valid[0])
    np.testing.assert_allclose(observed.predicted[0, 0], 2.0)
    nonfinite_state = jnp.asarray([[2.0, 0.0, 0.0, jnp.nan, 0.0, 0.0]])
    nonfinite = astro.TrackingObservationPlan(stations, schedule).evaluate(
        nonfinite_state
    )
    assert not bool(nonfinite.valid[0])
    assert int(nonfinite.status[0]) == int(astro.AstrodynamicsStatus.NONFINITE_INPUT)

    bad_schedule = astro.ObservationSchedule(
        jnp.asarray([0.0]),
        jnp.asarray([1]),
        jnp.asarray([0]),
        jnp.zeros((1, 2)),
        jnp.eye(2)[None],
        jnp.asarray([False]),
        ("range",),
    )
    with pytest.raises(ValueError, match="absent tracking station"):
        astro.TrackingObservationPlan(stations, bad_schedule)


def test_variational_od_and_mission_closure():
    astro = phx.applications.astrodynamics
    variational = astro.VariationalPropagationPlan(
        lambda t, x, p, args: p[0] * x,
        jnp.linspace(0.0, 0.1, 5),
        jnp.zeros((1, 1)),
        parameter_dimension=1,
    ).propagate(jnp.asarray([1.0]), jnp.asarray([2.0]), jnp.eye(1))
    assert bool(jnp.all(variational.valid))
    np.testing.assert_allclose(variational.states[-1, 0], jnp.exp(0.2), rtol=1e-6)

    od = astro.BatchOrbitDeterminationPlan(
        lambda parameter, args: 2.0 * parameter,
        jnp.asarray([4.0]),
        jnp.eye(1),
    ).solve(jnp.asarray([0.0]))
    assert bool(od.valid)
    np.testing.assert_allclose(od.estimate, jnp.asarray([2.0]), atol=1e-10)

    access = astro.AccessPlan().evaluate(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([2.0, 0.0, 0.0]),
    )
    assert bool(access.visible)


def test_mission_geometry_covariance_and_od_controls_reject_invalid_inputs():
    astro = phx.applications.astrodynamics
    access = astro.AccessPlan().evaluate(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.zeros(3),
        jnp.asarray([2.0, 0.0, 0.0]),
    )
    assert not bool(access.valid)
    assert not bool(access.visible)

    conjunction = astro.ConjunctionPlan(0.1).evaluate(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        jnp.full((3, 3), jnp.nan),
    )
    assert not bool(conjunction.valid)
    assert int(conjunction.status) == int(astro.AstrodynamicsStatus.NONFINITE_INPUT)

    with pytest.raises(TypeError, match="maximum_iterations"):
        astro.BatchOrbitDeterminationPlan(
            lambda parameter, args: parameter,
            jnp.asarray([1.0]),
            jnp.eye(1),
            maximum_iterations=True,
        )
    provider = lambda time, args: jnp.asarray([time, 0.0, 0.0])
    with pytest.raises(TypeError, match="max_iterations"):
        astro.LightTimePlan(
            provider,
            provider,
            provider,
            1.0,
            max_iterations=True,
            transmitter_provider_id="transmitter",
            receiver_provider_id="receiver",
            gravitating_body_provider_id="body",
        )
