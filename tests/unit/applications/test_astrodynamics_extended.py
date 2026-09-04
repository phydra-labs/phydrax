import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _context(origin="earth", orientation="icrf"):
    astro = phx.applications.astrodynamics
    return astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0, 0.0), "TT")),
        astro.FrameDefinition(origin, orientation, pseudo_inertial=True),
    )


def _provenance(context):
    astro = phx.applications.astrodynamics
    return astro.AstrodynamicsDataProvenance(
        producer="test",
        producer_version="1",
        source_id="synthetic",
        checksum="sha256:test",
        license_id="test-data",
        frame_id=context.frame.frame_id,
        epoch_id=context.epoch.epoch_id,
        scale_id=context.scale.scale_id,
        differentiability="constant",
    )


def test_lambert_quarter_circle_and_multirevolution_capacity():
    astro = phx.applications.astrodynamics
    context = _context()
    result = astro.solve_lambert(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        0.5 * jnp.pi,
        1.0,
        context,
        astro.LambertPlan(max_revolutions=1),
    )
    assert result.departure_velocity.shape == (3, 3)
    assert bool(result.valid[0])
    np.testing.assert_allclose(
        result.departure_velocity[0], jnp.asarray([0.0, 1.0, 0.0]), atol=2.0e-9
    )
    np.testing.assert_allclose(
        result.arrival_velocity[0], jnp.asarray([-1.0, 0.0, 0.0]), atol=2.0e-9
    )


def test_typed_array_adapter_converts_values_without_changing_context_semantics():
    astro = phx.applications.astrodynamics
    assert astro.AstrodynamicsScaleContract is phx.DimensionalScaleContract
    context = _context("earth", "icrf")
    centimetres_per_second = phx.units.derived_unit(
        "cm/s",
        ((phx.units.CENTIMETER, 1), (phx.units.SECOND, -1)),
    )
    state = astro.cartesian_state_from_scaled_arrays(
        jnp.asarray([100.0, 200.0, 300.0]),
        jnp.asarray([200.0, 0.0, -100.0]),
        context,
        position_unit=phx.units.CENTIMETER,
        velocity_unit=centimetres_per_second,
    )
    np.testing.assert_allclose(state.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.velocity, [2.0, 0.0, -1.0])
    assert state.context.context_id == context.context_id
    assert state.context.frame.frame_id == context.frame.frame_id
    assert state.context.epoch.epoch_id == context.epoch.epoch_id
    assert state.context.epoch.time_scale == "TT"


def test_sgp4_adapter_preserves_teme_utc_context_while_converting_units():
    astro = phx.applications.astrodynamics

    class Satrec:
        @staticmethod
        def sgp4(day, fraction):
            del day, fraction
            return 0, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)

    context = astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(
            astro.TimeInstant(astro.JulianDate(2451545.0), "UTC"),
            continuous=False,
        ),
        astro.FrameDefinition("earth", "TEME", pseudo_inertial=True),
    )
    trajectory = astro.trajectory_from_sgp4(
        Satrec(),
        jnp.asarray([2451545.0, 2451545.0]),
        jnp.asarray([0.0, 1.0 / 86400.0]),
        context,
    )
    np.testing.assert_allclose(trajectory.times, [0.0, 1.0])
    np.testing.assert_allclose(trajectory.states[0, :3], [1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(trajectory.states[0, 3:], [4000.0, 5000.0, 6000.0])
    assert trajectory.context.context_id == context.context_id
    assert trajectory.context.epoch.time_scale == "UTC"
    noninertial = astro.AstrodynamicsContext(
        context.scale,
        context.epoch,
        astro.FrameDefinition("earth", "TEME", pseudo_inertial=False),
    )
    with pytest.raises(ValueError, match="Earth-TEME UTC"):
        astro.trajectory_from_sgp4(
            Satrec(),
            jnp.asarray([2451545.0]),
            jnp.asarray([0.0]),
            noninertial,
        )


def test_time_frame_ephemeris_and_third_body_contracts():
    astro = phx.applications.astrodynamics
    source_context = _context("earth", "icrf")
    target_context = astro.AstrodynamicsContext(
        source_context.scale,
        source_context.epoch,
        astro.FrameDefinition("earth", "rotated", pseudo_inertial=True),
    )
    provenance = _provenance(source_context)
    offset = astro.TimeScaleTransform.tai_to_tt(provenance).apply(jnp.asarray(1.0))
    assert bool(offset.valid)
    np.testing.assert_allclose(offset.relative_seconds, 33.184, atol=1.0e-12)

    evaluator = astro.ConstantKinematicEvaluator(
        jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    transform = astro.KinematicFrameTransform(
        evaluator,
        source_context.frame,
        target_context.frame,
        transform_id="quarter-turn",
    )
    state = astro.CartesianOrbitState(
        jnp.asarray([1.0, 0.0, 0.0]), jnp.zeros(3), source_context
    )
    rotated, evidence = transform.apply(state, 0.0, target_context)
    restored, _ = transform.apply_inverse(rotated, 0.0, source_context)
    assert bool(evidence.valid)
    np.testing.assert_allclose(restored.position, state.position, atol=1.0e-12)

    kilometre_scale = astro.AstrodynamicsScaleContract(
        phx.units.KILOMETER,
        phx.units.KILOGRAM,
        phx.units.SECOND,
    )
    mismatched_target = astro.AstrodynamicsContext(
        kilometre_scale,
        source_context.epoch,
        target_context.frame,
    )
    with pytest.raises(ValueError, match="matching scale contracts"):
        transform.apply(state, 0.0, mismatched_target)
    mismatched_source = astro.AstrodynamicsContext(
        kilometre_scale,
        source_context.epoch,
        source_context.frame,
    )
    with pytest.raises(ValueError, match="matching scale contracts"):
        transform.apply_inverse(rotated, 0.0, mismatched_source)

    catalog = astro.CelestialBodyCatalog(
        ("sun",), jnp.asarray([1.0]), jnp.asarray([1.0]), source_context
    )
    ephemeris = astro.TabulatedEphemeris(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[2.0, 1.0, 0.0, 0.0, 1.0, 0.0]]]),
        catalog,
        provenance,
    )
    sample = ephemeris.evaluate(0.5, 0)
    third_body = astro.ThirdBodyGravity(ephemeris, 0).evaluate(
        0.5, jnp.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert bool(sample.valid)
    assert bool(third_body.valid)


def test_direct_nbody_and_cr3bp_invariants():
    astro = phx.applications.astrodynamics
    context = _context("barycenter")
    particles = phx.discretization.particle.ParticleSetPlan(
        jnp.asarray([0, 1]), jnp.asarray([1.0, 1.0]), ambient_dimension=3
    ).prepare()
    gravity = astro.DirectNBodyGravityPlan(particles, context)
    state = astro.NBodyState(
        jnp.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        jnp.asarray([[0.0, -0.7, 0.0], [0.0, 0.7, 0.0]]),
        particles,
        context,
    )
    evaluation = gravity.evaluate(state.position)
    assert bool(evaluation.valid)
    np.testing.assert_allclose(evaluation.net_internal_force, 0.0, atol=1.0e-14)
    rollout = astro.NBodyPropagationPlan(gravity, jnp.linspace(0.0, 0.1, 5)).rollout(
        state
    )
    assert bool(rollout.successful)
    assert float(jnp.max(jnp.abs(rollout.energy - rollout.energy[0]))) < 1.0e-5
    near_plan = astro.NearlyKeplerianPlan(
        1.0,
        jnp.asarray([1.0e-3]),
        jnp.linspace(0.0, 0.2, 5),
        context,
    )
    near_result = near_plan.rollout(
        astro.NearlyKeplerianState(
            jnp.asarray([[1.0, 0.0, 0.0]]),
            jnp.asarray([[0.0, 1.0, 0.0]]),
            context,
        )
    )
    assert bool(near_result.successful)
    assert bool(jnp.all(near_result.valid))

    cr3bp = astro.CR3BPSystem(0.01)
    points = cr3bp.lagrange_points()
    assert bool(jnp.all(points.valid))
    for point in points.points:
        derivative = cr3bp.vector_field(0.0, jnp.concatenate((point, jnp.zeros(3))))
        np.testing.assert_allclose(derivative, 0.0, atol=2.0e-10)


def test_spacecraft_burn_and_measurement_adapters():
    astro = phx.applications.astrodynamics
    context = _context()
    particles = phx.discretization.particle.ParticleSetPlan(
        jnp.asarray([0]), jnp.asarray([2.0]), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.particle.RigidBodySetPlan(
        jnp.asarray([0]), jnp.eye(3)[None]
    ).prepare(particles)
    kinematics = bodies.kinematics(
        jnp.zeros((1, 3)),
        jnp.zeros((1, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        jnp.zeros((1, 3)),
    )
    burn = astro.FiniteBurnPlan(
        jnp.asarray([1.0, 0.0, 0.0]),
        maximum_thrust=2.0,
        specific_impulse=300.0,
        burn_id="main",
    ).evaluate(kinematics, 0.5)
    assert bool(burn.valid)
    np.testing.assert_allclose(burn.thrust, 1.0)
    mass_state, mass_valid = astro.deplete_propellant(
        astro.VariableMassSpacecraftState(kinematics, jnp.asarray([1.0])),
        burn,
        1.0,
    )
    assert bool(mass_valid)
    np.testing.assert_allclose(
        mass_state.propellant_mass,
        jnp.asarray([1.0]) - burn.mass_flow_rate,
    )

    measurements = astro.OrbitMeasurementPlan(
        "range_rate",
        jnp.asarray([0.0]),
        jnp.zeros((1, 3)),
        jnp.zeros((1, 3)),
        jnp.eye(1),
        context,
        measurement_id="range-rate",
    ).evaluate(jnp.asarray([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]))
    assert bool(measurements.valid[0])
    np.testing.assert_allclose(measurements.predicted[0, 0], 2.0)
    np.testing.assert_allclose(measurements.jacobian[0, 0, 3], 1.0)
