import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _context():
    astro = phx.applications.astrodynamics
    return astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0), "TT")),
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
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


def test_time_eop_frame_and_ephemeris_closure(tmp_path):
    astro = phx.applications.astrodynamics
    context = _context()
    provenance = _provenance(context)
    route = astro.PreparedTimeRoute(
        (
            astro.TimeScaleTransform.gps_to_tai(provenance),
            astro.TimeScaleTransform.tai_to_tt(provenance),
        )
    )
    result = route.apply(0.0)
    np.testing.assert_allclose(result.relative_seconds, 51.184)

    eop = astro.PreparedEarthOrientation(
        astro.EarthOrientationRecordSet(
            jnp.asarray([0.0, 86400.0]),
            jnp.zeros(2),
            jnp.zeros(2),
            jnp.zeros(2),
            jnp.zeros(2),
            jnp.zeros(2),
            jnp.zeros(2),
            jnp.zeros(2),
            jnp.asarray([False, False]),
            provenance,
        ),
        2451545.0,
    ).evaluate(0.0)
    assert bool(eop.valid)
    np.testing.assert_allclose(
        eop.rotation_gcrs_to_itrs @ eop.rotation_gcrs_to_itrs.T, jnp.eye(3), atol=1e-12
    )

    catalog = astro.CelestialBodyCatalog(("body",), [1.0], [1.0], context)
    coefficients = jnp.zeros((1, 1, 3, 3))
    coefficients = coefficients.at[0, 0, 0, 0].set(0.5)
    coefficients = coefficients.at[0, 0, 0, 1].set(0.5)
    ephemeris = astro.ChebyshevEphemeris([0.0, 1.0], coefficients, catalog, provenance)
    sample = ephemeris.evaluate(0.25, 0)
    assert bool(sample.valid)
    np.testing.assert_allclose(sample.state.position[0], 0.25, atol=1e-12)
    np.testing.assert_allclose(sample.state.velocity[0], 1.0, atol=1e-12)


def test_gravity_ias15_and_hierarchy_closure():
    astro = phx.applications.astrodynamics
    context = _context()
    provenance = _provenance(context)
    cosine = jnp.zeros((3, 3)).at[0, 0].set(1.0)
    field = astro.SphericalHarmonicGravityField(
        cosine, jnp.zeros_like(cosine), 1.0, 1.0, context, provenance
    )
    gravity = astro.SphericalHarmonicGravity(field)
    evaluated = gravity.evaluate(0.0, jnp.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        evaluated.acceleration, jnp.asarray([-0.25, 0.0, 0.0]), atol=1e-10
    )

    solver = phx.solver.IAS15Plan(relative_tolerance=1e-10, absolute_tolerance=1e-12)
    trajectory = solver.solve(
        lambda t, q, v, args: -q,
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        jnp.linspace(0.0, 1.0, 5),
    )
    assert bool(jnp.all(trajectory.valid))
    np.testing.assert_allclose(trajectory.position[-1, 0], jnp.cos(1.0), atol=2e-7)

    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tree = astro.PreparedOctree3D(positions, jnp.ones(2), leaf_capacity=1)
    hierarchy = astro.BarnesHutGravityPlan3D(tree, jnp.ones(2)).evaluate(positions)
    assert bool(hierarchy.valid)
    np.testing.assert_allclose(hierarchy.acceleration[0, 0], 0.25, atol=1e-12)
    np.testing.assert_allclose(hierarchy.acceleration[1, 0], -0.25, atol=1e-12)
