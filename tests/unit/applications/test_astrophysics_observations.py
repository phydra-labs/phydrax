import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _context():
    astro = phx.applications.astrodynamics
    return astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0, 0.0), "TT")),
        astro.FrameDefinition("star", "observer-inertial", pseudo_inertial=True),
    )


def test_projection_and_occultation_limits_and_gradients():
    physics = phx.applications.astrophysics
    projection = physics.ObserverProjectionPlan(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        jnp.asarray([0.0, 0.0, 1.0]),
        _context(),
    )
    projected = projection.project(jnp.asarray([0.3, 0.4, 1.0]))
    np.testing.assert_allclose(projected.projected_separation, 0.5)
    assert bool(projected.foreground)

    disk = physics.PolynomialLimbDarkenedDisk(jnp.asarray([0.3, 0.2]))
    occultation = physics.CircularOccultationPlan(disk, quadrature_order=256)
    disjoint = occultation.evaluate(2.0, 0.1)
    full = occultation.evaluate(0.0, 2.0)
    partial = occultation.evaluate(jnp.asarray([0.2, 0.4]), 0.1)
    np.testing.assert_allclose(disjoint.relative_flux, 1.0)
    np.testing.assert_allclose(full.relative_flux, 0.0)
    assert bool(jnp.all((partial.relative_flux > 0.0) & (partial.relative_flux < 1.0)))

    tangent = jax.jvp(
        lambda separation: occultation.evaluate(separation, 0.1).relative_flux,
        (jnp.asarray(0.5),),
        (jnp.asarray(1.0),),
    )[1]
    assert bool(jnp.isfinite(tangent))


def test_photon_counting_bandpass_and_poisson_composition():
    physics = phx.applications.astrophysics
    wavelength = jnp.asarray([4.0e-7, 5.0e-7, 6.0e-7])
    provenance = physics.ObservationDataProvenance.native("synthetic-filter")
    band = physics.PhotonCountingBandpass(
        wavelength, jnp.ones(3), provenance, band_id="uniform"
    )
    flux = jnp.full((2, 3), 1.0e-9)
    plan = physics.TransitPhotometryPlan(
        (band,),
        jnp.asarray([0, 0]),
        jnp.asarray([10.0, 20.0]),
        collecting_area=2.0,
        background_rate=1.0,
    )
    result = eqx.filter_jit(plan.evaluate)(jnp.asarray([1.0, 0.5]), flux)
    assert bool(jnp.all(result.valid))
    assert bool(jnp.all(result.poisson_supported))
    np.testing.assert_allclose(
        result.expected_counts[1] / 20.0 - 1.0,
        0.5 * (result.expected_counts[0] / 10.0 - 1.0),
        rtol=1.0e-12,
    )
    log_prob = physics.transit_poisson_log_prob(result, jnp.asarray([1.0, 1.0]))
    assert bool(jnp.isfinite(log_prob))


def test_response_signal_ray_and_image_operators_are_composable():
    physics = phx.applications.astrophysics
    binned = physics.BinnedResponsePlan(jnp.eye(2), response_id="identity").evaluate(
        jnp.asarray([1.0, 2.0])
    )
    np.testing.assert_allclose(binned.predicted, jnp.asarray([1.0, 2.0]))

    image_plan = physics.ImageResponsePlan(
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        response_id="delta",
    )
    image = jnp.arange(9.0).reshape(3, 3)
    np.testing.assert_allclose(image_plan.evaluate(image).image, image, atol=1.0e-12)

    ray = physics.RayTransferPlan(jnp.ones((2, 4)), ray_id="constant")
    transferred = ray.evaluate(jnp.ones((2, 4)), jnp.zeros((2, 4)))
    np.testing.assert_allclose(transferred.intensity, 4.0)

    provenance = physics.ObservationDataProvenance.native("signal")
    signal = physics.FrequencyDomainSignal(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([1.0 + 0.0j, 1.0 + 0.0j]),
        jnp.zeros(2, dtype=complex),
        provenance,
        signal_id="wave",
    )
    response = physics.FrequencyResponsePlan(
        jnp.ones(2, dtype=complex),
        jnp.zeros(2, dtype=complex),
        jnp.ones(2, dtype=complex),
        jnp.ones(2),
        frequency_spacing=1.0,
        response_id="detector",
    ).evaluate(signal)
    assert bool(response.valid)
    assert bool(jnp.isfinite(response.log_likelihood))
