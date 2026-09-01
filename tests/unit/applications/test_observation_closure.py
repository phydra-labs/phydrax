import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_wcs_imaging_and_survey_closure():
    physics = phx.applications.astrophysics
    wcs = physics.TangentSipWcsPlan(
        jnp.asarray([1.0, 0.5]),
        jnp.asarray([100.0, 100.0]),
        jnp.eye(2),
        jnp.zeros((3, 3)),
        jnp.zeros((3, 3)),
    )
    pixel = wcs.world_to_pixel(jnp.asarray([1.001, 0.5005]))
    world = wcs.pixel_to_world(pixel.coordinates)
    assert bool(pixel.valid)
    assert bool(world.valid)
    np.testing.assert_allclose(world.coordinates, jnp.asarray([1.001, 0.5005]), atol=1e-9)

    provenance = physics.ObservationDataProvenance.native("calibration")
    psf = jnp.zeros((3, 3)).at[1, 1].set(1.0)
    response = physics.ImageResponsePlan(psf, response_id="delta")
    calibration = physics.ImagingCalibration(
        jnp.zeros((3, 3)),
        jnp.zeros((3, 3)),
        jnp.ones((3, 3)),
        jnp.ones((3, 3)),
        jnp.zeros((3, 3), dtype=bool),
        jnp.full((3, 3), 100.0),
        provenance,
    )
    image = physics.CalibratedImagingPlan(response, calibration).evaluate(
        jnp.ones((3, 3)), 2.0
    )
    np.testing.assert_allclose(image.expected_adu, 2.0)
    assert bool(image.valid)


def test_radiative_waveform_and_exoplanet_closure():
    physics = phx.applications.astrophysics
    transfer = physics.ScalarRadiativeTransferPlan(jnp.ones(4)).evaluate(
        jnp.ones(4), jnp.zeros(4)
    )
    assert bool(transfer.valid)
    np.testing.assert_allclose(transfer.emergent, 4.0)

    provenance = physics.ObservationDataProvenance.native("qnm")
    modes = physics.QnmModeTable(
        jnp.asarray([1.0]),
        jnp.asarray([2.0]),
        jnp.asarray([[2, 2, 0]]),
        provenance,
    )
    waveform = physics.RingdownPlan(modes).time_domain(
        jnp.asarray([0.0, 1.0]), jnp.asarray([1.0 + 0.0j])
    )
    np.testing.assert_allclose(waveform[0], 1.0)

    oblate = physics.OblateOccultationPlan(radial_order=32, angular_order=64).evaluate(
        jnp.asarray([2.0, 0.0]), 0.1, 0.08, 0.0
    )
    np.testing.assert_allclose(oblate.relative_flux, 1.0, atol=1e-6)
    lens = physics.FiniteSourceMicrolensingPlan(0.01).evaluate(jnp.asarray([1.0, 0.0]))
    assert bool(lens.valid)
    assert float(lens.magnification) > 1.0
