import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _artifact(kind):
    return cosmology.ScientificArtifactEnvelope(
        artifact_kind=kind,
        content_digest=f"{kind}-fixture",
        producer="test",
        producer_version="current",
        build_id="fixture",
        license_id="internal",
        resource_id="static",
        status="complete",
    )


def test_low_resolution_sky_tod_mapmaking_and_bandpower_handoff():
    pixel_count = 12
    synthesis = jnp.zeros((3 * pixel_count, 3))
    for pixel in range(pixel_count):
        synthesis = synthesis.at[3 * pixel : 3 * pixel + 3, :].set(jnp.eye(3))
    sky_plan = cosmology.HarmonicSkySynthesisPlan(
        synthesis,
        jnp.diag(jnp.asarray([1.0, 0.5, 0.25])),
        nside=1,
        lmax=2,
        pixelization="HEALPix-RING",
        artifact=_artifact("sht-provider"),
    )
    sky = sky_plan.realize(jax.random.key(1))
    pixels = jnp.repeat(jnp.arange(pixel_count), 4)
    angles = jnp.tile(
        jnp.asarray([0.0, jnp.pi / 4.0, jnp.pi / 2.0, 3.0 * jnp.pi / 4.0]), pixel_count
    )
    pointing = cosmology.CmbPointingProduct(
        pixels,
        angles,
        jnp.zeros_like(pixels, dtype=bool),
        jnp.tile(jnp.arange(4), pixel_count),
        pixel_count=pixel_count,
    )
    beam = cosmology.CmbBeamProduct(30.0 * jnp.pi / (180.0 * 60.0), "150GHz")
    tod = cosmology.CmbTodSimulationPlan(
        pointing,
        beam,
        net_microkelvin_sqrt_second=1.0e-8,
        sample_interval_seconds=0.1,
    ).simulate(sky, jax.random.key(2))
    result = cosmology.CmbMapmakingPlan(pixel_count).solve(tod)
    assert bool(result.successful)
    np.testing.assert_allclose(result.map, sky.iqu, atol=1e-6)

    raw_layout = cosmology.CoordinateLayout(("TT:l2", "EE:l2", "BB:l2"))
    binned_layout = cosmology.CoordinateLayout(("TT:b0", "POL:b0"))
    handoff = cosmology.CmbBandpowerHandoff(
        [1.0, 2.0, 3.0],
        raw_layout,
        [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
        binned_layout,
        "cmb-spectrum-product",
    )
    np.testing.assert_allclose(handoff.binned.values, [1.0, 2.5])
