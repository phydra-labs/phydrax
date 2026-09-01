import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def test_desi_release_window_covariance_and_likelihood():
    source = cosmology.CoordinateLayout(("P0:k0", "P2:k0", "P4:k0"))
    observed = cosmology.CoordinateLayout(("d0", "d1"))
    artifact = cosmology.ScientificArtifactEnvelope(
        artifact_kind="survey-release",
        content_digest="desi-fixture",
        producer="DESI",
        producer_version="DR1-v1.5",
        build_id="release",
        license_id="public-release",
        resource_id="static",
        status="complete",
    )
    manifest = cosmology.SurveyReleaseManifest(
        release="DESI-DR1-v1.5",
        tracer="LRG-GCcomb",
        redshift_bin="0.4-0.6",
        statistic="P0-P2-P4",
        fiducial_id="fixture-fiducial",
        scale_cut_id="fixture-cuts",
        covariance_corrections="release-supplied",
        artifact=artifact,
    )
    window = jnp.asarray([[1.0, 0.5, 0.0], [0.0, 0.25, 1.0]])
    data = jnp.asarray([2.0, 1.0])
    release = cosmology.SurveyReleaseProduct(
        source,
        observed,
        data,
        window,
        jnp.eye(2),
        0.0,
        manifest,
    )
    likelihood = cosmology.DesiFullShapeLikelihoodPlan(release)

    def value(amplitude):
        theory = cosmology.TheoryVector(
            amplitude * jnp.asarray([1.0, 2.0, 0.5]),
            source,
            "provider-product",
        )
        return likelihood.evaluate(theory).log_probability

    result = likelihood.evaluate(
        cosmology.TheoryVector(jnp.asarray([1.0, 2.0, 0.5]), source, "provider-product")
    )
    np.testing.assert_allclose(result.residual, 0.0)
    expected = -0.5 * 2.0 * np.log(2.0 * np.pi)
    np.testing.assert_allclose(result.log_probability, expected)
    derivative = jax.grad(value)(jnp.asarray(1.0))
    np.testing.assert_allclose(derivative, 0.0, atol=1e-12)
