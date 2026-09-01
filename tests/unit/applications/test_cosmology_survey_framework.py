import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _artifact():
    return cosmology.ScientificArtifactEnvelope(
        artifact_kind="survey-fixture",
        content_digest="fixture",
        producer="test",
        producer_version="current",
        build_id="fixture",
        license_id="internal",
        resource_id="static",
        status="complete",
    )


def _coordinate(component, value):
    return cosmology.SurveyCoordinate(
        domain="Fourier",
        statistic="power-multipole",
        observable="galaxy-density",
        fields=("delta-g",),
        tracer_ids=("LRG",),
        selection_ids=("GCcomb",),
        tomographic_bins=(0,),
        component=component,
        coordinate_kind="k",
        coordinate_value=value,
        unit="h/Mpc",
        frame="fiducial-comoving",
        h_convention="h-scaled",
    )


def test_generic_survey_slice_composes_theory_response_and_likelihood():
    coordinates = (_coordinate("P0", 0.1), _coordinate("P2", 0.1))
    theory = cosmology.SurveyTheoryProduct(
        jnp.asarray([2.0, 1.0]), coordinates, "theory-fixture"
    )
    source = theory.as_theory_vector().layout
    observed = cosmology.CoordinateLayout(("d0", "d1"))
    plan = cosmology.SurveyFrameworkPlan(
        source,
        observed,
        jnp.eye(2),
        jnp.asarray([2.0, 1.0]),
        jnp.eye(2),
        0.0,
        cosmology.desi_full_shape_slice(_artifact()),
    )
    result = plan.evaluate(theory)
    assert bool(result.successful)
    np.testing.assert_allclose(result.residual, 0.0)
    derivative = jax.grad(
        lambda amplitude: (
            plan.evaluate(
                cosmology.SurveyTheoryProduct(
                    amplitude * jnp.asarray([2.0, 1.0]),
                    coordinates,
                    "theory-fixture",
                )
            ).log_probability
        )
    )(jnp.asarray(1.0))
    np.testing.assert_allclose(derivative, 0.0, atol=1e-12)


def test_three_vertical_slice_manifests_have_distinct_capabilities():
    artifact = _artifact()
    manifests = (
        cosmology.desi_full_shape_slice(artifact),
        cosmology.spin2_pseudocl_slice(artifact),
        cosmology.joint_survey_slice(artifact),
    )
    assert len({manifest.manifest_id for manifest in manifests}) == 3
    assert all(manifest.capabilities for manifest in manifests)
    assert all(manifest.negative_boundaries for manifest in manifests)
