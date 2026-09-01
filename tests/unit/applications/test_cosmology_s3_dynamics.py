import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _artifact():
    return cosmology.ScientificArtifactEnvelope(
        artifact_kind="s3-basis",
        content_digest="fixture",
        producer="test",
        producer_version="current",
        build_id="fixture",
        license_id="internal",
        resource_id="static",
        status="complete",
    )


def test_s3_geometry_geodesic_kdk_and_parallel_transport():
    manifold = cosmology.S3ManifoldPlan(2.0)
    q = jnp.asarray([[2.0, 0.0, 0.0, 0.0]])
    tangent = jnp.asarray([[0.0, 0.1, 0.0, 0.0]])
    target = manifold.exponential(q, tangent)
    recovered = manifold.logarithm(q, target)
    np.testing.assert_allclose(recovered, tangent, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(manifold.distance(q, target), 0.1, rtol=1e-10)
    transported = manifold.parallel_transport(q, target, tangent)
    np.testing.assert_allclose(jnp.sum(target * transported, axis=-1), 0.0, atol=1e-12)

    plan = cosmology.S3GeodesicKDKPlan(manifold)
    state = plan.initialize(q, tangent, [1.0], 0.5)
    result = plan.advance(
        state,
        0.6,
        1.0,
        0.0,
        0.0,
        jnp.zeros_like(q),
        jnp.zeros_like(q),
    )
    assert bool(result.successful)
    assert result.norm_defect < 1e-12
    assert result.tangent_defect < 1e-12


def test_s3_harmonic_poisson_and_particle_transfer():
    evaluation = jnp.asarray([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]])
    gradient = jnp.zeros((4, 2, 4)).at[:, 1, 1].set(jnp.asarray([1.0, -1.0, 1.0, -1.0]))
    basis = cosmology.S3HarmonicBasisPlan(
        ((0, 0, 0), (1, 0, 0)),
        evaluation,
        gradient,
        0.25 * jnp.ones((4,)),
        radius=1.0,
        artifact=_artifact(),
    )
    poisson = cosmology.S3PoissonPlan(basis, 1.0)
    solved = poisson.solve(jnp.asarray([1.0, -1.0, 1.0, -1.0]), 1.0, 1.0)
    assert bool(solved.successful)
    np.testing.assert_allclose(solved.potential_coefficients[0], 0.0)

    deposit = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    gather = deposit.T
    transfer = cosmology.S3ParticleMeshPlan(poisson, deposit, gather)
    result = transfer.evaluate([0.5, 0.5], 1.0, 1.0)
    assert bool(result.successful)
    np.testing.assert_allclose(result.mass_defect, 0.0)
