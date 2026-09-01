import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def test_physical_dependency_projection_checks_only_shared_parameters():
    first = cosmology.CosmologyPhysicalState(
        [70.0, 0.3, 2.1e-9],
        ("hubble_constant", "matter_density", "primordial_amplitude"),
        "scale",
    )
    second = cosmology.CosmologyPhysicalState(
        [70.0, 0.3, 3.0e-9],
        ("hubble_constant", "matter_density", "primordial_amplitude"),
        "scale",
    )
    geometry = cosmology.PhysicalDependencyProjection(
        ("hubble_constant", "matter_density")
    )
    transfer = cosmology.PhysicalDependencyProjection(first.names)
    np.testing.assert_allclose(
        geometry.project(first).require_compatible(
            geometry.project(second), jnp.asarray(1.0)
        ),
        1.0,
    )
    with pytest.raises((ValueError, RuntimeError), match="different physical"):
        jax.block_until_ready(
            transfer.project(first).require_compatible(
                transfer.project(second), jnp.asarray(1.0)
            )
        )
    assert first.content_id() != second.content_id()


def test_artifact_content_identity_and_derivative_contracts():
    artifact = cosmology.ScientificArtifactEnvelope(
        artifact_kind="fixture",
        content_digest="abc123",
        producer="test",
        producer_version="current",
        build_id="build",
        license_id="internal",
        resource_id="resource",
        status="complete",
    )
    assert artifact.artifact_id
    native = cosmology.DifferentiationContract.native()
    constant = cosmology.DifferentiationContract.constant()
    combined = native.meet(constant)
    assert not combined.upstream_physical_parameters
    assert not combined.query_coordinates
    assert combined.local_parameters


def test_shared_observation_and_correlated_gaussian_are_differentiable():
    source = cosmology.CoordinateLayout(("P0:k0", "P2:k0", "P4:k0"))
    target = cosmology.CoordinateLayout(("d0", "d1"))
    matrix = jnp.asarray([[1.0, 0.5, 0.0], [0.0, 0.25, 1.0]])
    observation = cosmology.LinearObservationPlan(matrix, source, target)
    covariance = cosmology.PrecisionCovarianceAction(jnp.eye(2), 0.0, target)
    likelihood = cosmology.CorrelatedGaussianPlan(
        jnp.asarray([2.0, 1.0]), observation, covariance
    )

    def log_probability(amplitude):
        theory = cosmology.TheoryVector(
            amplitude * jnp.asarray([1.0, 2.0, 0.5]), source, "theory-content"
        )
        return likelihood.evaluate(theory).log_probability

    value, derivative = jax.value_and_grad(log_probability)(jnp.asarray(1.0))
    assert jnp.isfinite(value)
    assert jnp.isfinite(derivative)
    expected = observation.apply(
        cosmology.TheoryVector(jnp.asarray([1.0, 2.0, 0.5]), source, "theory-content")
    )
    np.testing.assert_allclose(expected.values, [2.0, 1.0])
