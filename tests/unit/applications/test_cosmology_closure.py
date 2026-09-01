import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _cosmology_context():
    cosmology = phx.applications.cosmology
    scale = cosmology.CosmologyScaleContract("L", "M", "T")
    background = cosmology.FLRWBackground(1.0, 0.3, scale=scale)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="closure",
        numerical_policy_id="native",
        physics_policy_id="closure",
        scale_id=scale.scale_id,
        source_kind="native",
        differentiation="native-parameter",
    )
    return scale, background, provenance


def test_early_universe_and_boltzmann_closure():
    cosmology = phx.applications.cosmology
    scale, background, provenance = _cosmology_context()
    relic = cosmology.RelicBackgroundPlan(1.0).evaluate(jnp.asarray([0.5, 1.0]))
    assert bool(jnp.all(relic.total_radiation_density > 0.0))

    bbn = cosmology.BbnReactionNetworkPlan(
        jnp.asarray([[-1.0], [1.0]]),
        jnp.asarray([1.0, 1.0]),
        lambda t, abundance, args: jnp.asarray([0.1 * abundance[0]]),
        jnp.linspace(0.0, 1.0, 17),
    ).solve(jnp.asarray([1.0, 0.0]))
    assert bool(jnp.all(bbn.valid))
    np.testing.assert_allclose(bbn.baryon_conservation_error, 0.0, atol=1e-12)

    thermodynamics = cosmology.RecombinationPlan(jnp.geomspace(1.0e-4, 1.0, 128)).build(
        scale, provenance, background.realization
    )
    assert bool(jnp.all(thermodynamics.ionization_fraction >= 0.0))
    np.testing.assert_allclose(
        jnp.trapezoid(thermodynamics.visibility, thermodynamics.scale_factors),
        1.0,
        rtol=1e-6,
    )

    plan = cosmology.EinsteinBoltzmannPlan(
        jnp.linspace(1.0, 2.0, 32),
        jnp.asarray([0.1, 0.2]),
        lambda time: 0.1 / time,
        lambda time: jnp.asarray(0.1),
        maximum_multipole=4,
    )
    initial = jnp.zeros((2, plan.state_dimension)).at[:, 5].set(1.0)
    solved = plan.solve(initial, jnp.ones(2))
    assert bool(solved.valid)
    assert solved.temperature_cl.shape == (5,)


def test_nonlinear_lensing_and_compact_object_closure():
    cosmology = phx.applications.cosmology
    multiplicity = cosmology.HaloMassFunctionPlan().multiplicity(jnp.asarray([0.5, 1.0]))
    assert bool(jnp.all(multiplicity > 0.0))

    convergence, gamma1, gamma2 = cosmology.LensingPlanePlan(1.0).convergence_and_shear(
        jnp.ones((4, 4)), 2.0
    )
    np.testing.assert_allclose(convergence, 0.5)
    assert bool(jnp.all(jnp.isfinite(gamma1)))
    assert bool(jnp.all(jnp.isfinite(gamma2)))

    compact = phx.applications.compact_objects
    pressure = jnp.linspace(1.0e-6, 0.2, 128)
    energy = 1.0 + 2.0 * pressure
    eos = compact.EquationOfStateTable(pressure, energy)
    result = compact.TovPlan(eos, jnp.linspace(1.0e-4, 2.0, 512)).solve(0.1)
    assert bool(result.valid)
    assert float(result.mass) > 0.0
    assert float(result.radius) > 0.0


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
